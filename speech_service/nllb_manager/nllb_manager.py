'''
This file was created by ]init[ AG 2022.

Module for Model "No Language Left Behind" (NLLB).
'''
from nllb_manager.download import download
import fasttext
import logging
import os
from pydantic import BaseModel
from sentence_cleaner_splitter.cleaner_splitter import SentenceSplitClean
import threading
from timeit import default_timer as timer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Dict, List
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class NllbResult(BaseModel):
    text: str = None


class NllbManager:

    lock = threading.Lock()

    def __new__(cls) -> type:
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(NllbManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'sentence_splitters'):
            return
        with NllbManager.lock:
            # Init cache for sentence splitters
            self.sentence_splitters: Dict[str, SentenceSplitClean] = {}

            # Load model for Language Identification (LID)
            # https://github.com/facebookresearch/fairseq/tree/nllb#lid-model
            download("https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin",
                     os.environ.get('MODEL_FOLDER') + "/lid218e.bin")
            self.lid_model = fasttext.load_model(
                os.environ.get('MODEL_FOLDER') + "/lid218e.bin")

            # Load model NLLB
            # facebook/nllb-200-distilled-600M, facebook/nllb-200-distilled-1.3B, facebook/nllb-200-3.3B
            # VRAM at least: 4 | 8 | 16 GB VRAM
            model_name = 'facebook/nllb-200-distilled-600M'
            self.device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1]/1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram > 4):
                    self.device = 'cuda:0'
                    model_name = 'facebook/nllb-200-3.3B' if vram >= 16 else 'facebook/nllb-200-distilled-1.3B' if vram >= 8 else 'facebook/nllb-200-distilled-600M'
            model_folder = os.environ.get('MODEL_FOLDER')
            log.info(
                f"Loading model {model_name!r} in folder {model_folder!r}...")

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=model_folder).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=model_folder)

            log.info("...done.")
            if self.device != 'cpu':
                log.info(
                    f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def identify_language(self, text: str) -> str:
        # (('__label__deu_Latn',), array([1.00001001]))
        prediction = self.lid_model.predict(text)
        return prediction[0][0][len('__label__'):]

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        log.info(f"Translating {texts=}...")
        start = timer()

        # Text documents must be split into sentences. The model is restricted to 512 tokens, which means less than 500 words!
        # NLLB used and extended the sentence embedding model LASER, which also contains a sentence splitter as utils.
        # see sentence_cleaner_splitter@git+https://github.com/facebookresearch/LASER.git#subdirectory=utils
        # TODO Sentence split might not be enough and a further split on token level might be becessary.
        # Check for further splitting: https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
        if src_lang not in self.sentence_splitters:
            self.sentence_splitters[src_lang] = SentenceSplitClean(
                src_lang, 'default')
        sentence_splitter = self.sentence_splitters[src_lang]
        norm_texts = []
        for text in texts:
            for _, _, line in sentence_splitter(text.replace('\u200b', ' ')):
                norm_texts.append(line.strip())
        # Init pipeline - seems to be thread save
        translation_pipeline = pipeline('translation',
                                        model=self.model,
                                        tokenizer=self.tokenizer,
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang,
                                        device=self.device)
#        norm_texts = list(filter(lambda t: len(t), map(
#            lambda t: t.replace('\u200b', ' ').strip(), texts)))
        results = translation_pipeline(norm_texts)
        log.info(f"...done in {timer() - start:.3f}s\n  {results=}")
        return [result.get('translation_text') for result in results]