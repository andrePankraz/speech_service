'''
This file was created by ]init[ AG 2022.

Module for Model "No Language Left Behind" (NLLB).
'''
import fasttext
import logging
from nllb_manager.download import download
import os
from pydantic import BaseModel
from sentence_cleaner_splitter.cleaner_splitter import SentenceSplitClean
import threading
from timeit import default_timer as timer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class NllbResult(BaseModel):
    text: str


class NllbManager:

    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(NllbManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'sentence_splitters'):
            return
        with NllbManager.lock:
            models_folder = os.environ.get('MODELS_FOLDER', '/opt/speech_service/models/')
            # Load model NLLB
            model_id = 'facebook/nllb-200-distilled-600M'
            device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    device = 'cuda:0'
                    model_id = 'facebook/nllb-200-3.3B' if vram >= 32 else 'facebook/nllb-200-distilled-1.3B' if vram >= 12 else 'facebook/nllb-200-distilled-600M'
                    # facebook/nllb-200-distilled-600M, facebook/nllb-200-distilled-1.3B, facebook/nllb-200-3.3B
                    # VRAM at least: 4 | 8 | 16 GB VRAM

            log.info(f"Loading model {model_id!r} in folder {models_folder!r}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=models_folder)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=models_folder).to(device)
            log.info("...done.")
            if device != 'cpu':
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

            # Load model for Language Identification (LID)
            # https://github.com/facebookresearch/fairseq/tree/nllb#lid-model
            download("https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin",
                     models_folder + "lid218e.bin")
            self.lid_model = fasttext.load_model(models_folder + "lid218e.bin")

            # Init cache for sentence splitters
            self.sentence_splitters: dict[str, SentenceSplitClean] = {}

    def identify_language(self, text: str) -> str:
        # (('__label__deu_Latn',), array([1.00001001]))
        prediction = self.lid_model.predict(text)
        return prediction[0][0][len('__label__'):]  # type: ignore

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        log.info(f"Translating {texts=}...")
        start = timer()
        # The model is trained on sentence level and problems are possible with longer sequences. So it's a good idea to split the text.
        # See paper: 'The maximum sequence length during training is 512 for both the encoder and the decoder.'
        # The model is restricted to 512 tokens, which means less than 500 words! Text documents should be split into sentences.
        # NLLB used and extended the sentence embedding model LASER, which also contains a sentence splitter as utils.
        # see sentence_cleaner_splitter@git+https://github.com/facebookresearch/LASER.git#subdirectory=utils
        # This is a small AI-model, which will be downloaded in the background.
        # TODO Sentence split might not be enough and a further split on token level might be becessary.
        # Check for further splitting:
        # https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
        if src_lang not in self.sentence_splitters:
            self.sentence_splitters[src_lang] = SentenceSplitClean(src_lang, 'default')
        sentence_splitter = self.sentence_splitters[src_lang]

        # Split into sentences and remember original indices
        norm_texts = []
        norm_indices = []
        for i, text in enumerate(texts):
            for _, _, line in sentence_splitter(text.replace('\u200b', ' ')):
                if line.strip():
                    norm_texts.append(line)
                    norm_indices.append(i)

        # Init pipeline - seems to be thread save, but is bound to specific language pairs, don't cache...
        translation_pipeline = pipeline('translation',
                                        self.model,
                                        tokenizer=self.tokenizer,
                                        device=self.model.device,
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang,
                                        max_length=512,
                                        batch_size=16,
                                        # beam search with early stopping, better than greedy:
                                        num_beams=3,
                                        early_stopping=True)
        res_texts = translation_pipeline(norm_texts)

        # "Unsplit" sentences with remembered original indices
        tgt_texts = [''] * len(texts)
        for i, t in zip(norm_indices, res_texts):  # type: ignore
            tgt_texts[i] += t['translation_text'] + ' '

        log.info(f"...done in {timer() - start:.3f}s\n  {res_texts=}")
        return tgt_texts
