import fasttext
import logging
import os
from pydantic import BaseModel
import sys
from timeit import default_timer as timer
from typing import List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NllbResult(BaseModel):
    text: str = None


class NllbManager:

    def __new__(cls) -> type:
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(NllbManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'model'):
            return

        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"VRAM available: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB out of {round(torch.cuda.mem_get_info(0)[1]/1024**3,1)} GB")
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # facebook/nllb-200-distilled-600M, facebook/nllb-200-distilled-1.3B, facebook/nllb-200-3.3B
        # VRAM at least: 4 | 8 | 16 GB VRAM
        model_name = 'facebook/nllb-200-distilled-600M'
        model_folder = os.environ.get('MODEL_FOLDER')
        logger.info(
            f"Loading model {model_name!r} in folder {model_folder!r}...")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=model_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_folder)

        # Load 2nd model for Language Identification (LID):
        # https://github.com/facebookresearch/fairseq/tree/nllb#lid-model
        # wget https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin
        self.lid_model = fasttext.load_model(
            os.environ.get('MODEL_FOLDER') + "/lid218e.bin")

        logger.info("...done.")
        if self.device != 'cpu':
            logger.info(
                f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB out of {round(torch.cuda.mem_get_info(0)[1]/1024**3,1)} GB")

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        logger.info(f"Translating {texts=}...")
        start = timer()
        translation_pipeline = pipeline('translation',
                                        model=self.model,
                                        tokenizer=self.tokenizer,
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang,
                                        device=self.device)
        norm_texts = list(filter(lambda t: len(t), map(
            lambda t: t.replace('\u200b', ' ').strip(), texts)))
        results = translation_pipeline(norm_texts)
        logger.info(f"...done in {timer() - start:.3f}s\n  {results=}")
        return [result.get('translation_text') for result in results]

    def identify_language(self, text: str) -> str:
        # (('__label__deu_Latn',), array([1.00001001]))
        prediction = self.lid_model.predict(text)
        return prediction[0][0][len('__label__'):]

    def test(self) -> None:
        texts = ['Essential to the food chain\u200b', 'And there’s more to mosquitoes than pollination: if they weren’t around, our ecosystem would change entirely. When just one species disappears, it almost always has a knock-on effect. As with the rest of the animal kingdom, mosquitoes are an essential part of the food chain; they’re an important food source for many fish, reptiles and birds. \u200b', '', 'Indeed, they’re so crucial to many bird species in the Arctic tundra that these birds travel to mosquito-heavy regions every year to eat the insects that hatch there during summer. At this time of year, these regions have the highest concentration of mosquitoes on the planet.\u200b', '',
                 'Mosquitoes also make up a large part of the diet of certain fish species, especially the aptly named mosquito fish. These can consume thousands of mosquito larvae a day. And let’s not forget the bats, frogs, dragonflies, birds and other fish that rely on mosquitoes as food, too.\u200b', '', 'If mosquitoes were to disappear, the animals that eat them might stop living in or visiting certain areas. So, while we relax in our gardens, thinking about how we’d like to get rid of these annoyances once and for all, I’m afraid it’s not that simple. There’s much more to the tiny mosquito than itchy bites and ruined holidays – and, as it seems they’re here to stay, we’d better learn to live with them.\u200b']
        src_lang = self.identify_language(texts[0])
        logger.info(
            f"Result from {src_lang=}: {self.translate(texts, src_lang, 'deu_Latn')}")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    whisper_manager = NllbManager()
    whisper_manager.test()


if __name__ == '__main__':
    main()
