'''
This file was created by ]init[ AG 2022.

Tests for NllbManager.
'''
import logging
from nllb_manager import NllbManager

log = logging.getLogger(__name__)


def test_translate():
    nllbManager = NllbManager()
    src_texts = ['There’s much more to the tiny mosquito than itchy bites and ruined holidays – and, as it seems they’re here to stay, we’d better learn to live with them.\u200b']
    src_lang = nllbManager.identify_language(src_texts[0])
    tgt_texts = nllbManager.translate(src_texts, src_lang, 'deu_Latn')
    log.info(f"Result from {src_lang=}: {tgt_texts}")
