'''
This file was created by ]init[ AG 2022.

Tests for NllbManager.
'''
import logging
from nllb_manager import NllbManager

log = logging.getLogger(__name__)


def test_translate():
    nllbManager = NllbManager()
    src_texts = [
        'Essential to the food chain​',
        '',
        'And there’s more to mosquitoes than pollination: if they weren’t around, our ecosystem would change entirely. When just one species disappears, it almost always has a knock-on effect. As with the rest of the animal kingdom, mosquitoes are an essential part of the food chain; they’re an important food source for many fish, reptiles and birds.',
        '\u200b',
        'There’s much more to the tiny mosquito than itchy bites and ruined holidays – and, as it seems they’re here to stay, we’d better learn to live with them.\u200b']
    src_lang = nllbManager.identify_language(src_texts[0])
    tgt_texts = nllbManager.translate(src_texts, src_lang, 'deu_Latn')
    log.info(f"Result from {src_lang=}: {tgt_texts}")
