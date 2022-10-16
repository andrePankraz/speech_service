'''
This file was created by ]init[ AG 2022.

Tests for LASER Sentence splitter.
See https://github.com/facebookresearch/LASER/blob/main/utils/src/cleaner_splitter.py
'''
import logging
from sentence_cleaner_splitter.cleaner_splitter import SentenceSplitClean

log = logging.getLogger(__name__)


def test_split():
    splitter = SentenceSplitClean('deu_Latn', 'default')
    for line_hash, sent, clean in splitter('Das ist ein Test.\u200bNoch ein Test. Und noch einer...muhahahaha - das ist schwer! Ich sage ja: "Nein"! Das wird nix.\n1\n234-567\nWhatever'):
        log.info(f"{line_hash}  - {sent} - {clean}")
