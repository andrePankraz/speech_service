'''
This file was created by ]init[ AG 2022.

Tests for WhisperManager.
'''
import logging
from whisper_manager import WhisperManager

log = logging.getLogger(__name__)


def test_transcribe():
    log.debug("TEST")
    whisperManager = WhisperManager()
    log.info(
        f"Result: {whisperManager.transcribe('uploads/Test.m4a')}")
