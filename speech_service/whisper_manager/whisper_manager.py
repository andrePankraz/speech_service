'''
This file was created by ]init[ AG 2022.

Module for Model "Whisper".
'''
import logging
import os
from pydantic import BaseModel
import sys
import threading
from timeit import default_timer as timer
import torch
from typing import List
import whisper

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class WhisperSegment(BaseModel):
    id: int = None
    start: int = None
    end: int = None
    text: str = None


class WhisperResult(BaseModel):
    language: str = None
    segments: List[WhisperSegment] = None


class WhisperManager:

    lock = threading.Lock()

    def __new__(cls) -> type:
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(WhisperManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'model'):
            return
        with WhisperManager.lock:
            # Load model Whisper
            # see _MODELS: tiny, base, small, medium, large
            # model card: https://github.com/openai/whisper/blob/main/model-card.md
            # 4 GB VRAM not enough for combining small & NLLB 600
            model_name = 'small'
            self.device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1]/1024**3, 1)
                log.info(
                    f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    self.device = 'cuda:0'
                    model_name = 'large' if vram >= 16 else 'medium' if vram >= 10 else 'small' if vram >= 8 else 'base'
            model_folder = os.environ.get('MODEL_FOLDER')
            log.info(
                f"Loading model {model_name!r} in folder {model_folder!r}...")

            self.model = whisper.load_model(
                model_name, device=self.device, download_root=model_folder)

            log.info("...done.")
            if self.device != 'cpu':
                log.info(
                    f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def transcribe(self, filename: str) -> WhisperResult:
        log.info(f"Transcribing {filename =}...")
        start = timer()
        # Whisper can also directly translate via parameter: task='translate'
        # but quality is much worse than NLLB
        with WhisperManager.lock:
            results = self.model.transcribe(filename)
        log.info(f"...done in {timer() - start:.3f}s")
        segments = [WhisperSegment(
            id=s['id'], start=s['start'], end=s['end'], text=s['text'].strip()) for s in results['segments']]
        return WhisperResult(language=results['language'], segments=segments)

    def test(self) -> None:
        log.info(
            f"Result: {self.transcribe('uploads/ivan_8848_1280x720_1578090984604303362.mp4')}")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    whisper_manager = WhisperManager()
    whisper_manager.test()


if __name__ == '__main__':
    main()
