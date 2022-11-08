'''
This file was created by ]init[ AG 2022.

Module for Model "Whisper".
'''
import logging
import numpy as np
import os
from pydantic import BaseModel
import sys
import threading
from timeit import default_timer as timer
import torch
import whisper

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class WhisperSegment(BaseModel):
    id: int
    start: int
    end: int
    text: str


class WhisperResult(BaseModel):
    language: str | None
    segments: list[WhisperSegment]


class WhisperManager:

    lock = threading.Lock()

    def __new__(cls):
        # Singleton!
        if not hasattr(cls, 'instance'):
            cls.instance = super(WhisperManager, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if hasattr(self, 'model'):
            return
        with WhisperManager.lock:
            # Load model Whisper
            # see _MODELS: tiny (40M), base (80M), small (250M), medium (800M), large (1.5B)
            # model card: https://github.com/openai/whisper/blob/main/model-card.md
            # 4 GB VRAM not enough for combining small & NLLB 600
            model_name = 'small'
            device = 'cpu'
            if torch.cuda.is_available():
                log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                mem_info = torch.cuda.mem_get_info(0)
                vram = round(mem_info[1] / 1024**3, 1)
                log.info(f"VRAM available: {round(mem_info[0]/1024**3,1)} GB out of {vram} GB")
                if (vram >= 4):
                    device = 'cuda:0'
                    model_name = 'large' if vram >= 32 else 'medium' if vram >= 12 else 'small' if vram >= 8 else 'base'
            model_folder = os.environ.get('MODEL_FOLDER', '/opt/speech_service/models/')
            log.info(f"Loading model {model_name!r} in folder {model_folder!r}...")

            self.model = whisper.load_model(model_name, device=device, download_root=model_folder)

            log.info("...done.")
            if device != 'cpu':
                log.info(f"VRAM left: {round(torch.cuda.mem_get_info(0)[0]/1024**3,1)} GB")

    def transcribe(self, audio: str | np.ndarray | torch.Tensor | bytes, src_lang: str | None = None) -> WhisperResult:
        log.info(f"Transcribing...")
        start = timer()

        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, np.float32).flatten()
        # Whisper can also directly translate via parameter: task='translate'
        # but quality is much worse than NLLB
        with WhisperManager.lock:
            results = self.model.transcribe(audio, language=src_lang)
        log.info(f"...done in {timer() - start:.3f}s")
        segments = [WhisperSegment(id=s['id'], start=s['start'], end=s['end'], text=s['text'].strip())
                    for s in results['segments']]  # type: ignore
        return WhisperResult(language=results['language'], segments=segments)   # type: ignore

    def test(self) -> None:
        log.info(f"Result: {self.transcribe('uploads/ivan_8848_1280x720_1578090984604303362.mp4')}")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    whisper_manager = WhisperManager()
    whisper_manager.test()


if __name__ == '__main__':
    main()
