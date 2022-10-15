'''
This file was created by ]init[ AG 2022.

Module for Speech Service.
'''
from fastapi import FastAPI, Form, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
import logging
from nllb_manager import NllbManager, LANGUAGES
import os
from pathlib import Path
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
import shortuuid
import shutil
import sys
from timeit import default_timer as timer
from typing import Optional, List
from whisper_manager import WhisperManager
import yt_dlp

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI()


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

# Following ordering is important for overlapping path matches...


class LanguagesResponse(BaseModel):
    language_id: str
    language_name: str


@app.get('/languages', response_model=List[LanguagesResponse])
async def languages() -> List[LanguagesResponse]:
    return [LanguagesResponse(language_id=k, language_name=v[0]) for k, v in LANGUAGES.items()]


class IdentityLanguageRequest(BaseModel):
    text: str


class IdentityLanguageResponse(BaseModel):
    language: str


@app.post('/identitfy_language/', response_model=IdentityLanguageResponse)
def identitfy_language(req: IdentityLanguageRequest) -> IdentityLanguageResponse:
    logger.info(f"Identitfy language: {req=}")
    nllb_manager = NllbManager()
    language = nllb_manager.identify_language(req.text)
    return IdentityLanguageResponse(language=language)


class TranslationRequest(BaseModel):
    texts: List[str]
    src_lang: str = None
    tgt_lang: str


class TranslationResponse(BaseModel):
    texts: List[str]
    src_lang: str
    tgt_lang: str


@app.post('/translate/', response_model=TranslationResponse)
def translate(req: TranslationRequest) -> TranslationResponse:
    logger.info(f"Translate: {req=}")
    nllb_manager = NllbManager()
    if req.src_lang != None:
        src_lang = req.src_lang
    else:
        src_lang = nllb_manager.identify_language(req.texts[0])
    texts = nllb_manager.translate(req.texts, src_lang, req.tgt_lang)
    return TranslationResponse(texts=texts, src_lang=src_lang, tgt_lang=req.tgt_lang)


class TranscriptionResponse(BaseModel):
    segments: List[dict]
    src_lang: str
    tgt_lang: str = None


def transcribe(path: Path, tgt_lang: Optional[str] = None) -> TranscriptionResponse:
    whisper_manager = WhisperManager()
    result = whisper_manager.transcribe(str(path))
    # convert whisper language codes into Flores-200 language codes for NLLB
    src_lang = next(k for k, v in LANGUAGES.items()
                    if v[1] == result.language)

    if tgt_lang == src_lang or not tgt_lang in LANGUAGES:
        # target language is fine, nothing to do
        return TranscriptionResponse(segments=result.segments, src_lang=src_lang)

    nllb_manager = NllbManager()
    translated_texts = nllb_manager.translate(
        [s.text for s in result.segments], src_lang, tgt_lang)
    for s, t in zip(result.segments, translated_texts):
        s.text = t

    return TranscriptionResponse(segments=result.segments, src_lang=src_lang, tgt_lang=tgt_lang)


@app.post('/transcribe_upload/', response_model=TranscriptionResponse)
def transcribe_upload(file: UploadFile, tgt_lang: Optional[str] = Form(None)) -> TranscriptionResponse:
    logger.info(f"Transcribe Upload: {file.filename}")
    start = timer()

    filepath = save_upload_file_tmp(file)
    try:
        result = transcribe(filepath, tgt_lang)
    finally:
        filepath.unlink()  # Delete the temp file

    logger.info(
        f"Transcribed Upload {file.filename!r} in {timer() - start:.3f}s")
    return result


class TranscriptionRequest(BaseModel):
    url: str
    tgt_lang: str = None


@app.post('/transcribe_download/', response_model=TranscriptionResponse)
def transcribe_download(req: TranscriptionRequest) -> TranscriptionResponse:
    logger.info(f"Transcribe Download: {req.url}")
    start = timer()
    # see https://github.com/ytdl-org/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312

    filepath = None

    def download_hook(e):
        nonlocal filepath
        if e['status'] == 'downloading':
            pass  # logger.info(f"    ...downloading... {e=}")
        elif e['status'] == 'finished':
            logger.info(f"    ...downloading finished: {e['filename']}")
            filepath = Path(e['filename'])
        elif e['status'] == 'error':
            logger.info(f"    ...download error: {e=}")

    logger.info(f"  Downloading {req.url!r}...")
    ydl_opts = {
        # %{title}
        'outtmpl': f"/opt/speech_service/uploads/%(id)s{shortuuid.uuid()}.%(ext)s",
        'format': 'bestaudio/best',
        'progress_hooks': [download_hook]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([req.url])
    logger.info(f"  ...downloaded {req.url!r} as {filepath!r}...")

    try:
        result = transcribe(filepath, req.tgt_lang)
    finally:
        filepath.unlink()  # Delete the temp file

    logger.info(
        f"Transcribed Download {req.url!r} in {timer() - start:.3f}s")
    return result


# see https://github.com/deepgram-devs/live-transcription-fastapi
@app.websocket("/transcribe_stream/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            await websocket.send_text('TEST')
    except Exception as e:
        raise Exception(f'Could not process audio: {e}')
    finally:
        await websocket.close()

app.mount('/', StaticFiles(directory='resources/html' if os.path.exists('resources')
          else 'speech_service/resources/html', html=True), name='static')


def main():
    import uvicorn
    # just 1 worker, or models will be loaded multiple times!
    uvicorn.run('speech_service:app', host='0.0.0.0', port=8200)


if __name__ == '__main__':
    main()
