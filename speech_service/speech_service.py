'''
This file was created by ]init[ AG 2022.

Module for Speech Service.
'''
import asyncio
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, Form, HTTPException, status, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
import logging
from nllb_manager import NllbManager, LANGUAGES
import os
from pathlib import Path
from pydantic import BaseModel
import shortuuid
import shutil
import sys
from tempfile import NamedTemporaryFile
from timeit import default_timer as timer
from whisper_manager import WhisperManager, WhisperResult, WhisperSegment
import yt_dlp

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def _identitfy_language(text: str) -> str:
    nllb_manager = NllbManager()
    return nllb_manager.identify_language(text)


def _translate(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    nllb_manager = NllbManager()
    return nllb_manager.translate(texts, src_lang, tgt_lang)


def _transcribe(audio: str | bytes, src_lang: str | None = None) -> WhisperResult:
    whisper_manager = WhisperManager()
    return whisper_manager.transcribe(audio, src_lang)


def _lang_iso_2_flores(language: str) -> str | None:
    # Convert whisper language codes into Flores-200 language codes for NLLB
    try:
        return next(k for k, v in LANGUAGES.items()
                    if v[1] == language)
    except StopIteration:
        log.warn(f"Cannot convert ISO language {language!r} to flores!")
        return None


app = FastAPI()


@app.on_event("startup")
def startup_event():
    log.info("Startup...")
    # Push long running GPU task into specialized single worker processes - one worker per GPU model
    global nllb_executor, whisper_executor
    nllb_executor = ProcessPoolExecutor(max_workers=1)
    whisper_executor = ProcessPoolExecutor(max_workers=1)
    app.mount('/', StaticFiles(directory='resources/html', html=True), name='static')


@app.on_event("shutdown")
def shutdown_event():
    log.info("Shutting down...")
    nllb_executor.shutdown()
    whisper_executor.shutdown()

# Following ordering is important for overlapping path matches...


class LanguagesResponse(BaseModel):
    language_id: str
    language_name: str


@app.get('/languages', response_model=list[LanguagesResponse])
async def languages() -> list[LanguagesResponse]:
    return [LanguagesResponse(language_id=k, language_name=v[0]) for k, v in LANGUAGES.items()]


class IdentityLanguageRequest(BaseModel):
    text: str


class IdentityLanguageResponse(BaseModel):
    language: str


@app.post('/identitfy_language/', response_model=IdentityLanguageResponse)
async def identitfy_language(req: IdentityLanguageRequest) -> IdentityLanguageResponse:
    log.info(f"Identitfy language: {req=}")
    start = timer()

    # Push long running task into specialized NLLB single worker process
    language = await asyncio.get_event_loop().run_in_executor(nllb_executor, _identitfy_language, req.text)

    log.info(
        f"Identitfied language in {timer() - start:.3f}s")
    return IdentityLanguageResponse(language=language)


class TranslationRequest(BaseModel):
    texts: list[str]
    src_lang: str | None
    tgt_lang: str


class TranslationResponse(BaseModel):
    texts: list[str]
    src_lang: str
    tgt_lang: str


@app.post('/translate/', response_model=TranslationResponse)
async def translate(req: TranslationRequest) -> TranslationResponse:
    log.info(f"Translate")
    start = timer()

    if req.src_lang is not None:
        src_lang = req.src_lang
    else:
        src_lang = await asyncio.get_event_loop().run_in_executor(nllb_executor, _identitfy_language, req.texts[0])
    # Push long running task into specialized NLLB single worker process
    translated_texts = await asyncio.get_event_loop().run_in_executor(nllb_executor, _translate, req.texts, src_lang, req.tgt_lang)

    log.info(
        f"Translated in {timer() - start:.3f}s")
    return TranslationResponse(texts=translated_texts, src_lang=src_lang, tgt_lang=req.tgt_lang)


class TranscriptionResponse(BaseModel):
    segments: list[WhisperSegment]
    src_lang: str | None
    tgt_lang: str | None = None


async def transcribe(path: Path, tgt_lang: str | None = None) -> TranscriptionResponse:
    # Push long running task into specialized Whisper single worker process
    whisper_result = await asyncio.get_event_loop().run_in_executor(
        whisper_executor, _transcribe, str(path))
    src_lang = _lang_iso_2_flores(whisper_result.language) if whisper_result.language else None
    if src_lang is None or src_lang == tgt_lang or tgt_lang not in LANGUAGES:
        # Target language is fine, nothing to do
        return TranscriptionResponse(segments=whisper_result.segments, src_lang=src_lang)
    # Push long running task into specialized NLLB single worker process
    translated_texts = await asyncio.get_event_loop().run_in_executor(nllb_executor, _translate,
                                                                      [s.text for s in whisper_result.segments], src_lang, tgt_lang)
    for s, t in zip(whisper_result.segments, translated_texts):
        s.text = t
    return TranscriptionResponse(segments=whisper_result.segments, src_lang=src_lang, tgt_lang=tgt_lang)


@app.post('/transcribe_upload/', response_model=TranscriptionResponse)
async def transcribe_upload(file: UploadFile, tgt_lang: str | None = Form(None)) -> TranscriptionResponse:
    log.info(f"Transcribe Upload: {file.filename}")
    start = timer()

    filepath = save_upload_file_tmp(file)
    try:
        result = await transcribe(filepath, tgt_lang)
    finally:
        filepath.unlink()  # Delete the temp file

    log.info(
        f"Transcribed Upload {file.filename!r} in {timer() - start:.3f}s")
    return result


class TranscriptionRequest(BaseModel):
    url: str
    tgt_lang: str | None


@app.post('/transcribe_download/', response_model=TranscriptionResponse)
async def transcribe_download(req: TranscriptionRequest) -> TranscriptionResponse:
    log.info(f"Transcribe Download: {req.url}")
    start = timer()

    # see
    # https://github.com/ytdl-org/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312
    filepath = None

    def download_hook(e):
        nonlocal filepath
        if e['status'] == 'downloading':
            pass  # logger.info(f"    ...downloading... {e=}")
        elif e['status'] == 'finished':
            log.info(f"    ...downloading finished: {e['filename']}")
            filepath = Path(e['filename'])
        elif e['status'] == 'error':
            log.info(f"    ...download error: {e=}")
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f'Could not download {req.url!r}: {e}')

    log.info(f"  Downloading {req.url!r}...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'max_filesize': 5000000,
        'outtmpl': f"/opt/speech_service/uploads/%(id)s{shortuuid.uuid()}.%(ext)s",
        'progress_hooks': [download_hook]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([req.url])
    if not filepath:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f'Could not download {req.url!r}!')
    log.info(f"  ...downloaded {req.url!r} as {filepath!r}...")
    try:
        result = await transcribe(filepath, req.tgt_lang)
    finally:
        filepath.unlink()  # Delete the temp file

    log.info(
        f"Transcribed Download {req.url!r} in {timer() - start:.3f}s")
    return result


@app.websocket("/transcribe_record/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            # Push long running task into specialized Whisper single worker process
            whisper_result = await asyncio.get_event_loop().run_in_executor(
                whisper_executor, _transcribe, data)
            if whisper_result.language and whisper_result.segments:
                src_lang = _lang_iso_2_flores(whisper_result.language)
                whisper_result.language = src_lang

                await websocket.send_json(whisper_result.dict())
    except Exception as e:
        raise Exception(f'Could not process audio: {e}')
    finally:
        await websocket.close()


def main():
    import uvicorn
    os.chdir('speech_service')
    # just 1 worker, or models will be loaded multiple times!
    uvicorn.run('speech_service:app', host='0.0.0.0', port=8200)


if __name__ == '__main__':
    main()
