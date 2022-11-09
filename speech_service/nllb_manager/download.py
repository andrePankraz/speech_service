'''
This file was created by ]init[ AG 2022.

Module for Downloading with caching.
'''
from datetime import datetime
from email.utils import parsedate_to_datetime, formatdate
import os
import requests
import shutil


def download(url, destination_file):
    headers = {}

    if os.path.exists(destination_file):
        mtime = os.path.getmtime(destination_file)
        headers["if-modified-since"] = formatdate(mtime, usegmt=True)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=10) as r:
            r.raise_for_status()
            if r.status_code == requests.codes.not_modified:
                return
            if r.status_code == requests.codes.ok:
                with open(destination_file, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                if last_modified := r.headers.get("last-modified"):
                    new_mtime = parsedate_to_datetime(last_modified).timestamp()
                    os.utime(destination_file, times=(datetime.now().timestamp(), new_mtime))
    except requests.exceptions.RequestException as err:
        if not os.path.exists(destination_file):
            raise SystemExit(err)


if __name__ == '__main__':
    model_folder = os.environ.get('MODEL_FOLDER', '/opt/speech_service/models/')
    download("https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin",
             model_folder + "lid218e.bin")
