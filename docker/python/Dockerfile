#FROM python:3.10-bullseye without GPU
#FROM nvcr.io/nvidia/pytorch:22.09-py3 with GPU / CUDA / PyTorch support (>15 GB!)
FROM nvcr.io/nvidia/pytorch:22.09-py3 AS base
VOLUME /tmp
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --assume-yes \
      ffmpeg
# more packages: end with "\" and add another a line
WORKDIR /opt/speech_service
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

FROM base AS dev
COPY requirements-dev.txt requirements-dev.txt
RUN pip3 install -r requirements-dev.txt

FROM base AS local
WORKDIR /opt/speech_service/speech_service
CMD ["uvicorn", "speech_service:app", "--host", "0.0.0.0", "--port", "8200"]
