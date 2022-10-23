#FROM python:3.10-bullseye without GPU
#FROM nvcr.io/nvidia/pytorch:22.09-py3 with GPU / CUDA / PyTorch support (>15 GB!)
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS base
VOLUME /tmp
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
  apt-get install -y git ffmpeg python3-pip
# Must explicitely install PyTorch for CUDA 11:
# 'Installed CUDA version 11.8 does not match the version torch was compiled with 10.2, unable to compile cuda/cpp extensions without a matching cuda version.'
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install --pre \
  torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117
WORKDIR /opt/speech_service
COPY requirements.txt requirements.txt
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install -r requirements.txt

FROM base AS dev
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install -r requirements-dev.txt

FROM base AS local
WORKDIR /opt/speech_service/speech_service
CMD ["uvicorn", "speech_service:app", "--host", "0.0.0.0", "--port", "8200"]
