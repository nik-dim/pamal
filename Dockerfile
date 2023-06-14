FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    git \
    ffmpeg

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt




# To test:
# 1- build the Dockerfile (e.g. docker build -t pamal .)
# 2- login to the docker container (e.g. docker run -it --gpus all pamal bash)
