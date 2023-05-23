FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ARG PYTHON=python3.8

SHELL ["/bin/bash", "-c"]

ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libgomp1 \
    libopencv-dev \
    openssh-client \
    openssh-server \
    wget \
    vim \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev

## Install python3.8
RUN apt-get update
RUN apt-get --yes install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get --yes install ${PYTHON}
RUN apt-get --yes install ${PYTHON}-distutils
RUN apt-get --yes install ${PYTHON}-dev
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

## Install pip for python3.8
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip
RUN pip --no-cache-dir install --upgrade pip setuptools

RUN apt-get install -y jq

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:/usr/local/cuda-11.0/lib:/usr/lib64/openmpi/lib/:/usr/local/lib:/usr/lib:/usr/local/mpi/lib:/lib/:" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN pip install --no-cache --upgrade \
    transformers==4.18.0 \
    numpy==1.23.0 \
    torch==1.11.0 \
    PyYAML==6.0 \
    regex==2022.6.2 \
    tqdm==4.64.0 \
    tensorboardX==2.5.1 \
    sentencepiece==0.1.96 \
    nltk==3.7 \
    pytorch-metric-learning==1.5.0 \
    sentence-transformers==2.2.2 \
    sacrebleu==2.1.0 \
    allennlp==2.9.3 \
    cached-path==1.1.2 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.2.0/en_core_web_md-3.2.0.tar.gz#egg=en_core_web_md

WORKDIR /

SHELL ["/bin/bash", "-c"]