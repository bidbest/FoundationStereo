FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV OPENCV_IO_ENABLE_OPENEXR=1

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    scikit-image \
    omegaconf \
    opencv-contrib-python \
    imgaug \
    ninja \
    timm \
    albumentations \
    jupyterlab \
    scipy \
    joblib \
    scikit-learn \
    ruamel.yaml \
    trimesh \
    pyyaml \
    imageio \
    open3d \
    transformations \
    einops \
    gdown \
    huggingface-hub \
    pandas \
    tqdm

RUN pip install flash-attn --no-build-isolation

RUN pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124

WORKDIR /3dgs_pipe/thirdparty/FoundationStereo
