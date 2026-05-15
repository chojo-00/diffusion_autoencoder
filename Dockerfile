FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure pip to use Kakao mirror (faster in Korea)
RUN printf "%s\n"\
    "[global]"\
    "index-url=https://mirror.kakao.com/pypi/simple/"\
    "extra-index-url=https://pypi.org/simple/"\
    "trusted-host=mirror.kakao.com"\
    > /etc/pip.conf

# Install system packages
RUN apt update -qq && apt install -qqy\
        sudo\
        tzdata\
        vim\
        curl\
        jq\
        git\
        libgl1-mesa-glx\
        libglib2.0-0\
    && apt clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install all Python packages in one layer (Duplicates removed)
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
    accelerate==0.27.2 \
    clean-fid==0.1.35 \
    click \
    colored-traceback \
    easydict \
    gdown==4.6.0 \
    imageio \
    ipdb \
    ipywidgets \
    jupyter \
    jupyter_contrib_nbextensions \
    jupyterlab==4.2.3 \
    lmdb \
    matplotlib \
    natsort \
    nibabel \
    numpy==1.24.4 \
    opencv-contrib-python-headless \
    opencv-python==4.5.5.64 \
    opencv-python-headless==4.5.5.64 \
    openpyxl \
    pandas \
    Pillow \
    prefetch_generator \
    psutil \
    pydicom==2.1.0 \
    pylibjpeg==1.3.0 \
    pylibjpeg-libjpeg==1.3.0 \
    pylibjpeg-openjpeg==1.2.0 \
    pylibjpeg-rle==1.2.0 \
    requests \
    rich \
    scikit-image \
    scikit-learn \
    scipy \
    tensorboard \
    termcolor \
    torch-ema \
    tqdm \
    wandb==0.15.4