# Base image with CUDA 11.8 and Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

 # Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    g++ \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY app.py .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# Use python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Download FastText language ID model
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O /app/lid.176.ftz

# Pre-download nltk 'punkt' data
RUN python -c "import nltk; nltk.download('punkt')"

# Preload EasyNMT model to cache it (optional, saves cold-start delay)
RUN python -c "from easynmt import EasyNMT; EasyNMT('opus-mt')"

# RunPod requires ENTRYPOINT to be set to runpod
ENTRYPOINT ["python", "app.py"]