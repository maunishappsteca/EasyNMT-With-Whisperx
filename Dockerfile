FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download FastText language ID model
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O /app/lid.176.ftz

# Pre-download nltk 'punkt' data
RUN python -c "import nltk; nltk.download('punkt')"

# Preload EasyNMT model to cache it (optional, saves cold-start delay)
RUN python -c "from easynmt import EasyNMT; EasyNMT('opus-mt')"

# Copy all code to container
COPY . .

# RunPod requires ENTRYPOINT to be set to runpod
ENTRYPOINT ["python", "-m", "runpod"]
