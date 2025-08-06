FROM python:3.10-slim

# Install system packages
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download FastText language ID model
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O /app/lid.176.ftz

# Download nltk data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy all files (code)
COPY . .

# RunPod requires ENTRYPOINT to be set to runpod
ENTRYPOINT ["python", "-m", "runpod"]
