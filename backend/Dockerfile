FROM python:3.11-slim


ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HOME=/root/.cache/huggingface

# 3. Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# 4. Safely download Hugging Face models using the CLI (No PyTorch memory overhead)
RUN pip install --upgrade huggingface_hub protobuf sentencepiece && \
    python -c "import gc; \
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification; \
print('Downloading GPT-2...'); \
AutoTokenizer.from_pretrained('gpt2'); \
m1 = AutoModelForCausalLM.from_pretrained('gpt2'); \
del m1; gc.collect(); \
print('Downloading DeBERTa-v3-small...'); \
AutoTokenizer.from_pretrained('microsoft/deberta-v3-small'); \
m2 = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-small'); \
del m2; gc.collect(); \
print('Models successfully cached!')"

# Create shared volume directory for ELA/FFT heatmaps
RUN mkdir -p /tmp/aletheia_artifacts

COPY . .

EXPOSE 8000