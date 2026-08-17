FROM python:3.11-slim

# Build arguments for authenticated Hugging Face downloads
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# 1. Install system dependencies required for OpenCV, C++ compilation, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python project dependencies and build requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade huggingface_hub protobuf sentencepiece pybind11

# 3. Pre-download NLTK tokenizers
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 4. Pre-cache Hugging Face models into Docker layer (Memory-safe download)
RUN python -c "import gc, os; \
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification; \
token = os.getenv('HF_TOKEN', None); \
print('Downloading GPT-2...'); \
AutoTokenizer.from_pretrained('gpt2', token=token); \
m1 = AutoModelForCausalLM.from_pretrained('gpt2', token=token); \
del m1; gc.collect(); \
print('Downloading DeBERTa-v3-small...'); \
AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', token=token); \
m2 = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', token=token); \
del m2; gc.collect(); \
print('Models successfully cached!')"

# 5. Copy C++ source files & compile the Pybind11 extension
COPY setup.py graph_segmenter.cpp ./
RUN python setup.py build_ext --inplace

# 6. Create directories for artifacts
RUN mkdir -p /tmp/aitrace_artifacts /tmp/aletheia_artifacts

# 7. Copy the rest of the application codebase
COPY . .

EXPOSE 8000