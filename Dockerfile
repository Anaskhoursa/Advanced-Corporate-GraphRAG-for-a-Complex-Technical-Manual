# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

# System deps:
#   build-essential  -> some wheels still compile
#   libgomp1         -> torch / faiss runtime
#   curl             -> healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download embedding models so the first query isn't a 200MB download
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && python -c "\
from fastembed import SparseTextEmbedding; \
SparseTextEmbedding(model_name='Qdrant/bm25')"

# Copy application code + data
COPY run_assistant.py      ./
COPY retrieval.py       ./
COPY import_neo4j.py             ./
COPY index_qdrant.py          ./
COPY entities.json      ./
COPY relations.json     ./
COPY steps.json         ./
COPY warnings.json      ./
COPY all_chunks.json    ./

# Default: run the interactive assistant
CMD ["python", "run_assistant.py"]