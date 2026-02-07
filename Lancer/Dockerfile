FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all files first
COPY . .

# Install dependencies (lightweight reranking with CPU torch)
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.32.0 \
    pydantic>=2.10.0 \
    pydantic-settings>=2.6.0 \
    httpx>=0.28.0 \
    aiohttp>=3.11.0 \
    python-dotenv>=1.0.0 \
    tenacity>=9.0.0 \
    slowapi>=0.1.9 \
    numpy \
    sentence-transformers \
    e2b-desktop>=2.2.0 \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
