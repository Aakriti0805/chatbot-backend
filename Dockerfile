FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ghostscript \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --timeout=60 --retries=3 -r requirements.txt

COPY huggingface_cache /root/.cache/huggingface
COPY paddlex_cache /root/.paddlex

COPY . .

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]