import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import threading
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)

# CPU par ek hi model instance saare requests share karta hai. Concurrent
# .encode() calls ko serialize karte hain taaki resource contention se
# corruption ya crash na ho — production mein iska asli fix ek dedicated
# GPU-backed embedding service ya request-batching queue hoga, ye lock
# sirf CPU par safe rehne ke liye hai, throughput ka permanent solution nahi.
_encode_lock = threading.Lock()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    with _encode_lock:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    return embeddings.tolist()


def generate_query_embedding(query: str) -> list[float]:
    with _encode_lock:
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
    return embedding[0].tolist()