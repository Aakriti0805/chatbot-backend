import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
import threading
import time

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1024
ENCODE_BATCH_SIZE = 32

model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)

# A single model instance on the CPU is shared by every request. Concurrent
# .encode() calls are serialized so resource contention cannot corrupt
# results or crash the process. The real production fix would be a dedicated
# GPU-backed embedding service or a request-batching queue; this lock only
# keeps the CPU setup safe and is not a permanent throughput solution.
_encode_lock = threading.Lock()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding input must contain only non-empty text strings")

    started_at = time.perf_counter()
    logger.info("Generating embeddings for %d text chunks", len(texts))
    try:
        with _encode_lock:
            embeddings = model.encode(
                texts,
                batch_size=ENCODE_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 1,
            )
        vectors = embeddings.tolist()
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: received {len(vectors)} vectors "
                f"for {len(texts)} texts"
            )
        if any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise RuntimeError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings"
            )
    except Exception:
        logger.exception("Embedding generation failed for %d text chunks", len(texts))
        raise

    logger.info(
        "Generated %d embeddings in %.2f seconds",
        len(vectors),
        time.perf_counter() - started_at,
    )
    return vectors


def generate_query_embedding(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("Query text must not be empty")

    with _encode_lock:
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
    vector = embedding[0].tolist()
    if len(vector) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Expected a {EMBEDDING_DIMENSION}-dimensional query embedding, "
            f"received {len(vector)}"
        )
    return vector