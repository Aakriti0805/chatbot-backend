import logging
from app.services.embedding_service import generate_embeddings
from app.models import Chunk
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(query: str, db: Session, top_k: int = 5, exclude_ids: list = None):
    query_embedding = generate_embeddings([query])[0]

    q = db.query(Chunk)

    if exclude_ids:
        q = q.filter(~Chunk.id.in_(exclude_ids))

    results = (
        q.order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
        .all()
    )

    logger.debug(f"Retrieved {len(results)} chunks for query: {query!r}")
    for i, chunk in enumerate(results, start=1):
        logger.debug(
            f"Chunk {i} | ID: {chunk.id} | Page: {chunk.page_number} | "
            f"Preview: {chunk.text[:200]!r}"
        )

    return results