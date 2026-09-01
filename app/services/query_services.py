import logging
from sqlalchemy.orm import Session
from app.services.retrieval_services import retrieve_relevant_chunks
from app.services.llm_service import generate_answer

logger = logging.getLogger(__name__)

MAX_ROUNDS = 4
CHUNKS_PER_ROUND = 5


def answer_query(query: str, db: Session, top_k: int = CHUNKS_PER_ROUND) -> dict:
    """
    Iteratively retrieves chunks and asks the LLM to answer, expanding the
    search with new chunks each round if the answer isn't found yet.
    Stops as soon as an answer is found, or after MAX_ROUNDS if not.
    """
    seen_chunk_ids: list[int] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        chunks = retrieve_relevant_chunks(
            query, db, top_k=top_k, exclude_ids=seen_chunk_ids
        )

        if not chunks:
            logger.debug(f"Round {round_num}: no more chunks left to search.")
            break

        seen_chunk_ids.extend([c.id for c in chunks])

        try:
            result = generate_answer(query, chunks)
        except Exception as e:
            logger.warning(f"Round {round_num}: generation failed — {e}")
            continue

        logger.debug(f"Round {round_num} raw LLM response: {result!r}")

        if result["answer"].strip().upper() != "NOT_FOUND":
            used_pages = set(result.get("pages_used", []))
            return {
                "answer": result["answer"],
                "sources": [
                    {"document_id": c.document_id, "page_number": c.page_number}
                    for c in chunks
                    if c.page_number in used_pages
                ],
                "rounds_used": round_num,
            }

    return {
        "answer": "I could not find a clear answer to this question in the uploaded documents.",
        "sources": [],
        "rounds_used": MAX_ROUNDS,
    }