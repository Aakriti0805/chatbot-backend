import logging
import math
import re
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import Chunk
from app.services.embedding_service import generate_query_embedding

logger = logging.getLogger(__name__)

# Common English words (including question words) that carry no identifying
# signal and would otherwise be treated as keywords just because they start
# a sentence and get capitalized ("What", "Which", "Is"...).
_STOPWORDS = {
    "is", "a", "an", "the", "of", "in", "on", "for", "to", "what", "who",
    "are", "and", "or", "does", "do", "has", "have", "with", "this", "that",
    "was", "were", "be", "it", "its", "as", "at", "by", "from", "which",
    "where", "when", "how", "fall", "falls", "under", "use", "uses", "used",
    "located",
}


def _extract_keywords(text: str) -> list[str]:
    """
    Pulls out identifying tokens from a query — proper nouns and codes
    (e.g. "Sitarampur", "LIKA", "BOXNHSM2", "Gujarat") — for literal
    keyword matching.

    Deliberately excludes plain lowercase words such as "district",
    "state", "brake", "type", "wagon": a token only counts as a keyword if
    it's capitalized, is written in all caps, or contains a digit — the
    pattern typical of an entity name or a code the user actually cares
    about, not generic question wording.
    """
    words = re.findall(r"[A-Za-z0-9]+", text)
    keywords = []
    for w in words:
        if len(w) < 3 or w.lower() in _STOPWORDS:
            continue
        if w[0].isupper() or w.isupper() or any(ch.isdigit() for ch in w):
            keywords.append(w)
    return keywords


def _keyword_boost(db: Session, exclude_ids: list[int], keywords: list[str], limit: int = 3):
    """
    Finds chunks whose text contains most of the query's keywords, via a
    plain case-insensitive substring search — independent of embedding
    similarity. This exists because embedding distance alone can bury a
    chunk that is an exact or near-exact literal match.

    The caller must pass keywords taken from the ORIGINAL (pre-expansion)
    query, not the expanded one; extra generic terms from the expansion
    would otherwise make exact matches fail.

    Matching at least half of the keywords (rounded up) is enough, so a
    small wording difference between the query and the stored text does not
    break the match.
    """
    if not keywords:
        return []

    conditions = [Chunk.text.ilike(f"%{kw}%") for kw in keywords]
    query = db.query(Chunk).filter(or_(*conditions))
    if exclude_ids:
        query = query.filter(~Chunk.id.in_(exclude_ids))

    candidates = query.limit(1000).all()

    required_hits = max(1, math.ceil(len(keywords) / 2))
    scored = []
    for chunk in candidates:
        chunk_text = (chunk.text or "").lower()
        hits = sum(1 for kw in keywords if kw.lower() in chunk_text)
        if hits >= required_hits:
            scored.append((chunk, hits))

    scored.sort(key=lambda pair: -pair[1])
    return [chunk for chunk, _hits in scored[:limit]]


def retrieve_relevant_chunks(
    query: str,
    db: Session,
    top_k: int = 5,
    exclude_ids: Optional[list[int]] = None,
    keyword_query: Optional[str] = None,
):
    """
    Retrieves chunks of uploaded documents (PDF / image / DOCX) with a hybrid
    approach: keyword-matched chunks first (their slots are guaranteed), and
    the remaining slots are filled by vector similarity.

    `query` is used for the embedding search (an expanded query is fine).
    `keyword_query`, if given, is used for literal keyword matching — pass
    the original question here. If omitted, `query` is used.

    `exclude_ids` holds the ids of chunks already seen in earlier rounds, so
    they are not returned again.
    """
    query_embedding = generate_query_embedding(query)
    keywords = _extract_keywords(keyword_query or query)
    excluded = list(exclude_ids or [])

    boosted = _keyword_boost(db, excluded, keywords)
    boosted_ids = [c.id for c in boosted]

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    vector_query = db.query(Chunk, distance)
    if excluded:
        vector_query = vector_query.filter(~Chunk.id.in_(excluded))
    if boosted_ids:
        vector_query = vector_query.filter(~Chunk.id.in_(boosted_ids))
    vector_results = vector_query.order_by(distance).limit(top_k).all()

    remaining_slots = max(0, top_k - len(boosted))
    vector_picks = [chunk for chunk, _distance in vector_results[:remaining_slots]]

    combined = boosted + vector_picks

    logger.debug(
        "keywords=%r -> %d keyword-boosted + %d vector-ranked = %d chunks for query: %r",
        keywords, len(boosted), len(vector_picks), len(combined), query,
    )

    return combined