import logging
import math
import re
from typing import Literal, Optional
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.services.embedding_service import generate_query_embedding
from app.models import Chunk, MasterDataChunk

logger = logging.getLogger(__name__)

# "documents"   -> only uploaded PDF/image chunks (Chunk table)
# "master_data" -> only structured database master-table chunks (MasterDataChunk table)
# "both"        -> content from both, merged by similarity
QuerySource = Literal["documents", "master_data", "both"]

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
    "state", "brake", "type", "wagon": these are also the literal column
    names in every stored chunk (e.g. every row's text contains
    "district_name - ..." whether or not it's the relevant row), so
    matching on them would make almost every row in a large table look
    like a hit. A token only counts as a keyword if it's capitalized, is
    written in all caps, or contains a digit — the pattern typical of an
    entity name or a code the user actually cares about, not generic
    question wording.
    """
    words = re.findall(r"[A-Za-z0-9]+", text)
    keywords = []
    for w in words:
        if len(w) < 3 or w.lower() in _STOPWORDS:
            continue
        if w[0].isupper() or w.isupper() or any(ch.isdigit() for ch in w):
            keywords.append(w)
    return keywords


def _keyword_boost(db: Session, model, exclude_ids: list[int], keywords: list[str], limit: int = 3):
    """
    Finds rows whose text contains most of the query's keywords, via a
    plain case-insensitive substring search — independent of embedding
    similarity. This exists because embedding distance alone can bury a
    chunk that is an exact or near-exact literal match, if it sits among
    many other chunks that share the same boilerplate structure.

    The caller is expected to pass keywords extracted from the ORIGINAL
    (pre-expansion) query, not a retrieval-expanded version of it — an
    expanded query adds extra generic terms (meant to help embedding
    search) that a short, specific record's text usually won't contain,
    which would otherwise make this match fail even for an exact hit.

    Requiring only about half the keywords (not all) keeps this tolerant
    of small wording differences between the query and the stored text.
    """
    if not keywords:
        return []

    conditions = [model.text.ilike(f"%{kw}%") for kw in keywords]
    query = db.query(model).filter(or_(*conditions))
    if exclude_ids:
        query = query.filter(~model.id.in_(exclude_ids))

    # Now that keywords are specific identifying tokens rather than generic
    # words, matching candidates should be few even in a large table — but
    # the cap is kept generous as a safety net against an unusually common
    # keyword slipping through.
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
    exclude_ids: Optional[list] = None,
    source: QuerySource = "both",
    keyword_query: Optional[str] = None,
):
    """
    Retrieves document chunks (PDFs/images) and/or structured master-data
    chunks (owner master, wagon type master, org master), depending on the
    caller's chosen `source`.

    `query` is used for the embedding search — this is typically the
    retrieval-expanded query, which is fine (even helpful) for vector
    similarity. `keyword_query` (if given) is used instead for literal
    keyword matching — pass the original/standalone question here, since
    an expanded query's extra terms would otherwise make exact-name
    matches fail (see _keyword_boost). Defaults to `query` if omitted.

    exclude_ids treats two ID spaces as one — Chunk and MasterDataChunk IDs
    come from different tables, so a string prefix is used to avoid
    collisions (e.g. "doc:5", "master:12").

    Retrieval is hybrid: each table's top_k vector-similarity matches are
    combined with a small number of keyword-matched chunks, so a literal
    entity-name match is never lost purely because the table it lives in
    is large and full of similar-looking boilerplate text. Keyword-boosted
    chunks are guaranteed a spot in the final result; the remaining slots
    are filled by vector similarity, ranked across both tables when
    source="both".
    """
    query_embedding = generate_query_embedding(query)
    keywords = _extract_keywords(keyword_query or query)

    excluded_doc_ids = []
    excluded_master_ids = []
    if exclude_ids:
        for eid in exclude_ids:
            if isinstance(eid, str) and eid.startswith("master:"):
                excluded_master_ids.append(int(eid.split(":")[1]))
            else:
                excluded_doc_ids.append(int(eid.split(":")[1]) if isinstance(eid, str) else eid)

    doc_boosted = []
    doc_vector_results = []
    if source in ("documents", "both"):
        doc_boosted = _keyword_boost(db, Chunk, excluded_doc_ids, keywords)
        boosted_ids = [c.id for c in doc_boosted]

        doc_query = db.query(
            Chunk, Chunk.embedding.cosine_distance(query_embedding).label("distance")
        )
        if excluded_doc_ids:
            doc_query = doc_query.filter(~Chunk.id.in_(excluded_doc_ids))
        if boosted_ids:
            doc_query = doc_query.filter(~Chunk.id.in_(boosted_ids))
        doc_vector_results = doc_query.order_by("distance").limit(top_k).all()

    master_boosted = []
    master_vector_results = []
    if source in ("master_data", "both"):
        master_boosted = _keyword_boost(db, MasterDataChunk, excluded_master_ids, keywords)
        boosted_ids = [c.id for c in master_boosted]

        master_query = db.query(
            MasterDataChunk,
            MasterDataChunk.embedding.cosine_distance(query_embedding).label("distance"),
        )
        if excluded_master_ids:
            master_query = master_query.filter(~MasterDataChunk.id.in_(excluded_master_ids))
        if boosted_ids:
            master_query = master_query.filter(~MasterDataChunk.id.in_(boosted_ids))
        master_vector_results = master_query.order_by("distance").limit(top_k).all()

    # Keyword-boosted chunks always make it in. The remaining slots (if any)
    # are filled by merging both tables' vector results, ranked by actual
    # cosine distance so neither table is unfairly favored just by order.
    boosted = doc_boosted + master_boosted
    remaining_slots = max(0, top_k - len(boosted))

    vector_with_distance = list(doc_vector_results) + list(master_vector_results)
    vector_with_distance.sort(key=lambda pair: pair[1])
    vector_picks = [obj for obj, _distance in vector_with_distance[:remaining_slots]]

    combined = boosted + vector_picks

    logger.debug(
        f"[source={source}] keywords={keywords!r} -> "
        f"{len(boosted)} keyword-boosted + {len(vector_picks)} vector-ranked "
        f"= {len(combined)} chunks for query: {query!r}"
    )

    return combined