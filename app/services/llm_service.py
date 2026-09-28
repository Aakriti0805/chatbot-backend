import os
import re
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

import json
from google import genai
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment")

client = genai.Client(api_key=api_key, vertexai=False)

GEMINI_MODEL = "gemini-3.6-flash"


class DocumentMetadata(BaseModel):
    dept: Optional[str] = Field(None, description="Department, e.g. Mechanical, Electrical, Civil")
    sub_dept: Optional[str] = Field(None, description="Specific sub-department, if mentioned")
    subject: Optional[str] = Field(None, description="The main topic of the document, in one short line")
    doc_date: Optional[str] = Field(None, description="The document's date, in YYYY-MM-DD format if available")


class QueryAnswer(BaseModel):
    answer: str = Field(description="The answer text, or exactly 'NOT_FOUND' if the context doesn't answer the question")
    refs_used: list[int] = Field(
        default_factory=list,
        description=(
            "REF numbers (the [REF n] markers shown in the context) actually "
            "used to construct the answer; empty if NOT_FOUND"
        ),
    )


class StandaloneQuery(BaseModel):
    standalone_query: str = Field(description="The follow-up question rewritten as a complete, standalone question that makes sense without the earlier conversation")


class ExpandedQuery(BaseModel):
    expanded_query: str = Field(description="The query rewritten with additional relevant keywords and context, for better embedding-based retrieval")


def extract_metadata(text: str) -> dict:
    """
    Extracts dept, sub_dept, subject, and doc_date from the document's text.
    """
    if not text:
        return {"dept": None, "sub_dept": None, "subject": None, "doc_date": None}

    prompt = f"Read the document text below and extract the requested fields:\n\n{text[:2000]}"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": DocumentMetadata,
            },
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {"dept": None, "sub_dept": None, "subject": None, "doc_date": None}


# Words/patterns that signal a follow-up question is leaning on the prior
# conversation (a pronoun, a demonstrative, or a phrase like "the previous
# one") — used by rewrite_query_with_history to decide whether the extra
# LLM call is actually needed. This is deliberately conservative (a small,
# clearly-referential list) — the cost of a false negative here (skipping
# the rewrite when it was actually needed) is a possibly-wrong retrieval
# for one turn, which is an acceptable trade for skipping the call on every
# question that plainly doesn't need it.
_REFERENCE_PATTERN = re.compile(
    r"\b(it|its|that|this|those|these|them|they|he|she|his|her|"
    r"previous|earlier|above|last one|prior|aforementioned|"
    r"same|again|also|too|either|neither|"
    r"unke|unka|unki|uska|uski|uske|iske|iska|iski|"
    r"wo|woh|inhe|unhe|ye\s+sab|pichla|pichle|pichli)\b",
    re.IGNORECASE,
)


def _looks_like_followup(query: str) -> bool:
    """
    Cheap heuristic (no LLM call) for whether `query` is likely leaning on
    the prior conversation for meaning. Checks for pronouns/demonstrative
    references in both English and common Hinglish forms, since users of
    this chatbot mix both. A very short query (a few words) is also
    treated as a possible follow-up ("list them", "how many?") even
    without a pattern match, since short queries are the ones most likely
    to be incomplete on their own.
    """
    stripped = query.strip()
    if not stripped:
        return False
    if _REFERENCE_PATTERN.search(stripped):
        return True
    word_count = len(stripped.split())
    return word_count <= 4


def rewrite_query_with_history(query: str, history: list[dict]) -> str:
    """
    If conversation history exists AND the new question looks like it's
    referencing that history (pronouns, demonstratives, or a very short
    follow-up), rewrites it into a complete, standalone question — so that
    embedding-based retrieval and SQL generation both have something
    concrete to work with.

    If there's no history yet, OR the question already reads as
    standalone (no reference words, reasonable length), this skips the
    LLM call entirely and returns the original query unchanged. This is
    the common case (a fresh, well-formed question) and previously paid
    for an LLM round trip on every single question regardless.
    """
    if not history:
        return query

    if not _looks_like_followup(query):
        return query

    history_text = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}" for turn in history
    )

    prompt = f"""
    This is a conversation with a railway maintenance chatbot. Below is the
    prior conversation followed by the user's new follow-up question.

    Prior conversation:
    {history_text}

    New follow-up question: {query}

    Rewrite this follow-up question as a complete, standalone question with
    no pronouns or unclear references (e.g. "it", "that", "the previous
    one") — everything should be explicitly stated. If the follow-up
    question is already standalone (no connection to the prior
    conversation), return it unchanged.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": StandaloneQuery,
            },
        )
        return json.loads(response.text)["standalone_query"]
    except Exception as e:
        logger.warning(f"Query rewriting failed, falling back to original query: {e}")
        return query


def expand_query_for_retrieval(query: str) -> str:
    """
    Short/vague queries (e.g. "how to login into irfmm") embed too
    generically — a general-purpose embedding model can rank the actually
    correct chunk far down the list when the document is full of similar
    UI-instruction text ("click on...", "enter details...", etc).

    This expands the query with related keywords/context BEFORE embedding
    search, purely to make retrieval more specific. It does NOT change what
    question is actually answered — generate_answer() still uses the
    original query, so the final answer's wording is unaffected.

    A query that's already reasonably long/specific gets no benefit from
    this (per the reasoning above, it exists for short/vague queries), so
    the LLM call is skipped for anything past a small word-count threshold
    — this is the common case for questions that already name a specific
    entity, code, or topic.

    Falls back to the original query if expansion fails for any reason,
    so a Gemini hiccup here never blocks retrieval entirely.
    """
    if not query or not query.strip():
        return query

    if len(query.split()) > 12:
        return query

    prompt = f"""
    This is a query that will be used for embedding-based semantic search
    over a railway maintenance manual (covers portal logins, wagon
    maintenance, defect recording, module masters, etc).

    Original query: {query}

    Rewrite it into a slightly longer, keyword-rich version that would
    match relevant document text more precisely — add likely related terms
    (e.g. "URL", "portal", "credentials" for a login query) without
    changing what is actually being asked. Keep it to one or two sentences.
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ExpandedQuery,
            },
        )
        return json.loads(response.text)["expanded_query"]
    except Exception as e:
        logger.warning(f"Query expansion failed, falling back to original query: {e}")
        return query


def _chunk_context_block(index: int, chunk) -> str:
    """
    Builds one [REF n] context block for the prompt. Works for both Chunk
    (uploaded PDFs/images — has page_number) and MasterDataChunk (database
    master tables — has source_table + record_id, no page_number) objects,
    without the caller needing to know which type it is.
    """
    if hasattr(chunk, "page_number"):
        origin = f"document page {chunk.page_number}"
    else:
        origin = f"{chunk.source_table} record {chunk.record_id}"
    return f"[REF {index} — {origin}]: {chunk.text}"


def generate_answer(query: str, chunks: list, history: list[dict] | None = None) -> dict:
    """
    Takes the user's query and a list of chunks (Chunk and/or
    MasterDataChunk objects), and asks Gemini to generate an answer using
    only that context, along with which [REF n] blocks it actually used.

    Using REF numbers (instead of raw page numbers) lets this function work
    uniformly whether `chunks` came from uploaded documents, database
    master tables, or a mix of both — the caller maps refs_used back to the
    original chunk objects to build proper source citations.

    `history` (optional) is passed in only for continuity/tone — e.g. if
    the user asked a follow-up like "tell me more" — it does not affect
    retrieval, which already happened via rewrite_query_with_history().
    """
    context_text = "\n\n".join(
        _chunk_context_block(idx, c) for idx, c in enumerate(chunks, start=1)
    )

    history_block = ""
    if history:
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in history
        )
        history_block = f"""
    Prior conversation (for continuity only — do not copy answers directly
    from it, use it only to understand context):
    {history_text}
    """

    prompt = f"""
    You are a helpful assistant for railway maintenance staff.
    Answer the user's question using ONLY the context provided below.
    {history_block}
    If the context does not contain enough information to answer the
    question, set "answer" to exactly "NOT_FOUND" and "refs_used" to an
    empty list.

    Otherwise, answer the question and list every [REF n] number in
    "refs_used" whose content you actually relied on to build the answer.
    If the same fact appears in multiple REFs, include all of them.

    Do not guess or make up information.

    Context:
    {context_text}

    Question: {query}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QueryAnswer,
            },
        )
        return json.loads(response.text)
    except Exception as e:
        logger.warning(f"generate_answer JSON parsing/generation failed: {e}")
        return {"answer": "NOT_FOUND", "refs_used": []}