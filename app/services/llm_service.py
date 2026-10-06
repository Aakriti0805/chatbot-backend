import json
import logging
import os
import re  # NEW: used by the follow-up detection pattern
import socket
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

from google import genai  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Without a timeout, a blocked network can leave a request hanging for a very
# long time. 60 seconds is generous for these prompts.
GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "60000"))

client = genai.Client(
    api_key=api_key,
    vertexai=False,
    http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
)


class LLMUnavailableError(RuntimeError):
    """
    Raised when Gemini cannot be used: the server cannot reach it (DNS,
    firewall, proxy, timeout) or the API itself returned an error.

    This is deliberately different from "the answer was not found". Callers
    let it propagate so the API returns a 503 with a clear message, instead
    of a misleading "could not find an answer".
    """


_NETWORK_ERRORS = (socket.gaierror, ConnectionError, TimeoutError, httpx.TransportError)


def _is_network_error(exc: BaseException) -> bool:
    """True if `exc` or anything in its cause chain is a network-level error."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _NETWORK_ERRORS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _generate(prompt: str, config: dict):
    """
    The single place that calls Gemini. Converts network failures and API
    errors into LLMUnavailableError; any other exception is re-raised as is.
    """
    # Automatic function calling is not used anywhere in this project; turning
    # it off also removes the noisy "AFC is enabled" log lines.
    full_config = {**config, "automatic_function_calling": {"disable": True}}
    try:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=full_config,
        )
    except genai_errors.APIError as exc:
        logger.error("Gemini API error %s: %s", exc.code, exc.message)
        raise LLMUnavailableError(f"Gemini API error {exc.code}: {exc.message}") from exc
    except Exception as exc:
        if _is_network_error(exc):
            logger.error("Cannot reach the Gemini API: %s", exc)
            raise LLMUnavailableError(
                "Cannot reach the Gemini API from this server "
                "(DNS / network / proxy / timeout). Check the server's outbound "
                "internet access."
            ) from exc
        raise


def generate_json(prompt: str, schema: type[BaseModel]) -> dict:
    """Asks Gemini for a JSON answer that follows `schema` and returns it as a dict."""
    response = _generate(
        prompt,
        {"response_mime_type": "application/json", "response_schema": schema},
    )
    return json.loads(response.text)


def generate_text(prompt: str) -> str:
    """Asks Gemini for a plain-text answer (temperature 0 for stable output)."""
    response = _generate(prompt, {"temperature": 0})
    return (response.text or "").strip()


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


# NEW: output schema for rewriting a follow-up question into a standalone one
class StandaloneQuery(BaseModel):
    standalone_query: str = Field(description="The follow-up question rewritten as a complete, standalone question that makes sense without the earlier conversation")


class ExpandedQuery(BaseModel):
    expanded_query: str = Field(description="The query rewritten with additional relevant keywords and context, for better embedding-based retrieval")


_EMPTY_METADATA = {"dept": None, "sub_dept": None, "subject": None, "doc_date": None}


def extract_metadata(text: str) -> dict:
    """
    Extracts dept, sub_dept, subject, and doc_date from the document's text.

    Metadata is a nice-to-have: if Gemini is unreachable or fails for any
    reason, empty metadata is returned so the document upload (text
    extraction + embeddings, which run locally) still succeeds.
    """
    if not text:
        return dict(_EMPTY_METADATA)

    prompt = f"Read the document text below and extract the requested fields:\n\n{text[:2000]}"

    try:
        return generate_json(prompt, DocumentMetadata)
    except Exception as e:
        logger.warning("Metadata extraction failed, saving the document without it: %s", e)
        return dict(_EMPTY_METADATA)


# NEW: words/patterns that signal a question leans on the earlier conversation
# (a pronoun, a demonstrative, "the previous one", common Hinglish forms).
# The list is deliberately small and clear: if a rewrite is ever skipped by
# mistake, retrieval is slightly off for one turn, but we avoid an extra LLM
# call on every question.
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
    NEW: cheap heuristic (no LLM call) for whether `query` depends on the
    earlier conversation for its meaning. Checks pronoun/reference words in
    both English and common Hinglish. A very short query (4 words or fewer)
    is also treated as a possible follow-up ("list them", "how many?"),
    since short queries are the ones most likely to be incomplete on their own.
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
    NEW: if there is conversation history AND the new question looks like it
    depends on it, rewrites the question into a complete standalone one, so
    retrieval and SQL generation have something concrete to work with
    (e.g. "list them" -> "list all RLY owners").

    If there is no history, or the question already reads as standalone, the
    LLM call is skipped and the original query is returned unchanged. If the
    rewrite fails (e.g. Gemini is down) it also falls back to the original
    query, so a problem here never blocks the whole flow.
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
        return generate_json(prompt, StandaloneQuery)["standalone_query"]
    except Exception as e:
        logger.warning("Query rewriting failed, falling back to original query: %s", e)
        return query


def expand_query_for_retrieval(query: str) -> str:
    """
    Short/vague queries (e.g. "how to login into irfmm") embed too
    generically — a general-purpose embedding model can rank the correct
    chunk far down the list when the document is full of similar UI
    instructions ("click on...", "enter details...", etc).

    This expands the query with related keywords BEFORE the embedding search,
    purely to make retrieval more specific. It does NOT change what question
    is actually answered: generate_answer() still uses the original query.

    A query that is already reasonably long/specific gets no benefit, so the
    LLM call is skipped for anything over a small word-count threshold.

    Falls back to the original query if expansion fails for any reason, so a
    Gemini hiccup here never blocks retrieval entirely.
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
    match relevant text more precisely — add likely related terms
    (e.g. "URL", "portal", "credentials" for a login query) without
    changing what is actually being asked. Keep it to one or two sentences.
    """

    try:
        return generate_json(prompt, ExpandedQuery)["expanded_query"]
    except Exception as e:
        logger.warning("Query expansion failed, falling back to original query: %s", e)
        return query


def _chunk_context_block(index: int, chunk) -> str:
    """Builds one [REF n] context block for the prompt from a Chunk."""
    return f"[REF {index} — document page {chunk.page_number}]: {chunk.text}"


# CHANGED: added an optional `history` parameter. It is used only for
# continuity; it does not affect retrieval.
def generate_answer(query: str, chunks: list, history: list[dict] | None = None) -> dict:
    """
    Takes the user's query and a list of Chunk objects, and asks Gemini to
    answer using only that context, along with which [REF n] blocks it
    actually used. The caller maps refs_used back to the chunks to build
    source citations.

    Raises LLMUnavailableError if Gemini cannot be reached, so an outage is
    not mistaken for "answer not found".
    """
    context_text = "\n\n".join(
        _chunk_context_block(idx, c) for idx, c in enumerate(chunks, start=1)
    )

    # NEW: prior-conversation block, built only when there is history
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
        return generate_json(prompt, QueryAnswer)
    except LLMUnavailableError:
        raise
    except Exception as e:
        # e.g. an empty/blocked response or malformed JSON for this round.
        logger.warning("generate_answer could not produce a valid answer: %s", e)
        return {"answer": "NOT_FOUND", "refs_used": []}