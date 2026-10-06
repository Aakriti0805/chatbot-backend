import logging
import uuid
from typing import Literal

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import ChatMessage, Document
from app.services.llm_service import (
    expand_query_for_retrieval,
    generate_answer,
    rewrite_query_with_history,
)
from app.services.retrieval_services import retrieve_relevant_chunks
from app.services.sql_query_services import answer_master_data_query

logger = logging.getLogger(__name__)

MAX_ROUNDS = 4
CHUNKS_PER_ROUND = 15
HISTORY_TURNS = 3  # NEW: how many previous question/answer pairs are remembered for follow-ups

QuerySource = Literal["documents", "master_data", "both"]

_NOT_FOUND_MESSAGES = {
    "documents": "I could not find a clear answer to this question in the uploaded documents.",
    "master_data": "I could not find a clear answer to this question in the master data records.",
    "both": "I could not find a clear answer to this question in the uploaded documents or the master data records.",
}


def _get_recent_history(conversation_id: str, db: Session) -> list[dict]:
    """NEW: the last HISTORY_TURNS question/answer pairs of this conversation, oldest first."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(HISTORY_TURNS * 2)
        .all()
    )
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


def _save_exchange(conversation_id: str, query: str, result: dict, db: Session) -> None:
    """
    NEW: saves the user's question and the assistant's answer. The answer
    must still reach the user even if saving the history fails, so a failure
    here is only logged.
    """
    try:
        db.add(ChatMessage(conversation_id=conversation_id, role="user", content=query))
        db.add(
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=result["answer"],
                sources=result.get("sources") or [],
            )
        )
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("Could not save chat history: %s", e)


def _build_sources(chunks: list, used_refs: set, db: Session) -> list[dict]:
    """
    Builds the source list for the REFs the model actually used. Only Chunk
    (uploaded document) objects reach this function — the master-data path
    builds its own source dicts in sql_query_services.answer_master_data_query.
    """
    document_ids = {c.document_id for c in chunks}
    documents = (
        db.query(Document).filter(Document.id.in_(document_ids)).all()
        if document_ids else []
    )
    documents_by_id = {d.id: d for d in documents}

    sources = []
    seen = set()

    for ref_num, chunk in enumerate(chunks, start=1):
        if ref_num not in used_refs:
            continue

        key = (chunk.document_id, chunk.page_number)
        if key in seen:
            continue
        seen.add(key)

        document = documents_by_id.get(chunk.document_id)

        if document and document.source_url:
            download_url = document.source_url
        else:
            download_url = f"/documents/{chunk.document_id}/download"

        sources.append({
            "type": "document",
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            "filename": document.filename if document else None,
            "download_url": download_url,
        })

    return sources


def _run_query(
    standalone_query: str,
    history: list[dict],
    db: Session,
    top_k: int,
    source: QuerySource,
) -> dict:
    """
    CHANGED: the body of the old answer_query lives here, with the same
    logic. The only differences are that retrieval/SQL use the standalone
    query, and generate_answer also receives the history.
    """
    # Master-data questions are answered ONLY via live SQL — there is no
    # row-level embedding fallback for this source. If SQL can't answer it,
    # a master_data-only request goes straight to the NOT_FOUND message.
    if source in ("master_data", "both"):
        sql_result = answer_master_data_query(standalone_query)
        if sql_result is not None:
            return sql_result

    if source == "master_data":
        return {
            "answer": _NOT_FOUND_MESSAGES["master_data"],
            "sources": [],
            "rounds_used": 0,
        }

    # The chunk-retrieval path below covers UPLOADED DOCUMENTS only,
    # regardless of the caller's `source`.
    retrieval_query = expand_query_for_retrieval(standalone_query)

    seen_chunk_ids: list[int] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        chunks = retrieve_relevant_chunks(
            retrieval_query,
            db,
            top_k=top_k,
            exclude_ids=seen_chunk_ids,
            keyword_query=standalone_query,
        )

        if not chunks:
            logger.debug("Round %d: no more chunks left to search.", round_num)
            break

        seen_chunk_ids.extend(c.id for c in chunks)

        result = generate_answer(standalone_query, chunks, history=history)

        if result["answer"].strip().upper() != "NOT_FOUND":
            used_refs = set(result.get("refs_used", []))
            return {
                "answer": result["answer"],
                "sources": _build_sources(chunks, used_refs, db),
                "rounds_used": round_num,
            }

    return {
        "answer": _NOT_FOUND_MESSAGES.get(source, _NOT_FOUND_MESSAGES["both"]),
        "sources": [],
        "rounds_used": MAX_ROUNDS,
    }


def answer_query(
    query: str,
    db: Session,
    top_k: int = CHUNKS_PER_ROUND,
    source: QuerySource = "both",
    conversation_id: str | None = None,
) -> dict:
    """
    Answers one question. If a conversation_id is given, the earlier turns of
    that conversation are remembered (for follow-up questions); otherwise a
    new conversation starts, and its conversation_id is returned in the
    response.

    Raises LLMUnavailableError (from llm_service) if Gemini cannot be
    reached; the API layer turns that into a 503.
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    history = _get_recent_history(conversation_id, db)
    standalone_query = rewrite_query_with_history(query, history)

    result = _run_query(standalone_query, history, db, top_k, source)

    _save_exchange(conversation_id, query, result, db)
    result["conversation_id"] = conversation_id
    return result


def list_conversations(db: Session) -> list[dict]:
    """NEW: all conversations for the sidebar, newest first. The title is the first message."""
    conversation_stats = (
        db.query(
            ChatMessage.conversation_id,
            func.min(ChatMessage.id).label("first_message_id"),
            func.max(ChatMessage.created_at).label("last_updated_at"),
        )
        .group_by(ChatMessage.conversation_id)
        .subquery()
    )

    first_messages = (
        db.query(ChatMessage.conversation_id, ChatMessage.content)
        .join(
            conversation_stats,
            ChatMessage.id == conversation_stats.c.first_message_id,
        )
        .all()
    )
    titles_by_conversation = {conv_id: content for conv_id, content in first_messages}

    conversations = (
        db.query(
            conversation_stats.c.conversation_id,
            conversation_stats.c.last_updated_at,
        )
        .order_by(conversation_stats.c.last_updated_at.desc())
        .all()
    )

    result = []
    for conversation_id, last_updated_at in conversations:
        title = titles_by_conversation.get(conversation_id, "").strip()

        if len(title) > 60:
            title = title[:60].rstrip() + "..."

        result.append({
            "conversation_id": conversation_id,
            "title": title or "New conversation",
            "last_updated_at": last_updated_at,
        })

    return result


def get_conversation_messages(conversation_id: str, db: Session) -> list[dict]:
    """NEW: all messages of one conversation, oldest first."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )

    return [
        {
            "role": m.role,
            "content": m.content,
            "sources": m.sources,
            "created_at": m.created_at,
        }
        for m in messages
    ]