import logging
import uuid
from typing import Literal
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.services.retrieval_services import retrieve_relevant_chunks
from app.services.llm_service import generate_answer, rewrite_query_with_history, expand_query_for_retrieval
from app.services.sql_query_services import answer_master_data_query
from app.models import ChatMessage, Document, Chunk, MasterDataChunk

logger = logging.getLogger(__name__)

MAX_ROUNDS = 4
CHUNKS_PER_ROUND = 15
HISTORY_TURNS = 3

# "documents"   -> only uploaded PDFs/images
# "master_data" -> only database master tables
# "both"        -> both
QuerySource = Literal["documents", "master_data", "both"]


def _get_recent_history(conversation_id: str, db: Session) -> list[dict]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(HISTORY_TURNS * 2)
        .all()
    )
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


def _save_message(conversation_id: str, role: str, content: str, db: Session) -> None:
    db.add(ChatMessage(conversation_id=conversation_id, role=role, content=content))
    db.commit()


def _build_sources(chunks: list, used_refs: set, db: Session) -> list[dict]:
    """
    `chunks` can contain both Chunk (uploaded documents) and
    MasterDataChunk (database master tables) objects, so each chunk's type
    is checked to build the appropriately shaped source dict.

    `used_refs` is the set of 1-based REF numbers (a chunk's position in
    `chunks`) that generate_answer() reported as actually used — this
    tells us exactly which chunks the LLM relied on for the final answer.
    """
    document_ids = {c.document_id for c in chunks if isinstance(c, Chunk)}
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

        if isinstance(chunk, Chunk):
            key = ("document", chunk.document_id, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)

            document = documents_by_id.get(chunk.document_id)

            # If this document came from CRIS and has its original public URL
            # stored, use that directly (already-hosted, no auth needed).
            # Otherwise (e.g. an admin-uploaded document with no public URL),
            # fall back to our own download endpoint.
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

        elif isinstance(chunk, MasterDataChunk):
            key = ("master_data", chunk.source_table, chunk.record_id)
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "type": "master_data",
                "source_table": chunk.source_table,
                "record_id": chunk.record_id,
            })

    return sources


def answer_query(
    query: str,
    db: Session,
    top_k: int = CHUNKS_PER_ROUND,
    conversation_id: str | None = None,
    source: QuerySource = "both",
) -> dict:
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # History and the standalone rewrite now happen BEFORE the SQL branch.
    # Previously, a follow-up like "list them" or "but there were 208 of
    # them?" was sent to answer_master_data_query() as raw text, with no
    # idea what "them" referred to, because the rewrite only happened
    # later in the document-retrieval path. Doing it here first means both
    # the SQL path and the document path get a fully resolved, standalone
    # question.
    history = _get_recent_history(conversation_id, db)
    standalone_query = rewrite_query_with_history(query, history)

    # Any question about the master-data tables — a specific lookup, a
    # whole-table or category count/list, or a cross-field summary — is
    # tried via text-to-SQL first: the LLM writes real SQL against the
    # actual schema instead of being matched against pre-defined question
    # patterns, so this isn't limited to categories anyone anticipated.
    # Falls through to chunk-based retrieval below if the question isn't
    # about this data, or SQL generation/execution/summarization fails.
    if source in ("master_data", "both"):
        sql_result = answer_master_data_query(standalone_query)
        if sql_result is not None:
            _save_message(conversation_id, "user", query, db)
            _save_message(conversation_id, "assistant", sql_result["answer"], db)
            sql_result["conversation_id"] = conversation_id
            return sql_result

    retrieval_query = expand_query_for_retrieval(standalone_query)

    _save_message(conversation_id, "user", query, db)

    seen_chunk_ids: list[str] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        chunks = retrieve_relevant_chunks(
            retrieval_query,
            db,
            top_k=top_k,
            exclude_ids=seen_chunk_ids,
            source=source,
            # Keyword matching uses the pre-expansion question, not the
            # embedding-oriented expanded one — expansion adds extra
            # generic terms that would otherwise dilute an exact literal
            # match (e.g. a specific company or wagon-type name).
            keyword_query=standalone_query,
        )

        if not chunks:
            logger.debug(f"Round {round_num}: no more chunks left to search.")
            break

        # Prefix with type so retrieval_services' exclude_ids logic filters
        # each subsequent round against the *right* table (master vs doc).
        for chunk in chunks:
            if isinstance(chunk, MasterDataChunk):
                seen_chunk_ids.append(f"master:{chunk.id}")
            else:
                seen_chunk_ids.append(f"doc:{chunk.id}")

        try:
            result = generate_answer(query, chunks, history=history)
        except Exception as e:
            logger.warning(f"Round {round_num}: generation failed — {e}")
            continue

        if result["answer"].strip().upper() != "NOT_FOUND":
            used_refs = set(result.get("refs_used", []))
            final_answer = result["answer"]

            _save_message(conversation_id, "assistant", final_answer, db)

            return {
                "answer": final_answer,
                "sources": _build_sources(chunks, used_refs, db),
                "rounds_used": round_num,
                "conversation_id": conversation_id,
            }

    fallback_answer = "I could not find a clear answer to this question in the uploaded documents."
    _save_message(conversation_id, "assistant", fallback_answer, db)

    return {
        "answer": fallback_answer,
        "sources": [],
        "rounds_used": MAX_ROUNDS,
        "conversation_id": conversation_id,
    }


def list_conversations(db: Session) -> list[dict]:
    """
    For the sidebar — each conversation_id's first user message becomes
    its "title", and the most recent message's timestamp becomes
    "last_updated_at". Most recently updated conversation comes first,
    same as Claude's sidebar.
    """
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
    """
    All messages for one conversation, in chronological (oldest-first)
    order — used when the user clicks an old chat in the sidebar, to load
    its full history into the frontend.
    """
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at,
        }
        for m in messages
    ]