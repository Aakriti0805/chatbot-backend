import logging
from enum import Enum

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.query_services import (
    answer_query,
    get_conversation_messages,
    list_conversations,
)

router = APIRouter(prefix="/query", tags=["Query"])
logger = logging.getLogger(__name__)


class QuerySourceParam(str, Enum):
    documents = "documents"      # only uploaded PDFs/images/DOCX (Chunk table)
    master_data = "master_data"  # only database master tables (live SQL)
    both = "both"


class SourceItem(BaseModel):
    type: str  # "document" or "master_data"
    document_id: int | None = None
    page_number: int | None = None
    filename: str | None = None
    download_url: str | None = None
    source_table: str | None = None
    record_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    rounds_used: int
    conversation_id: str | None = None  # NEW: the frontend sends this back with the next follow-up


@router.get("/ask", response_model=QueryResponse)
def ask_question(
    question: str = Query(..., min_length=1),
    source: QuerySourceParam = Query(
        QuerySourceParam.both,
        description=(
            "Where to look for the answer: 'documents' (uploaded files "
            "only), 'master_data' (database master tables only), or 'both'."
        ),
    ),
    conversation_id: str | None = Query(
        None,
        description=(
            "The conversation_id returned by a previous answer. Send it for "
            "follow-up questions; leave it empty to start a new conversation."
        ),
    ),
    db: Session = Depends(get_db),
):
    # NOTE: add your auth dependency here when ready, e.g.:
    # current_user: dict = Depends(get_current_user)
    #
    # If Gemini cannot be reached, answer_query raises LLMUnavailableError,
    # which main.py turns into a clear 503 response.
    return answer_query(
        question, db, source=source.value, conversation_id=conversation_id
    )


@router.get("/conversations")
def list_all_conversations(db: Session = Depends(get_db)):
    """NEW: sidebar list, newest conversation first."""
    return list_conversations(db)


@router.get("/conversations/{conversation_id}")
def get_one_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """NEW: all messages of one conversation."""
    return get_conversation_messages(conversation_id, db)