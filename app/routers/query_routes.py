import logging
from datetime import datetime
from enum import Enum
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.query_services import (
    answer_query,
    list_conversations,
    get_conversation_messages,
)
from pydantic import BaseModel

router = APIRouter(prefix="/query", tags=["Query"])
logger = logging.getLogger(__name__)


class QuerySourceParam(str, Enum):
    documents = "documents"      # only uploaded PDFs/images (Chunk table)
    master_data = "master_data"  # only database master tables (MasterDataChunk table)
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
    conversation_id: str


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    last_updated_at: datetime


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


@router.get("/ask", response_model=QueryResponse)
def ask_question(
    question: str = Query(..., min_length=1),
    source: QuerySourceParam = Query(
        QuerySourceParam.both,
        description=(
            "Where to look for the answer: 'documents' (uploaded PDFs/"
            "images only), 'master_data' (database master tables only), "
            "or 'both'."
        ),
    ),
    conversation_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    # NOTE: add your existing auth dependency here, e.g.:
    # current_user = Depends(get_current_user)
    result = answer_query(
        question, db, conversation_id=conversation_id, source=source.value
    )
    return result


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations_route(db: Session = Depends(get_db)):
    
    return list_conversations(db)


@router.get("/conversations/{conversation_id}", response_model=list[ConversationMessage])
def get_conversation_route(conversation_id: str, db: Session = Depends(get_db)):
    
    messages = get_conversation_messages(conversation_id, db)

    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return messages