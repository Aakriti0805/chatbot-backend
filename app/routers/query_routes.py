import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.query_services import answer_query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/query", tags=["Query"])
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceItem(BaseModel):
    document_id: int
    page_number: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    rounds_used: int


@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, db: Session = Depends(get_db)):
    # NOTE: add your existing auth dependency here, e.g.:
    # current_user = Depends(get_current_user)
    result = answer_query(request.question, db, top_k=5)
    return result