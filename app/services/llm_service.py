import os
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
    pages_used: list[int] = Field(default_factory=list, description="Page numbers actually used to construct the answer; empty if NOT_FOUND")


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


def generate_answer(query: str, chunks: list) -> dict:
    """
    Takes the user's query and a list of retrieved chunks, and asks Gemini
    to generate an answer using only that context, along with which page
    numbers the answer actually came from.
    """
    context_text = "\n\n".join(
        [f"[Page {c.page_number}]: {c.text}" for c in chunks]
    )

    prompt = f"""
    You are a helpful assistant for railway maintenance staff.
    Answer the user's question using ONLY the context provided below.

    If the context does not contain enough information to answer the
    question, set "answer" to exactly "NOT_FOUND" and "pages_used" to an
    empty list.

    Otherwise, answer the question and list every page number in
    "pages_used" whose content you actually relied on to build the answer.
    If the same fact appears on multiple pages, include all of them.

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
        return {"answer": "NOT_FOUND", "pages_used": []}