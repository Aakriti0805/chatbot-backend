import logging
import socket

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.database import engine
from app.routers import document_routes, master_data_routes, ocr_routes, query_routes
from app.services.llm_service import LLMUnavailableError

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Railway Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router)
app.include_router(query_routes.router)
app.include_router(ocr_routes.router)
app.include_router(master_data_routes.router)


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
    """
    Gemini could not be used (network/DNS/API error). Return a clear 503
    instead of a generic 500 or a misleading "answer not found".
    """
    return JSONResponse(
        status_code=503,
        content={"detail": f"AI service is currently unavailable: {exc}"},
    )


# Tables the application needs in the chatbot schema.
_REQUIRED_TABLES = ("documents", "chunks", "table_metadata_chunks")


@app.get("/")
def root():
    return {"message": "Chatbot backend is running"}


@app.get("/health")
def health():
    """
    Quick diagnostic: is the database reachable, do the required tables
    exist, and can this server resolve the Gemini hostname? Returns 503 if
    any check fails, so it can also be used by a monitor.
    """
    report: dict = {"database": "ok", "missing_tables": [], "gemini_dns": "ok"}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            for table in _REQUIRED_TABLES:
                exists = conn.execute(
                    text("SELECT to_regclass(:name)"), {"name": f"chatbot.{table}"}
                ).scalar()
                if exists is None:
                    report["missing_tables"].append(table)
    except Exception as exc:
        report["database"] = f"error: {exc}"

    try:
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
    except OSError as exc:
        report["gemini_dns"] = f"error: {exc}"

    healthy = (
        report["database"] == "ok"
        and not report["missing_tables"]
        and report["gemini_dns"] == "ok"
    )
    return JSONResponse(status_code=200 if healthy else 503, content=report)