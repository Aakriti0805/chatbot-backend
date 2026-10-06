import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import models
from app.database import get_db
from app.ingest_cris_pdfs import run_cris_ingestion
from app.services.document_services import process_document

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Simple in-memory guard so two ingestion runs never overlap.
_cris_ingestion_state = {"running": False}

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx"}


def _discard_failed_upload(db: Session, document_id: int, filepath: str) -> None:
    """
    Removes the Document row and the saved file of an upload whose
    processing failed. Without this, every failed upload leaves behind a
    document with zero chunks (and a file nobody can search).
    """
    try:
        db.rollback()
        db.query(models.Document).filter(models.Document.id == document_id).delete()
        db.commit()
    except Exception:
        logger.exception("Could not remove the Document row for failed upload id=%s", document_id)
        db.rollback()

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        logger.exception("Could not remove file of failed upload: %s", filepath)


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    dept: str = Form(None),
    sub_dept: str = Form(None),
    subject: str = Form(None),
    doc_date: str = Form(None),
    db: Session = Depends(get_db),
):
    original_filename = file.filename or ""
    file_ext = Path(original_filename).suffix.lower()

    if file_ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Supported: {', '.join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))}",
        )

    stored_filename = f"{uuid.uuid4().hex}{file_ext}"
    filepath = os.path.join(UPLOAD_DIR, stored_filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_doc = models.Document(
        filename=original_filename,
        filepath=filepath,
        source="admin_upload",
        dept=dept,
        sub_dept=sub_dept,
        subject=subject,
        doc_date=doc_date,
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    document_id = new_doc.id

    try:
        process_document(filepath=filepath, document_id=document_id, db=db)
    except Exception as exc:
        logger.exception("Processing failed for upload '%s' (document_id=%s)", original_filename, document_id)
        _discard_failed_upload(db, document_id, filepath)
        # ValueError = the file itself is the problem (no readable text, etc.)
        status_code = 422 if isinstance(exc, ValueError) else 500
        raise HTTPException(
            status_code=status_code,
            detail=f"Could not process '{original_filename}': {exc}",
        )

    return {
        "message": "File uploaded successfully",
        "document_id": document_id,
        "filename": original_filename,
    }


def _run_cris_ingestion_job():
    try:
        run_cris_ingestion()
    except Exception:
        logger.exception("CRIS ingestion failed")
    finally:
        _cris_ingestion_state["running"] = False


@router.get("/ingest-cris")
def ingest_cris_pdfs(background_tasks: BackgroundTasks):
    if _cris_ingestion_state["running"]:
        return {"message": "CRIS ingestion is already running, please wait for it to finish."}

    # Set the flag here (not inside the job) so a second request arriving
    # before the background task starts cannot start a duplicate run.
    _cris_ingestion_state["running"] = True
    background_tasks.add_task(_run_cris_ingestion_job)

    return {
        "message": "CRIS ingestion started in background. Check server logs for progress."
    }


@router.get("/{document_id}/download")
def download_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(models.Document).filter(models.Document.id == document_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.filepath or not os.path.exists(document.filepath):
        raise HTTPException(status_code=404, detail="File is missing on the server")

    return FileResponse(
        path=document.filepath,
        filename=document.filename,
        media_type="application/octet-stream",
    )