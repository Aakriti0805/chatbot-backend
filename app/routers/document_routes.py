import os
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app import auth, models
from app.services.document_services import process_document
from app.ingest_cris_pdfs import run_cris_ingestion

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Simple in-memory guard so two ingestion runs never overlap.
_cris_ingestion_state = {"running": False}

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx"}


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    dept: str = Form(None),
    sub_dept: str = Form(None),
    subject: str = Form(None),
    doc_date: str = Form(None),
    db: Session = Depends(get_db),
    current_admin: dict = Depends(auth.get_current_admin),
):
    original_filename = file.filename
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

    
    process_document(filepath=filepath, document_id=new_doc.id, db=db)
    return {
        "message": "File uploaded successfully",
        "document_id": new_doc.id,
        "filename": new_doc.filename,
    }


def _run_cris_ingestion_job():
    _cris_ingestion_state["running"] = True
    try:
        run_cris_ingestion()
    finally:
        _cris_ingestion_state["running"] = False


@router.get("/ingest-cris")
def ingest_cris_pdfs(background_tasks: BackgroundTasks):
    if _cris_ingestion_state["running"]:
        return {"message": "CRIS ingestion is already running, please wait for it to finish."}

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