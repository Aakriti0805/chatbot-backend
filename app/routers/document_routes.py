import os
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.database import get_db
from app import auth, models
from app.services.document_services import process_document

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/admin-test")
def admin_test(current_admin: dict = Depends(auth.get_current_admin)):
    return {"message": f"Welcome admin {current_admin['email']}, you have access!"}


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
    file_ext = Path(original_filename).suffix  # e.g. ".pdf"

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

    process_document(filepath, document_id=new_doc.id, db=db)

    return {
        "message": "File uploaded successfully",
        "document_id": new_doc.id,
        "filename": new_doc.filename,
    }