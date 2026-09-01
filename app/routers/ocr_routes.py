import os
import shutil

from fastapi import APIRouter, UploadFile, File
from app.services.textextraction_service import (
    extract_text_from_pdf,
    extract_text_from_image,
)

router = APIRouter(prefix="/ocr", tags=["OCR"])

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/extract")
async def extract_ocr(file: UploadFile = File(...)):

    filepath = os.path.join(UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extension = os.path.splitext(file.filename)[1].lower()

    if extension == ".pdf":
        pages = extract_text_from_pdf(filepath)

    elif extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        pages = extract_text_from_image(filepath)

    else:
        return {
            "success": False,
            "message": "Unsupported file type"
        }

    return {
        "success": True,
        "filename": file.filename,
        "pages": pages
    }