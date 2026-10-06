import gc
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.textextraction_service import (
    extract_text_from_image,
    extract_text_from_pdf,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _safe_remove(filepath: str, retries: int = 5, delay: float = 0.5):
    """
    Windows pe file thodi der lock reh sakti hai, isliye kuch baar retry karte hain.
    Delete fail ho jaye to bhi API ko error nahi dena, bas warning log kar do.
    """
    for _ in range(retries):
        try:
            os.remove(filepath)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            # Purane objects ke handle chhodne ke liye garbage collect chalao, phir wait karo
            gc.collect()
            time.sleep(delay)
    logger.warning("Temp file delete nahi ho payi: %s", filepath)


# Ye `def` hai, `async def` nahi: OCR slow aur blocking kaam hai.
# FastAPI sync endpoints ko thread pool mein chalata hai, isliye event loop free rehta hai.
@router.post("/extract")
def extract_ocr(file: UploadFile = File(...)):
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension == ".pdf":
        extractor = extract_text_from_pdf
    elif extension in IMAGE_EXTENSIONS:
        extractor = extract_text_from_image
    else:
        return {
            "success": False,
            "message": "Unsupported file type",
        }

    # Client ka diya filename kabhi path mein use nahi karte ("../../x" jaisa naam
    # UPLOAD_DIR se bahar likh sakta hai). Random temp naam use karo aur kaam ke baad delete karo.
    filepath = os.path.join(UPLOAD_DIR, f"ocr_{uuid.uuid4().hex}{extension}")

    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        pages = extractor(filepath)
    except Exception as exc:
        logger.exception("OCR extraction failed: %s", original_name)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")
    finally:
        # Cleanup ka error asli error ko chhupaye nahi, isliye safe remove
        _safe_remove(filepath)

    return {
        "success": True,
        "filename": original_name,
        "pages": pages,
    }