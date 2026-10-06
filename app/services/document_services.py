import logging
import time

from app.services.textextraction_service import process_file, extract_text_from_plain_text
from app.services.chunking_services import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.llm_service import extract_metadata
from app.models import Chunk, Document
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _sanitize_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("\x00", "")
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def process_document(
    document_id: int,
    db: Session,
    filepath: str | None = None,
    raw_text: str | None = None,
    pre_extracted_pages: list[dict] | None = None,
):
    """
    Teen tarah se document ingest ho sakta hai:
      - pre_extracted_pages: pages pehle se extract ho chuke hain (jaise parallel batch extraction se)
      - filepath: PDF / image (jpg, jpeg, png) / DOCX - process_file() khud sahi extractor choose karta hai
      - raw_text: user ne seedha text diya hai, koi file nahi

    NOTE: pehle ye function positional `filepath` leta tha aur seedha
    extract_text_from_pdf() call karta tha. Ab callers ko keyword arguments
    use karne honge (filepath=..., ya raw_text=...).
    """
    started_at = time.perf_counter()
    logger.info("Starting document processing: document_id=%s", document_id)
    try:
        # ---- Step 1: text extraction (OCR yahin hota hai agar page scanned hai) ----
        t = time.perf_counter()
        if pre_extracted_pages is not None:
            pages_text = pre_extracted_pages
        elif raw_text is not None:
            pages_text = extract_text_from_plain_text(raw_text)
        elif filepath is not None:
            pages_text = process_file(filepath=filepath)
        else:
            raise ValueError(
                "Provide one of: filepath, raw_text, or pre_extracted_pages"
            )

        extracted_characters = sum(len(page.get("text", "")) for page in pages_text)
        logger.info(
            "Text extraction complete: document_id=%s pages=%d characters=%d took=%.2fs",
            document_id,
            len(pages_text),
            extracted_characters,
            time.perf_counter() - t,
        )

        # ---- Step 2: chunking ----
        t = time.perf_counter()
        chunks = chunk_text(pages_text)
        if not chunks:
            raise ValueError(
                "No usable text was extracted for embedding; check the PDF/OCR output"
            )

        texts = [_sanitize_text(chunk["text"]) for chunk in chunks]
        if any(not text.strip() for text in texts):
            raise ValueError("Chunking produced empty text; refusing to embed it")
        logger.info(
            "Chunking complete: document_id=%s chunks=%d took=%.2fs",
            document_id,
            len(chunks),
            time.perf_counter() - t,
        )

        # ---- Step 3: metadata extraction (Gemini API call) ----
        t = time.perf_counter()
        logger.info("Extracting metadata: document_id=%s", document_id)
        sample_text = " ".join(page.get("text", "") for page in pages_text[:3])
        metadata = extract_metadata(sample_text)
        logger.info(
            "Metadata extraction complete: document_id=%s took=%.2fs",
            document_id,
            time.perf_counter() - t,
        )

        # ---- Step 4: embeddings ----
        t = time.perf_counter()
        embeddings = generate_embeddings(texts)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch for document_id={document_id}: "
                f"{len(embeddings)} embeddings for {len(chunks)} chunks"
            )
        logger.info(
            "Embedding step done: document_id=%s took=%.2fs",
            document_id,
            time.perf_counter() - t,
        )

        # ---- Step 5: DB mein save ----
        t = time.perf_counter()
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.dept = metadata.get("dept")
            document.sub_dept = metadata.get("sub_dept")
            document.subject = metadata.get("subject")
            document.doc_date = metadata.get("doc_date")

        for chunk, text, embedding in zip(chunks, texts, embeddings):
            db.add(
                Chunk(
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    text=text,
                    embedding=embedding,
                )
            )

        db.commit()
        logger.info(
            "DB save complete: document_id=%s took=%.2fs",
            document_id,
            time.perf_counter() - t,
        )

        logger.info(
            "Document processing complete: document_id=%s chunks=%d elapsed=%.2fs",
            document_id,
            len(chunks),
            time.perf_counter() - started_at,
        )
    except Exception:
        db.rollback()
        logger.exception("Document processing failed: document_id=%s", document_id)
        raise