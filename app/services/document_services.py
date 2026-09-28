from app.services.textextraction_service import process_file, extract_text_from_plain_text
from app.services.chunking_services import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.llm_service import extract_metadata
from app.models import Chunk, Document
from sqlalchemy.orm import Session


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
    Supports three ingestion paths:
      - pre_extracted_pages: already-extracted pages (e.g. from parallel batch extraction)
      - filepath: PDF / image (jpg, jpeg, png) / DOCX — process_file() routes it to
        the right extractor automatically
      - raw_text: direct text input (no file, user supplied text directly)

    NOTE: this function used to take a positional `filepath` argument and call
    extract_text_from_pdf() directly — callers now need to use the new
    keyword-argument style (filepath=..., or raw_text=...).
    """
    if pre_extracted_pages is not None:
        pages_text = pre_extracted_pages
    elif raw_text is not None:
        pages_text = extract_text_from_plain_text(raw_text)
    elif filepath is not None:
        pages_text = process_file(filepath=filepath)
    else:
        raise ValueError("Provide one of: filepath, raw_text, or pre_extracted_pages")

    sample_text = " ".join([p["text"] for p in pages_text[:3]])
    metadata = extract_metadata(sample_text)

    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        document.dept = metadata.get("dept")
        document.sub_dept = metadata.get("sub_dept")
        document.subject = metadata.get("subject")
        document.doc_date = metadata.get("doc_date")
        db.commit()

    chunks = chunk_text(pages_text)

    
    texts = [_sanitize_text(c["text"]) for c in chunks]
    embeddings = generate_embeddings(texts)

    for chunk, text, embedding in zip(chunks, texts, embeddings):
        db_chunk = Chunk(
            document_id=document_id,
            page_number=chunk["page_number"],
            text=text,
            embedding=embedding
        )
        db.add(db_chunk)

    db.commit()
    print(f"Saved {len(chunks)} chunks with embeddings for document_id={document_id}")