from app.services.textextraction_service import extract_text_from_pdf
from app.services.chunking_services import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.llm_service import extract_metadata  
from app.models import Chunk, Document
from sqlalchemy.orm import Session


def process_document(filepath: str, document_id: int, db: Session, pre_extracted_pages: list[dict] | None = None):

    # Use already-extracted pages if provided (from parallel batch extraction),
    # otherwise fall back to extracting them here (e.g. for old/other callers).
    if pre_extracted_pages is not None:
        pages_text = pre_extracted_pages
    else:
        pages_text = extract_text_from_pdf(filepath)

    sample_text = " ".join([p["text"] for p in pages_text[:3]])
    metadata = extract_metadata(sample_text)

    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        document.dept = metadata.get("dept")
        document.sub_dept = metadata.get("sub_dept")
        document.subject = metadata.get("subject")
        document.doc_date = metadata.get("doc_date")
        db.commit()

    # Step 3: Chunking — split text into semantically coherent chunks
    chunks = chunk_text(pages_text)

    # Step 4: Embeddings — generate a vector for each chunk
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    # Step 5: Save all chunks + embeddings to the database
    for chunk, embedding in zip(chunks, embeddings):
        db_chunk = Chunk(
            document_id=document_id,
            page_number=chunk["page_number"],
            text=chunk["text"],
            embedding=embedding
        )
        db.add(db_chunk)

    db.commit()
    print(f"Saved {len(chunks)} chunks with embeddings for document_id={document_id}")