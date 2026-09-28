import os
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse, unquote

from app.models import Document, Chunk
from app.services.textextraction_service import process_file
from app.services.document_services import process_document


load_dotenv(override=True)

SOURCE_DATABASE_URL = os.getenv("SOURCE_DATABASE_URL")
DEST_DATABASE_URL = os.getenv("DATABASE_URL")

if not SOURCE_DATABASE_URL:
    raise ValueError("SOURCE_DATABASE_URL is not set")

if not DEST_DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")


source_engine = create_engine(SOURCE_DATABASE_URL)

# search_path is needed here because pgvector's "vector" type lives in the
# chatbot schema (not public). Without it, SQLAlchemy's bulk-insert
# query-path (inserting many chunks at once) throws "type vector does not
# exist" — this already happened once, on a 45-chunk document.
dest_engine = create_engine(
    DEST_DATABASE_URL,
    connect_args={"options": "-csearch_path=chatbot,public"},
)

DestSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=dest_engine
)


DOWNLOAD_DIR = "uploaded_files/cris_sync"

SUPPORTED_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".docx")


def get_file_urls():
    """
    Fetches PDF/image/DOCX rows from the CRIS source database.
    Excludes doc_type = 'video' (YouTube links) — those aren't documents
    to extract text from.
    """
    with source_engine.connect() as conn:
        result = conn.execute(text("""
            SELECT process_name, embed_url
            FROM lb1adm.fmm_video_tutorial_links_t
            WHERE doc_type IS DISTINCT FROM 'video'
              AND (
                embed_url ILIKE '%.pdf'
                OR embed_url ILIKE '%.jpg'
                OR embed_url ILIKE '%.jpeg'
                OR embed_url ILIKE '%.png'
                OR embed_url ILIKE '%.docx'
              )
        """))
        return result.fetchall()


def infer_extension(url: str) -> str:
    """
    Extracts the file extension from a URL (ignoring query params/fragments).

    Some CRIS embed_url values are proxy/redirect endpoints where the real
    extension sits in a query parameter rather than the URL path itself
    (e.g. ".../download?file=photo.jpg") — os.path.splitext on the bare
    path returns "" in that case. That used to silently fall back to
    ".pdf" in create_safe_filename() for EVERY such URL, including jpg/png
    images, which meant images from the source DB were being downloaded
    and saved with a .pdf extension and then routed through the PDF
    extractor instead of the image/OCR extractor — so only "real" .pdf
    URLs were ever processed correctly.

    Fix: if the path has no usable extension, also check the full URL
    string for one of the supported extensions before giving up.
    """
    path = urlparse(url).path
    filename = unquote(os.path.basename(path))
    ext = os.path.splitext(filename)[1].lower()

    if ext in SUPPORTED_EXTENSIONS:
        return ext

    url_lower = url.lower()
    for candidate in SUPPORTED_EXTENSIONS:
        if candidate in url_lower:
            return candidate

    return ext


def download_file(url, filename):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


def is_already_processed(filename, db):
    """
    A document only counts as "already processed" if a Document row
    exists AND it actually has chunks saved. A Document with 0 chunks
    means a previous run failed partway through and should be retried,
    not skipped forever.
    """
    existing = (
        db.query(Document)
        .filter(Document.filename == filename)
        .first()
    )

    if not existing:
        return False

    has_chunks = (
        db.query(Chunk.id)
        .filter(Chunk.document_id == existing.id)
        .first()
        is not None
    )

    return has_chunks


def create_safe_filename(process_name, url):
    """Extension is inferred from the URL since this handles multiple file types."""
    ext = infer_extension(url)
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"  -> WARNING: could not infer a known extension from URL, defaulting to .pdf: {url}")
        ext = ".pdf"

    safe_name = (
        process_name[:80]
        .replace("/", "_")
        .replace("\\", "_")
        .strip()
    )
    return safe_name + ext


def run_cris_ingestion() -> dict:
    """
    Core ingestion logic, reusable from both the CLI entrypoint and the
    FastAPI background task. Returns a stats dict instead of just printing,
    so an API caller can log/inspect the result too.
    """
    print("=" * 70)
    print("CRIS FILE INGESTION -> PRODUCTION DATABASE")
    print("=" * 70)

    file_rows = get_file_urls()
    print(f"Total files found in source (roams): {len(file_rows)}")

    db = DestSessionLocal()

    processed_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        for index, (process_name, url) in enumerate(file_rows, start=1):

            safe_name = create_safe_filename(process_name, url)

            print(f"\n[{index}/{len(file_rows)}] {safe_name}")

            if is_already_processed(safe_name, db):
                print("  -> Skipping (already processed)")
                skipped_count += 1
                continue

            try:
                filepath = download_file(url, safe_name)
                print(f"  -> Downloaded: {filepath}")
            except Exception as e:
                print(f"  -> Failed to download: {e}")
                failed_count += 1
                continue

            try:
                existing_doc = (
                    db.query(Document)
                    .filter(Document.filename == safe_name)
                    .first()
                )

                if existing_doc:
                    existing_doc.filepath = filepath
                    existing_doc.source = "cris_sync"
                    existing_doc.source_url = url  # CRIS's original public download link
                    db.commit()
                    db.refresh(existing_doc)
                    document_id = existing_doc.id
                else:
                    new_doc = Document(
                        filename=safe_name,
                        filepath=filepath,
                        source="cris_sync",
                        source_url=url,  # CRIS's original public download link
                    )
                    db.add(new_doc)
                    db.commit()
                    db.refresh(new_doc)
                    document_id = new_doc.id

            except Exception as e:
                print(f"  -> Failed to create Document row: {e}")
                db.rollback()
                failed_count += 1
                continue

            try:
                pages_text = process_file(filepath)
            except Exception as e:
                print(f"  -> Extraction failed: {e}")
                failed_count += 1
                continue

            if not pages_text:
                print("  -> No extracted content, skipping downstream save")
                failed_count += 1
                continue

            try:
                process_document(
                    filepath=filepath,
                    document_id=document_id,
                    db=db,
                    pre_extracted_pages=pages_text,
                )
                print(f"  -> Processed and saved chunks for: {safe_name}")
                processed_count += 1

            except Exception as e:
                print(f"  -> Failed downstream processing: {e}")
                db.rollback()
                failed_count += 1

        print("\n" + "=" * 70)
        print("INGESTION COMPLETED")
        print(f"Processed : {processed_count}")
        print(f"Skipped   : {skipped_count}")
        print(f"Failed    : {failed_count}")
        print("=" * 70)

        return {
            "total_found": len(file_rows),
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }

    finally:
        db.close()


if __name__ == "__main__":
    run_cris_ingestion()