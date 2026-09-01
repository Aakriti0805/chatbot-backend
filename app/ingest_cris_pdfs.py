import os
import psycopg2
import requests

from app.database import SessionLocal
from app.models import Document, Chunk

from app.services.textextraction_service import extract_text_from_pdf
from app.services.document_services import process_document


CRIS_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "cris_data",
    "user": "postgres",
    "password": "akku2005",
}

DOWNLOAD_DIR = "uploaded_files/cris_sync"


def get_pdf_urls():
    conn = psycopg2.connect(**CRIS_DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT process_name, embed_url
            FROM lb1adm.fmm_video_tutorial_links_t
            WHERE embed_url ILIKE '%.pdf'
        """)
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()


def download_pdf(url, filename):
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


def create_safe_filename(process_name):
    safe_name = (
        process_name[:80]
        .replace("/", "_")
        .replace("\\", "_")
        .strip()
    )
    return safe_name + ".pdf"


def main():

    print("=" * 70)
    print("CRIS PDF INGESTION STARTED (sequential mode)")
    print("=" * 70)

    pdf_rows = get_pdf_urls()
    print(f"Total PDFs found in CRIS table: {len(pdf_rows)}")

    db = SessionLocal()

    processed_count = 0
    failed_count = 0
    skipped_count = 0

    try:

        for index, (process_name, url) in enumerate(pdf_rows, start=1):

            safe_name = create_safe_filename(process_name)

            print(f"\n[{index}/{len(pdf_rows)}] {safe_name}")

            if is_already_processed(safe_name, db):
                print("  -> Skipping (already processed)")
                skipped_count += 1
                continue

            # -----------------------------------------------------
            # STEP 1: Download this PDF
            # -----------------------------------------------------
            try:
                filepath = download_pdf(url, safe_name)
                print(f"  -> Downloaded: {filepath}")
            except Exception as e:
                print(f"  -> Failed to download: {e}")
                failed_count += 1
                continue

            # -----------------------------------------------------
            # STEP 2: Create or reuse the Document row
            # (reuse avoids duplicate rows for previously-failed PDFs)
            # -----------------------------------------------------
            try:
                existing_doc = (
                    db.query(Document)
                    .filter(Document.filename == safe_name)
                    .first()
                )

                if existing_doc:
                    existing_doc.filepath = filepath
                    existing_doc.source = "cris_sync"
                    db.commit()
                    db.refresh(existing_doc)
                    document_id = existing_doc.id
                else:
                    new_doc = Document(
                        filename=safe_name,
                        filepath=filepath,
                        source="cris_sync",
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

            # -----------------------------------------------------
            # STEP 3: Extract text from THIS PDF only
            # (runs directly in this process, no multiprocessing)
            # -----------------------------------------------------
            try:
                pages_text = extract_text_from_pdf(filepath)
            except Exception as e:
                print(f"  -> Extraction failed: {e}")
                failed_count += 1
                continue

            if not pages_text:
                print("  -> No extracted content, skipping downstream save")
                failed_count += 1
                continue

            # -----------------------------------------------------
            # STEP 4: Chunk + embed + save immediately for this PDF
            # -----------------------------------------------------
            try:
                # IMPORTANT: process_document must accept
                # pre_extracted_pages so it doesn't re-run extraction.
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
        print("CRIS PDF INGESTION COMPLETED")
        print(f"Processed : {processed_count}")
        print(f"Skipped   : {skipped_count}")
        print(f"Failed    : {failed_count}")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    main()