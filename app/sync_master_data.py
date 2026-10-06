import datetime
import sys

from app.database import SessionLocal, engine
from app.models import TableMetadataChunk
from app.services.embedding_service import generate_embeddings
from app.services.sql_query_services import get_catalog_metadata_texts

# Kept as an alias so older imports (`from app.sync_master_data import
# dest_engine`) keep working. It is the shared engine from app.database,
# which already sets search_path=chatbot,public (needed because pgvector's
# "vector" type lives in the chatbot schema, not public).
dest_engine = engine
DestSessionLocal = SessionLocal

# Number of tables whose metadata is embedded in one batch. Each table's
# metadata text is short, so this is plenty.
EMBEDDING_BATCH_SIZE = 32


def parse_table_ref(ref: str) -> tuple[str, str]:
    """
    Splits "schema.table" into (schema, table). A name without a schema is
    rejected on purpose: the source DB has ~20 schemas and the same table
    name can exist in several of them, so guessing the schema could sync
    the wrong table.
    """
    if "." not in ref:
        raise ValueError(
            f"Table ref '{ref}' must be schema-qualified, e.g. 'lb1adm.roams_owner_master'"
        )
    schema, table = ref.split(".", 1)
    schema, table = schema.strip(), table.strip()
    if not schema or not table:
        raise ValueError(f"Invalid table ref: {ref!r}")
    return schema, table


def _ensure_metadata_table() -> None:
    """Creates the table_metadata_chunks table if it does not exist yet."""
    TableMetadataChunk.__table__.create(bind=dest_engine, checkfirst=True)


def _sync_refs(refs: list[str], catalog_texts: dict[str, str], db) -> list[dict]:
    """
    Embeds and upserts the metadata of `refs`. A table whose text is
    unchanged and already has an embedding is not embedded again.
    """
    existing = {
        c.source_table: c
        for c in db.query(TableMetadataChunk)
        .filter(TableMetadataChunk.source_table.in_(refs))
        .all()
    }

    status: dict[str, str] = {}
    pending = []  # rows that need a new embedding

    for ref in refs:
        new_text = catalog_texts[ref]
        row = existing.get(ref)

        if row is None:
            pending.append(TableMetadataChunk(source_table=ref, text=new_text))
            status[ref] = "inserted"
        elif row.text != new_text or row.embedding is None:
            row.text = new_text
            row.embedding = None
            pending.append(row)
            status[ref] = "updated"
        else:
            status[ref] = "unchanged"

    for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
        batch = pending[start:start + EMBEDDING_BATCH_SIZE]
        embeddings = generate_embeddings([r.text for r in batch])
        for row, embedding in zip(batch, embeddings):
            row.embedding = embedding
            row.updated_at = datetime.datetime.utcnow()
            db.add(row)
        db.commit()
        print(f"    {min(start + EMBEDDING_BATCH_SIZE, len(pending))}/{len(pending)} tables embedded")

    db.commit()

    return [
        {
            "source_table": ref,
            "inserted": int(status[ref] == "inserted"),
            "updated": int(status[ref] == "updated"),
            "unchanged": int(status[ref] == "unchanged"),
        }
        for ref in refs
    ]


def run_master_data_sync(table_refs: list[str] | None = None) -> dict:
    """
    Embeds ONLY table metadata (name, description, columns) into
    chatbot.table_metadata_chunks — never row data.

    - If table_refs is given ("schema.table" format), only those tables are
      synced (they must exist in the catalog).
    - If table_refs is empty/None, ALL catalog tables are synced, and the
      metadata of tables that are no longer in the catalog is removed.
    """
    print("=" * 70)
    print("TABLE METADATA SYNC -> chatbot.table_metadata_chunks")
    print("=" * 70)

    catalog_texts = get_catalog_metadata_texts()
    full_sync = not table_refs

    if full_sync:
        refs = list(catalog_texts.keys())
        print(f"Full sync: {len(refs)} tables in the catalog")
    else:
        print(f"Requested: {table_refs}")
        refs = []
        for ref in table_refs:
            try:
                schema, table = parse_table_ref(ref)
            except Exception as e:
                print(f"  -> Skipping '{ref}': {e}")
                continue
            key = f"{schema}.{table}"
            if key not in catalog_texts:
                print(f"  -> Skipping '{key}': this table is not in the catalog")
                continue
            refs.append(key)

    _ensure_metadata_table()

    db = DestSessionLocal()
    try:
        results = _sync_refs(refs, catalog_texts, db) if refs else []

        if full_sync:
            stale = (
                db.query(TableMetadataChunk)
                .filter(~TableMetadataChunk.source_table.in_(list(catalog_texts.keys())))
                .all()
            )
            for row in stale:
                db.delete(row)
            db.commit()
            if stale:
                print(f"  Removed stale metadata: {[r.source_table for r in stale]}")

        inserted = sum(r["inserted"] for r in results)
        updated = sum(r["updated"] for r in results)
        unchanged = sum(r["unchanged"] for r in results)
        print(f"\nDone: inserted={inserted}, updated={updated}, unchanged={unchanged}")
        print("=" * 70)
        return {"tables": results}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Usage:
    #   python -m app.sync_master_data                      -> all catalog tables
    #   python -m app.sync_master_data lb1adm.roams_owner_master lb1adm.fmm_org_m
    run_master_data_sync(sys.argv[1:] or None)