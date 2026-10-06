import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from sqlalchemy import text

from app.database import engine
from app.sync_master_data import run_master_data_sync

router = APIRouter(prefix="/master-data", tags=["Master Data"])
logger = logging.getLogger(__name__)

# Postgres advisory lock: stops two server processes (e.g. several uvicorn
# workers) from running the sync at the same time. The in-memory flag below
# only protects a single process.
_SYNC_LOCK_KEY = 727384910
_sync_state = {"running": False}


def _run_master_data_sync_job(table_refs: list[str]):
    try:
        with engine.connect() as conn:
            got_lock = conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": _SYNC_LOCK_KEY}
            ).scalar()
            if not got_lock:
                logger.info("Master data sync already running elsewhere, skipping this trigger.")
                return
            try:
                run_master_data_sync(table_refs)
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _SYNC_LOCK_KEY})
    except Exception:
        logger.exception("Master data sync failed")
    finally:
        _sync_state["running"] = False


@router.get("/sync")
def sync_master_data_route(
    background_tasks: BackgroundTasks,
    tables: str = Query(
        ...,
        description="Required. Comma-separated, schema-qualified table names, e.g. 'lb1adm.roams_owner_master,fms.some_table'.",
    ),
):
    """
    Embeds the metadata (name, description, columns — not row data) of
    exactly the named table(s) into chatbot.table_metadata_chunks. Each table
    must be given as schema.table. Nothing is auto-discovered; only what is
    listed here gets synced.
    """
    if _sync_state["running"]:
        return {"message": "Master data sync is already running, please wait for it to finish."}

    table_refs = [t.strip() for t in tables.split(",") if t.strip()]
    if not table_refs:
        raise HTTPException(status_code=400, detail="Provide at least one schema-qualified table name.")

    invalid = [t for t in table_refs if "." not in t]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"These are not schema-qualified (need 'schema.table'): {invalid}",
        )

    # Set the flag here (not inside the job) so a second request arriving
    # before the background task starts cannot start a duplicate run.
    _sync_state["running"] = True
    background_tasks.add_task(_run_master_data_sync_job, table_refs)

    return {"message": f"Master data sync started in background for: {', '.join(table_refs)}"}