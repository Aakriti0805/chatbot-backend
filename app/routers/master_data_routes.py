from fastapi import APIRouter, Depends, BackgroundTasks
from app import auth
from app.sync_master_data import run_master_data_sync

router = APIRouter(prefix="/master-data", tags=["Master Data"])

# Simple in-memory guard so two sync runs never overlap — same pattern as
# the CRIS document ingestion route.
_master_sync_state = {"running": False}


def _run_master_data_sync_job():
    _master_sync_state["running"] = True
    try:
        run_master_data_sync()
    finally:
        _master_sync_state["running"] = False


@router.get("/sync")
def sync_master_data_route(
    background_tasks: BackgroundTasks,
):
    """
    Triggers sync_master_data.run_master_data_sync() in the background —
    pulls fresh rows from owner master, wagon type master, and org master
    tables, embeds new/changed rows, and upserts them into
    chatbot.master_data_chunks.
    """
    if _master_sync_state["running"]:
        return {"message": "Master data sync is already running, please wait for it to finish."}

    background_tasks.add_task(_run_master_data_sync_job)

    return {
        "message": "Master data sync started in background. Check server logs for progress."
    }