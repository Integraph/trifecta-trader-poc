"""
Accuracy Endpoints — /accuracy/*

GET  /accuracy/summary          AccuracyReporter.summary() as JSON
GET  /accuracy/ticker/{ticker}  Per-ticker accuracy report
POST /accuracy/update           Trigger immediate update cycle (synchronous)
POST /accuracy/backfill         Trigger backfill via TaskManager (async)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.admin.dependencies import get_db
from src.admin.task_manager import get_task_manager

logger = logging.getLogger(__name__)

accuracy_router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class UpdateRequest(BaseModel):
    ticker: Optional[str] = None


class BackfillRequest(BaseModel):
    days_back: int = 30


# ── Endpoints ─────────────────────────────────────────────────────────────────

@accuracy_router.get("/summary")
async def get_accuracy_summary(days: int = Query(default=30, ge=1, le=365)):
    """Return accuracy summary from AccuracyReporter.summary()."""
    db = get_db()
    try:
        from src.accuracy.reporter import AccuracyReporter
        reporter = AccuracyReporter(db)
        data = reporter.summary(days=days)

        # Add pending count for convenience
        with db._conn() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM signal_outcomes WHERE status IN ('pending','partial')"
            ).fetchone()[0]
        data["pending_outcomes"] = pending
        return data
    except Exception as e:
        logger.error("Accuracy summary failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@accuracy_router.get("/ticker/{ticker}")
async def get_accuracy_ticker(ticker: str):
    """Return ticker-specific accuracy report."""
    db = get_db()
    try:
        from src.accuracy.reporter import AccuracyReporter
        reporter = AccuracyReporter(db)
        return reporter.ticker_report(ticker.upper())
    except Exception as e:
        logger.error("Ticker accuracy report failed for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@accuracy_router.post("/update")
async def trigger_accuracy_update(body: Optional[UpdateRequest] = None):
    """Trigger an immediate accuracy update cycle.

    This is synchronous (typically fast — a few seconds for 10-20 pending outcomes).
    Use /accuracy/backfill for the async/long-running backfill operation.
    """
    db = get_db()
    req = body or UpdateRequest()
    try:
        from src.accuracy.updater import AccuracyUpdater
        updater = AccuracyUpdater(db=db)
        result  = updater.run_update(ticker=req.ticker)
        return result
    except Exception as e:
        logger.error("Accuracy update failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@accuracy_router.post("/backfill", status_code=202)
async def trigger_backfill(body: Optional[BackfillRequest] = None):
    """Trigger an accuracy backfill (async task — returns 202 with task_id).

    Poll GET /tasks/{task_id} for the result.
    """
    db   = get_db()
    req  = body or BackfillRequest()
    now  = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    task_id  = f"backfill_{now.strftime('%Y%m%d_%H%M%S')}"
    task_mgr = get_task_manager()

    def _do_backfill():
        from src.accuracy.updater import AccuracyUpdater
        return AccuracyUpdater(db=db).backfill(days_back=req.days_back)

    task_mgr.submit(task_id, _do_backfill)

    return {
        "task_id":    task_id,
        "status":     "running",
        "days_back":  req.days_back,
        "started_at": now.isoformat(),
    }
