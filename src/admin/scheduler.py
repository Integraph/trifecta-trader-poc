"""
Scheduler Endpoints — /scheduler/*

GET  /scheduler/status       Current scheduler state, last run, config
POST /scheduler/trigger      Start an immediate watchlist scan (async)
GET  /scheduler/trigger/{id} Poll for trigger result
GET  /scheduler/history      Recent scheduler run history from DB
"""

import logging
from datetime import datetime, date, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from src.admin.dependencies import get_daemon, get_db, require_daemon
from src.admin.task_manager import get_task_manager

logger = logging.getLogger(__name__)

scheduler_router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class TriggerRequest(BaseModel):
    watchlist: str = "default"
    publish:   bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@scheduler_router.get("/status")
async def get_scheduler_status(daemon=None):
    """Return current scheduler state, config, and last run details."""
    daemon = get_daemon()
    if daemon is None:
        return {
            "enabled":    False,
            "is_running": False,
            "schedule":   "unknown",
            "next_run":   None,
            "last_run":   None,
            "config":     {},
        }

    sched = daemon._scheduler
    cfg   = daemon._cfg.get("scheduler", {})

    if sched is None:
        return {
            "enabled":    cfg.get("enabled", True),
            "is_running": False,
            "schedule":   _fmt_schedule(cfg),
            "next_run":   None,
            "last_run":   None,
            "config":     cfg,
        }

    detail = getattr(sched, "_last_run_detail", None)
    next_rt = sched.next_run_time()

    return {
        "enabled":    cfg.get("enabled", True),
        "is_running": sched.is_running,
        "schedule":   _fmt_schedule(cfg),
        "next_run":   next_rt.isoformat() if next_rt else None,
        "last_run":   detail,
        "config":     cfg,
    }


@scheduler_router.post("/trigger", status_code=202)
async def trigger_scan(body: Optional[TriggerRequest] = None):
    """Trigger an immediate watchlist scan in a background thread.

    Returns 202 Accepted with a task_id. Poll GET /scheduler/trigger/{task_id}.
    """
    daemon = get_daemon()
    if daemon is None or daemon._scheduler is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error":   "scheduler_not_running",
                "message": "Cannot trigger scan: scheduler is not running.",
            },
        )

    req         = body or TriggerRequest()
    now         = datetime.now(timezone.utc)
    task_id     = f"scan_{now.strftime('%Y%m%d_%H%M%S')}"
    task_mgr    = get_task_manager()

    def _do_scan():
        return daemon._scheduler.run_now()

    task_mgr.submit(task_id, _do_scan)

    return {
        "task_id":    task_id,
        "status":     "running",
        "started_at": now.isoformat(),
    }


@scheduler_router.get("/trigger/{task_id}")
async def get_trigger_result(task_id: str):
    """Poll for the result of a triggered scan."""
    task_mgr = get_task_manager()
    status   = task_mgr.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": "task_not_found"})
    return status


@scheduler_router.get("/history")
async def get_scheduler_history(
    days:  int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return recent scheduler run history grouped by trade_date."""
    db     = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    sql = """
        SELECT trade_date,
               COUNT(*)         AS tickers_processed,
               SUM(CASE decision WHEN 'BUY'  THEN 1 ELSE 0 END) AS buy_count,
               SUM(CASE decision WHEN 'SELL' THEN 1 ELSE 0 END) AS sell_count,
               SUM(CASE decision WHEN 'HOLD' THEN 1 ELSE 0 END) AS hold_count,
               AVG(quality_score)     AS avg_quality_score,
               SUM(elapsed_seconds)   AS total_elapsed_seconds,
               SUM(cost_usd)          AS total_cost_usd
          FROM analyses
         WHERE trade_date >= ?
         GROUP BY trade_date
         ORDER BY trade_date DESC
         LIMIT ?
    """
    with db._conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, (cutoff, limit)).fetchall()]

    runs = []
    for row in rows:
        runs.append({
            "trade_date":           row["trade_date"],
            "tickers_processed":    row["tickers_processed"],
            "decisions": {
                "BUY":  row["buy_count"],
                "SELL": row["sell_count"],
                "HOLD": row["hold_count"],
            },
            "avg_quality_score":    round(row["avg_quality_score"] or 0, 2),
            "total_elapsed_seconds": round(row["total_elapsed_seconds"] or 0, 1),
            "total_cost_usd":       round(row["total_cost_usd"] or 0, 4),
        })

    return {"runs": runs}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_schedule(cfg: dict) -> str:
    hour      = cfg.get("watchlist_hour", 8)
    minute    = cfg.get("watchlist_minute", 30)
    tz        = cfg.get("timezone", "US/Eastern")
    weekdays  = cfg.get("weekdays_only", True)
    days_str  = "weekdays" if weekdays else "daily"
    return f"{hour:02d}:{minute:02d} {tz} {days_str}"
