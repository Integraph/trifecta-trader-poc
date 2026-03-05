"""
Queue Endpoints — /queue/*

GET    /queue/status           Queue reader state and file counts
GET    /queue/pending          Contents of pending queue files
GET    /queue/completed        Completed files (with optional day filter)
POST   /queue/enqueue          Manually add a ticker to the queue
POST   /queue/retry/{filename} Move a failed file back to pending
DELETE /queue/clear            Remove files from a queue directory
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.admin.dependencies import get_daemon, get_db

logger = logging.getLogger(__name__)

queue_router = APIRouter()

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _queue_dir(daemon=None) -> str:
    if daemon and daemon._cfg:
        return daemon._cfg.get("queue_reader", {}).get("queue_dir", "queue")
    return "queue"


def _count_files(d: Path) -> int:
    try:
        return len(list(d.glob("*.json")))
    except Exception:
        return 0


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Request models ────────────────────────────────────────────────────────────

class EnqueueRequest(BaseModel):
    ticker:   str
    priority: str = "high"
    reason:   str = "Manual admin request"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@queue_router.get("/status")
async def get_queue_status():
    """Return queue reader state and file counts."""
    daemon = get_daemon()
    qdir   = _queue_dir(daemon)
    qr     = daemon._queue_reader if daemon else None
    cfg    = daemon._cfg.get("queue_reader", {}) if daemon else {}

    pending_dir    = Path(qdir) / "pending"
    processing_dir = Path(qdir) / "processing"
    completed_dir  = Path(qdir) / "completed"

    return {
        "enabled":            cfg.get("enabled", True),
        "is_running":         qr.is_running if qr else False,
        "poll_interval_seconds": cfg.get("poll_interval_seconds", 30),
        "last_poll":          qr.last_poll.isoformat() if qr and qr.last_poll else None,
        "counts": {
            "pending":    _count_files(pending_dir),
            "processing": _count_files(processing_dir),
            "completed":  _count_files(completed_dir),
            "failed":     0,  # files with retry_count >= max tracked in DB
        },
        "config": {
            "queue_dir":       qdir,
            "target_trader":   cfg.get("target_trader", "trifecta-trader"),
            "max_retries":     cfg.get("max_retries", 2),
            "cooldown_seconds": cfg.get("cooldown_seconds", 60),
        },
    }


@queue_router.get("/pending")
async def get_pending_queue():
    """Return contents of all pending queue files, sorted by priority."""
    daemon      = get_daemon()
    qdir        = _queue_dir(daemon)
    pending_dir = Path(qdir) / "pending"

    if not pending_dir.exists():
        return {"candidates": []}

    files = sorted(
        pending_dir.glob("*.json"),
        key=lambda p: (
            PRIORITY_ORDER.get((_read_json(p) or {}).get("priority", "medium"), 1),
            p.name,
        ),
    )

    candidates = []
    for path in files:
        msg = _read_json(path)
        if msg is None:
            continue
        candidates.append({
            "filename":         path.name,
            "ticker":           msg.get("ticker"),
            "priority":         msg.get("priority", "medium"),
            "opportunity_score": msg.get("opportunity_score"),
            "catalysts":        msg.get("catalysts", []),
            "asset_type":       msg.get("asset_type", "stock"),
            "retry_count":      msg.get("retry_count", 0),
            "queued_at":        msg.get("timestamp"),
            "source":           msg.get("source", "scanner"),
        })

    return {"candidates": candidates}


@queue_router.get("/completed")
async def get_completed_queue(
    days:  int = Query(default=1, ge=1, le=30),
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return completed queue files."""
    daemon         = get_daemon()
    qdir           = _queue_dir(daemon)
    completed_dir  = Path(qdir) / "completed"

    if not completed_dir.exists():
        return {"completed": []}

    files = sorted(completed_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for path in files[:limit]:
        msg = _read_json(path)
        if msg:
            results.append({
                "filename":       path.name,
                "ticker":         msg.get("ticker"),
                "completed_at":   msg.get("completed_at"),
                "analysis_result": msg.get("analysis_result", {}),
            })

    return {"completed": results}


@queue_router.post("/enqueue", status_code=201)
async def enqueue_ticker(body: EnqueueRequest):
    """Manually add a ticker to the pending queue."""
    daemon      = get_daemon()
    qdir        = _queue_dir(daemon)
    pending_dir = Path(qdir) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    ticker    = body.ticker.upper()
    now       = datetime.now(timezone.utc)
    ts        = now.strftime("%Y%m%d_%H%M%S")
    filename  = f"admin_{ticker}_{ts}.json"
    file_path = pending_dir / filename

    message = {
        "scanner_id":       "admin_manual",
        "timestamp":        now.isoformat(),
        "asset_type":       "stock",
        "ticker":           ticker,
        "opportunity_score": 0.0,
        "catalysts":        [],
        "signal_scores":    {},
        "key_data":         {},
        "target_trader":    "trifecta-trader",
        "priority":         body.priority,
        "status":           "pending",
        "source":           "admin_api",
        "reason":           body.reason,
        "retry_count":      0,
    }

    try:
        file_path.write_text(json.dumps(message, indent=2))
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

    return {
        "filename": filename,
        "ticker":   ticker,
        "priority": body.priority,
        "queued_at": now.isoformat(),
    }


@queue_router.post("/retry/{filename}")
async def retry_failed(filename: str):
    """Move a file from completed/ or processing/ back to pending/ with retry_count reset."""
    daemon      = get_daemon()
    qdir        = _queue_dir(daemon)
    base        = Path(qdir)

    # Search in completed and processing
    source = None
    for subdir in ("completed", "processing"):
        candidate = base / subdir / filename
        if candidate.exists():
            source = candidate
            break

    if source is None:
        raise HTTPException(status_code=404, detail={"error": "file_not_found", "filename": filename})

    try:
        msg = _read_json(source)
        if msg is None:
            raise ValueError("Cannot parse file")
        msg["retry_count"] = 0
        msg["status"]      = "pending"
        msg.pop("last_error", None)
        msg.pop("completed_at", None)

        dest = base / "pending" / filename
        dest.write_text(json.dumps(msg, indent=2, default=str))
        source.unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

    return {"status": "ok", "filename": filename, "moved_to": "pending"}


@queue_router.delete("/clear")
async def clear_queue(target: str = Query(default="completed")):
    """Remove files from the specified queue subdirectory.

    target: one of 'pending', 'completed', 'processing', 'all'
    """
    daemon = get_daemon()
    qdir   = _queue_dir(daemon)
    base   = Path(qdir)

    if target == "all":
        targets = ["pending", "completed", "processing"]
    elif target in ("pending", "completed", "processing"):
        targets = [target]
    else:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_target", "message": "target must be pending/completed/processing/all"},
        )

    removed = 0
    for subdir in targets:
        d = base / subdir
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass

    return {"removed": removed, "target": target}
