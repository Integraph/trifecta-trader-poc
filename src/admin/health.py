"""
Health Endpoint — GET /health

Returns pipeline operational status for the Platform UI health badge.

Color logic:
  green  → daemon running AND scheduler last run succeeded AND queue reader polling
  yellow → daemon running BUT: scheduler failed last run, OR accuracy errored,
           OR Supabase write_enabled=false, OR Ollama unreachable
  red    → daemon not running OR scheduler disabled while config says enabled
           OR queue reader stopped unexpectedly
"""

import logging
import os
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter

from src.admin.dependencies import get_daemon, get_db

logger = logging.getLogger(__name__)

health_router = APIRouter()


# ── Color logic ───────────────────────────────────────────────────────────────

def compute_health_color(
    daemon_running: bool,
    scheduler_last_result: Optional[str],
    scheduler_enabled_in_config: bool,
    scheduler_actually_running: bool,
    queue_reader_running: bool,
    queue_enabled_in_config: bool,
    accuracy_last_error: Optional[str],
    supabase_configured: bool,
    supabase_write_enabled: bool,
) -> tuple:
    """Pure function — compute color and overall status from subsystem states.

    Returns:
        (color: str, status: str)

    Note: Ollama reachability is reported in subsystems but not factored into
    the overall status — Ollama is an optional integration.
    """
    # RED conditions
    if not daemon_running:
        return "red", "unhealthy"
    if scheduler_enabled_in_config and not scheduler_actually_running:
        return "red", "degraded"
    if queue_enabled_in_config and not queue_reader_running and daemon_running:
        # Queue reader stopped unexpectedly
        return "red", "degraded"

    # YELLOW conditions — only flag things that are configured but broken
    if scheduler_last_result == "error":
        return "yellow", "degraded"
    if accuracy_last_error:
        return "yellow", "degraded"
    # Supabase: only degrade if credentials exist but write is explicitly disabled
    if supabase_configured and not supabase_write_enabled:
        return "yellow", "degraded"

    return "green", "healthy"


# ── Supabase config check ──────────────────────────────────────────────────────

def _get_supabase_status() -> dict:
    try:
        import yaml
        cfg_path = Path("config/supabase.yaml")
        if not cfg_path.exists():
            return {"configured": False, "write_enabled": False, "last_write": None}
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        has_creds = bool(
            os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")
        )
        return {
            "configured":    has_creds,
            "write_enabled": cfg.get("write_enabled", False) and has_creds,
            "last_write":    None,
        }
    except Exception as e:
        return {"configured": False, "write_enabled": False, "last_write": None, "error": str(e)}


# ── Ollama reachability ────────────────────────────────────────────────────────

def _check_ollama() -> dict:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        models = resp.json().get("models", [])
        first_model = models[0]["name"] if models else None
        return {"reachable": True, "model": first_model}
    except Exception:
        return {"reachable": False, "model": None}


# ── Queue counts ──────────────────────────────────────────────────────────────

def _get_queue_counts(queue_dir: str = "queue") -> dict:
    today = date.today().isoformat()
    pending_dir    = Path(queue_dir) / "pending"
    processing_dir = Path(queue_dir) / "processing"
    completed_dir  = Path(queue_dir) / "completed"

    def _count(d: Path) -> int:
        try:
            return len(list(d.glob("*.json")))
        except Exception:
            return 0

    def _count_today(d: Path) -> int:
        try:
            return sum(
                1 for f in d.glob("*.json")
                if today in f.name or today.replace("-", "") in f.name
            )
        except Exception:
            return 0

    return {
        "pending":         _count(pending_dir),
        "processing":      _count(processing_dir),
        "completed_today": _count_today(completed_dir),
    }


# ── Accuracy outcome counts ────────────────────────────────────────────────────

def _get_accuracy_counts(db) -> dict:
    try:
        with db._conn() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM signal_outcomes WHERE status IN ('pending','partial')"
            ).fetchone()[0]
            complete = conn.execute(
                "SELECT COUNT(*) FROM signal_outcomes WHERE status = 'complete'"
            ).fetchone()[0]
        return {"pending_outcomes": pending, "complete_outcomes": complete}
    except Exception:
        return {"pending_outcomes": 0, "complete_outcomes": 0}


# ── Dependency check endpoint ─────────────────────────────────────────────────

@health_router.get("/health/dependencies")
async def dependency_check():
    """Return install status of all optional packages.

    Useful for diagnosing missing-dependency errors before running analyses.
    """
    from src.admin.startup_checks import check_dependencies
    results  = check_dependencies()
    all_ok   = all(r["installed"] for r in results)
    missing  = [r["pip_name"] for r in results if not r["installed"]]
    return {
        "status":   "ok" if all_ok else "missing_packages",
        "all_ok":   all_ok,
        "missing":  missing,
        "packages": results,
    }


# ── Main endpoint ─────────────────────────────────────────────────────────────

@health_router.get("/health")
async def get_health():
    """Return overall pipeline health status.

    This endpoint powers the Platform UI health badge.
    Only `status`, `color`, and `mode` are read by the badge;
    subsystems are for the admin UI detail cards.

    mode:
        "standalone" — API started without a daemon (python -m src.admin.app)
        "full"        — API started via the daemon (python -m src.run_daemon --api)
    """
    daemon = get_daemon()
    db     = get_db()
    now    = datetime.now(timezone.utc).isoformat()
    pid    = os.getpid()

    daemon_running = daemon is not None
    mode           = "full" if daemon_running else "standalone"

    # ── Standalone short-circuit ──────────────────────────────────────────────
    # When no daemon is attached the subsystem checks are meaningless.
    # Report external integrations as info only; status is "standalone" (blue).
    if not daemon_running:
        supabase_status = _get_supabase_status()
        ollama_status   = _check_ollama()
        acc_counts      = _get_accuracy_counts(db)
        return {
            "status":         "standalone",
            "color":          "blue",
            "mode":           mode,
            "timestamp":      now,
            "uptime_seconds": None,
            "subsystems": {
                "daemon": {
                    "status":     "stopped",
                    "pid":        pid,
                    "start_time": None,
                },
                "scheduler":       {"status": "stopped", "last_run": None, "last_run_result": None, "next_run": None},
                "queue_reader":    {"status": "stopped", "pending_count": 0, "processing_count": 0, "completed_today": 0, "last_poll": None},
                "accuracy_updater": {"status": "idle", "last_run": None, "next_run": None,
                                     "pending_outcomes": acc_counts.get("pending_outcomes", 0),
                                     "complete_outcomes": acc_counts.get("complete_outcomes", 0)},
                "supabase": supabase_status,
                "ollama":   ollama_status,
            },
        }

    # ── Full daemon mode ──────────────────────────────────────────────────────
    uptime_seconds   = None
    start_time_iso   = None

    sched_status      = "stopped"
    sched_running     = False
    sched_last_run    = None
    sched_last_result = None
    sched_last_detail = {}
    sched_next_run    = None
    sched_enabled_cfg = True

    qr_running     = False
    qr_last_poll   = None
    queue_counts   = _get_queue_counts()
    qr_enabled_cfg = True

    acc_last_error = None
    acc_next_run   = None

    uptime_seconds = daemon._uptime_seconds()
    if daemon._start_time:
        start_time_iso = daemon._start_time.isoformat()

    # Scheduler state
    sched = daemon._scheduler
    sched_enabled_cfg = daemon._cfg.get("scheduler", {}).get("enabled", True)
    if sched is not None:
        sched_running = sched.is_running
        sched_status  = "running" if sched_running else "stopped"
        sched_last_run = getattr(sched, "_last_run", None)
        detail = getattr(sched, "_last_run_detail", None)
        if detail:
            sched_last_result = detail.get("result")
            sched_last_detail  = detail
        next_rt = sched.next_run_time()
        sched_next_run = next_rt.isoformat() if next_rt else None

    # Queue reader state
    qr = daemon._queue_reader
    qr_enabled_cfg = daemon._cfg.get("queue_reader", {}).get("enabled", True)
    if qr is not None:
        qr_running   = qr.is_running
        qr_last_poll = qr.last_poll.isoformat() if qr.last_poll else None

    queue_dir = daemon._cfg.get("queue_reader", {}).get("queue_dir", "queue")
    queue_counts = _get_queue_counts(queue_dir)

    # External checks
    supabase_status = _get_supabase_status()
    ollama_status   = _check_ollama()
    acc_counts      = _get_accuracy_counts(db)

    # Determine color
    color, overall_status = compute_health_color(
        daemon_running               = daemon_running,
        scheduler_last_result        = sched_last_result,
        scheduler_enabled_in_config  = sched_enabled_cfg,
        scheduler_actually_running   = sched_running,
        queue_reader_running         = qr_running,
        queue_enabled_in_config      = qr_enabled_cfg,
        accuracy_last_error          = acc_last_error,
        supabase_configured          = supabase_status.get("configured", False),
        supabase_write_enabled       = supabase_status.get("write_enabled", False),
    )

    return {
        "status":          overall_status,
        "color":           color,
        "mode":            mode,
        "timestamp":       now,
        "uptime_seconds":  uptime_seconds,
        "subsystems": {
            "daemon": {
                "status":     "running",
                "pid":        pid,
                "start_time": start_time_iso,
            },
            "scheduler": {
                "status":                   sched_status,
                "last_run":                 sched_last_run.isoformat() if sched_last_run else None,
                "last_run_result":          sched_last_detail.get("result"),
                "last_run_tickers":         sched_last_detail.get("tickers_processed"),
                "last_run_elapsed_seconds": sched_last_detail.get("elapsed_seconds"),
                "next_run":                 sched_next_run,
            },
            "queue_reader": {
                "status":           "running" if qr_running else "stopped",
                "pending_count":    queue_counts.get("pending", 0),
                "processing_count": queue_counts.get("processing", 0),
                "completed_today":  queue_counts.get("completed_today", 0),
                "last_poll":        qr_last_poll,
            },
            "accuracy_updater": {
                "status":            "idle",
                "last_run":          None,
                "next_run":          acc_next_run,
                "pending_outcomes":  acc_counts.get("pending_outcomes", 0),
                "complete_outcomes": acc_counts.get("complete_outcomes", 0),
            },
            "supabase": supabase_status,
            "ollama":   ollama_status,
        },
    }
