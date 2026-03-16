"""
Test Run Endpoint — /test-run

POST /test-run           Submit a single-ticker analysis (async, returns 202)
GET  /test-run/{task_id} Poll for the result
POST /test-run/ab        Submit two parallel analyses for A/B comparison
GET  /test-run/ab/{ab_id} Poll combined A/B status

publish defaults to False — test runs should never accidentally push to Supabase.
"""

import logging
from collections import OrderedDict
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.admin.task_manager import get_task_manager

logger = logging.getLogger(__name__)

test_run_router = APIRouter()

# ── A/B store (ephemeral, capped at 20 entries) ───────────────────────────────

_AB_STORE: OrderedDict = OrderedDict()  # ab_id → metadata dict
_AB_MAX = 20


# ── Request models ────────────────────────────────────────────────────────────

class TestRunRequest(BaseModel):
    ticker:        str
    hybrid_config: Optional[str] = None
    publish:       bool = False         # Safety default: never publish from test
    trade_date:    Optional[str] = None


class ABCompareRequest(BaseModel):
    ticker:     str
    trade_date: Optional[str] = None
    config_a:   str
    config_b:   str
    publish:    bool = False


# ── Single test run ───────────────────────────────────────────────────────────

@test_run_router.post("", status_code=202)
async def submit_test_run(body: TestRunRequest):
    """Submit a single-ticker test run. Returns 202 with task_id."""
    from src.admin.dependencies import get_daemon
    daemon        = get_daemon()
    hybrid_config = body.hybrid_config
    if hybrid_config is None and daemon is not None:
        hybrid_config = daemon._cfg.get("scheduler", {}).get("hybrid_config", "hybrid_haiku_tools")
    if hybrid_config is None:
        hybrid_config = "hybrid_haiku_tools"

    ticker     = body.ticker.upper()
    trade_date = body.trade_date or date.today().isoformat()
    now        = datetime.now(timezone.utc)
    task_id    = f"test_{ticker}_{now.strftime('%Y%m%d_%H%M%S')}"
    task_mgr   = get_task_manager()

    def _do_test_run():
        return _run_analysis_safe(
            ticker=ticker,
            hybrid_config=hybrid_config,
            publish=body.publish,
            trade_date=trade_date,
        )

    task_mgr.submit(task_id, _do_test_run)

    return {
        "task_id":    task_id,
        "status":     "running",
        "ticker":     ticker,
        "started_at": now.isoformat(),
    }


# ── A/B comparison ────────────────────────────────────────────────────────────

@test_run_router.post("/ab", status_code=202)
async def submit_ab_compare(body: ABCompareRequest):
    """Submit two parallel analyses for A/B comparison. Returns ab_id + both task_ids."""
    from src.hybrid_llm import CONFIGS

    ticker     = body.ticker.upper()
    trade_date = body.trade_date or date.today().isoformat()

    # Validate both configs exist
    missing = [c for c in [body.config_a, body.config_b] if c not in CONFIGS]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Config(s) not found: {missing}"},
        )

    now    = datetime.now(timezone.utc)
    stamp  = now.strftime("%Y%m%d_%H%M%S")
    ab_id  = f"ab_{ticker}_{stamp}"
    task_a = f"test_{ticker}_{stamp}_a"
    task_b = f"test_{ticker}_{stamp}_b"

    task_mgr = get_task_manager()

    def _run_a():
        return _run_analysis_safe(ticker, body.config_a, body.publish, trade_date)

    def _run_b():
        return _run_analysis_safe(ticker, body.config_b, body.publish, trade_date)

    task_mgr.submit(task_a, _run_a)
    task_mgr.submit(task_b, _run_b)

    # Store A/B metadata (cap at _AB_MAX)
    _AB_STORE[ab_id] = {
        "ab_id":      ab_id,
        "ticker":     ticker,
        "trade_date": trade_date,
        "config_a":   body.config_a,
        "config_b":   body.config_b,
        "task_id_a":  task_a,
        "task_id_b":  task_b,
        "started_at": now.isoformat(),
    }
    while len(_AB_STORE) > _AB_MAX:
        _AB_STORE.popitem(last=False)

    return {
        "ab_id":      ab_id,
        "task_id_a":  task_a,
        "task_id_b":  task_b,
        "status":     "running",
        "ticker":     ticker,
        "started_at": now.isoformat(),
    }


@test_run_router.get("/ab/{ab_id}")
async def get_ab_result(ab_id: str):
    """Poll the combined A/B comparison status."""
    meta = _AB_STORE.get(ab_id)
    if meta is None:
        raise HTTPException(status_code=404, detail={"error": "AB comparison not found"})

    task_mgr = get_task_manager()
    status_a = task_mgr.get_status(meta["task_id_a"]) or {}
    status_b = task_mgr.get_status(meta["task_id_b"]) or {}

    running_statuses = {"running"}
    overall = "running" if (
        status_a.get("status") in running_statuses or
        status_b.get("status") in running_statuses
    ) else "complete"

    return {
        "ab_id":      ab_id,
        "ticker":     meta["ticker"],
        "trade_date": meta["trade_date"],
        "status":     overall,
        "started_at": meta["started_at"],
        "config_a": {
            "name":    meta["config_a"],
            "task_id": meta["task_id_a"],
            "status":  status_a.get("status", "unknown"),
            "result":  status_a.get("result"),
            "error":   status_a.get("error"),
        },
        "config_b": {
            "name":    meta["config_b"],
            "task_id": meta["task_id_b"],
            "status":  status_b.get("status", "unknown"),
            "result":  status_b.get("result"),
            "error":   status_b.get("error"),
        },
    }


@test_run_router.get("/{task_id}")
async def get_test_run_result(task_id: str):
    """Poll for the result of a single test run."""
    task_mgr = get_task_manager()
    status   = task_mgr.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": "task_not_found"})
    return status


# ── Private helper ────────────────────────────────────────────────────────────

def _run_analysis_safe(
    ticker: str,
    hybrid_config: str,
    publish: bool,
    trade_date: str,
) -> dict:
    """Run a single analysis and return a sanitised result dict."""
    from src.run_analysis import run_analysis
    import time

    start = time.time()
    result = run_analysis(
        ticker=ticker,
        trade_date=trade_date,
        hybrid=hybrid_config,
        use_cache=True,
        cost_breakdown=True,
        publish=publish,
        debug=False,
    )
    elapsed = round(time.time() - start, 1)

    qs = result.get("quality_score", {}) or {}
    tp = result.get("trade_params", {}) or {}
    cb = result.get("cost_breakdown", {}) or {}

    return {
        "ticker":        ticker,
        "trade_date":    trade_date,
        "hybrid_config": hybrid_config,
        "decision":      result.get("decision"),
        "quality_score": qs,
        "trade_params":  tp,
        "cost_breakdown": {
            "total_usd":   cb.get("total_usd"),
            "by_provider": cb.get("by_provider", {}),
        },
        "elapsed_seconds": elapsed,
        "published":        publish,
        "result_file":      result.get("result_file"),
    }
