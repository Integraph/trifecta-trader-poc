"""
Test Run Endpoint — /test-run

POST /test-run           Submit a single-ticker analysis (async, returns 202)
GET  /test-run/{task_id} Poll for the result

This is the admin's "does it still work?" button. Runs a single-ticker
analysis using run_analysis() and returns the full result.

publish defaults to False — test runs should never accidentally push to Supabase.
"""

import logging
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.admin.task_manager import get_task_manager

logger = logging.getLogger(__name__)

test_run_router = APIRouter()


# ── Request model ─────────────────────────────────────────────────────────────

class TestRunRequest(BaseModel):
    ticker:        str
    hybrid_config: Optional[str] = None
    publish:       bool = False         # Safety default: never publish from test
    trade_date:    Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@test_run_router.post("", status_code=202)
async def submit_test_run(body: TestRunRequest):
    """Submit a single-ticker test run. Returns 202 with task_id.

    Poll GET /test-run/{task_id} for the result.
    """
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


@test_run_router.get("/{task_id}")
async def get_test_run_result(task_id: str):
    """Poll for the result of a test run."""
    task_mgr = get_task_manager()
    status   = task_mgr.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": "task_not_found"})
    return status


# ── Private ───────────────────────────────────────────────────────────────────

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
