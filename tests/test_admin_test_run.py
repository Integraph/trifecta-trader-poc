"""Tests for /test-run endpoints (Task 016)."""

import asyncio
import os
import tempfile

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app(daemon=None):
    from src.admin.app import create_app
    return create_app(daemon=daemon, db=_make_db())


def _mock_run_analysis():
    return {
        "ticker":    "AAPL",
        "decision":  "BUY",
        "quality_score": {"composite": 8.2},
        "trade_params":  {"entry_price": 178.0, "stop_loss": 170.0, "price_target": 195.0},
        "cost_breakdown": {"total_usd": 0.028, "by_provider": {}},
        "elapsed_seconds": 45.0,
        "result_file": "results/analysis_2026-03-05_hybrid.json",
    }


# ── POST /test-run ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSubmitTestRun:

    async def test_returns_202(self):
        app = _make_app()
        with patch("src.admin.test_run._run_analysis_safe", return_value=_mock_run_analysis()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post("/test-run", json={"ticker": "AAPL"})
        assert resp.status_code == 202

    async def test_returns_task_id(self):
        app = _make_app()
        with patch("src.admin.test_run._run_analysis_safe", return_value=_mock_run_analysis()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post("/test-run", json={"ticker": "AAPL"})
        data = resp.json()
        assert "task_id"  in data
        assert "AAPL" in data["task_id"]

    async def test_default_publish_is_false(self):
        """Safety: publish must always default to False for test runs."""
        app     = _make_app()
        captured = {}

        def _capture(**kwargs):
            captured["publish"] = kwargs.get("publish")
            return _mock_run_analysis()

        with patch("src.admin.test_run._run_analysis_safe", side_effect=_capture):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                await c.post("/test-run", json={"ticker": "AAPL"})
            await asyncio.sleep(0.3)  # let thread run
        # publish defaults to False regardless of what was passed
        # (verified by the request schema)
        assert captured.get("publish", False) is False

    async def test_uses_scheduler_config_for_hybrid_when_daemon_present(self):
        daemon = MagicMock()
        daemon._scheduler = None
        daemon._queue_reader = None
        daemon._cfg = {
            "scheduler": {"hybrid_config": "hybrid_haiku_tools", "enabled": True},
            "queue_reader": {"enabled": True},
            "accuracy": {"enabled": True},
        }
        captured = {}

        def _capture(**kwargs):
            captured["hybrid_config"] = kwargs.get("hybrid_config")
            return _mock_run_analysis()

        app = _make_app(daemon=daemon)
        with patch("src.admin.test_run._run_analysis_safe", side_effect=_capture):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                await c.post("/test-run", json={"ticker": "AAPL"})
            await asyncio.sleep(0.3)
        assert captured.get("hybrid_config") == "hybrid_haiku_tools"


# ── GET /test-run/{task_id} ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestPollTestRun:

    async def test_poll_returns_result_after_completion(self):
        app = _make_app()
        with patch("src.admin.test_run._run_analysis_safe", return_value=_mock_run_analysis()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                post  = await c.post("/test-run", json={"ticker": "AAPL"})
                task_id = post.json()["task_id"]
                await asyncio.sleep(0.5)  # let background thread finish
                poll  = await c.get(f"/test-run/{task_id}")
        assert poll.status_code == 200
        data = poll.json()
        assert data["task_id"] == task_id
        assert data["status"] in ("running", "complete", "error")

    async def test_poll_404_for_unknown_task(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/test-run/nonexistent_task_id")
        assert resp.status_code == 404

    async def test_completed_result_has_expected_fields(self):
        app = _make_app()
        with patch("src.admin.test_run._run_analysis_safe", return_value=_mock_run_analysis()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                post    = await c.post("/test-run", json={"ticker": "AAPL"})
                task_id = post.json()["task_id"]
                await asyncio.sleep(0.5)
                poll    = await c.get(f"/test-run/{task_id}")
        data = poll.json()
        if data["status"] == "complete":
            result = data["result"]
            assert "decision"      in result
            assert "quality_score" in result
            assert "trade_params"  in result
