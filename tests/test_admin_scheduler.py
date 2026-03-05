"""Tests for /scheduler/* endpoints (Task 016)."""

import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
import tempfile, os


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app_with_daemon(sched=None):
    daemon = MagicMock()
    daemon._scheduler   = sched
    daemon._queue_reader = None
    daemon._cfg = {
        "scheduler": {
            "enabled": True,
            "watchlist_hour": 8,
            "watchlist_minute": 30,
            "timezone": "US/Eastern",
            "weekdays_only": True,
            "hybrid_config": "hybrid_haiku_tools",
            "publish": True,
            "watchlist": "default",
        },
        "queue_reader": {"enabled": True},
        "accuracy": {"enabled": True},
    }
    from src.admin.app import create_app
    return create_app(daemon=daemon, db=_make_db())


# ── /scheduler/status ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSchedulerStatus:

    async def test_status_returns_200_no_daemon(self):
        from src.admin.app import create_app
        app = create_app(daemon=None, db=_make_db())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/scheduler/status")
        assert resp.status_code == 200
        assert resp.json()["is_running"] is False

    async def test_status_returns_config(self):
        mock_sched = MagicMock()
        mock_sched.is_running = True
        mock_sched.next_run_time.return_value = None
        mock_sched._last_run_detail = {
            "timestamp": "2026-03-05T08:30:00Z",
            "result": "success",
            "tickers_processed": 8,
            "elapsed_seconds": 142.5,
            "decisions": {"AAPL": "BUY"},
            "error": None,
        }
        app = _make_app_with_daemon(sched=mock_sched)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/scheduler/status")
        data = resp.json()
        assert data["is_running"] is True
        assert data["config"]["watchlist_hour"] == 8
        assert data["last_run"]["tickers_processed"] == 8


# ── /scheduler/trigger ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSchedulerTrigger:

    async def test_trigger_returns_202(self):
        mock_sched = MagicMock()
        mock_sched.is_running = True
        mock_sched.run_now.return_value = {"tickers_processed": 2}
        app = _make_app_with_daemon(sched=mock_sched)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/scheduler/trigger")
        assert resp.status_code == 202

    async def test_trigger_returns_task_id(self):
        mock_sched = MagicMock()
        mock_sched.is_running = True
        mock_sched.run_now.return_value = {}
        app = _make_app_with_daemon(sched=mock_sched)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/scheduler/trigger")
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "running"

    async def test_trigger_409_when_no_scheduler(self):
        from src.admin.app import create_app
        app = create_app(daemon=None, db=_make_db())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/scheduler/trigger")
        assert resp.status_code == 409

    async def test_trigger_poll_by_task_id(self):
        mock_sched = MagicMock()
        mock_sched.is_running = True
        mock_sched.run_now.return_value = {"tickers_processed": 1}
        app = _make_app_with_daemon(sched=mock_sched)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            post_resp = await c.post("/scheduler/trigger")
            task_id   = post_resp.json()["task_id"]
            import asyncio; await asyncio.sleep(0.3)   # let thread complete
            poll_resp = await c.get(f"/scheduler/trigger/{task_id}")
        assert poll_resp.status_code == 200
        assert poll_resp.json()["task_id"] == task_id

    async def test_trigger_poll_404_unknown_id(self):
        mock_sched = MagicMock()
        mock_sched.is_running = True
        app = _make_app_with_daemon(sched=mock_sched)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/scheduler/trigger/nonexistent_id")
        assert resp.status_code == 404


# ── /scheduler/history ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSchedulerHistory:

    async def test_history_returns_runs(self):
        db = _make_db()
        # Seed two analyses on the same date
        for i, ticker in enumerate(["AAPL", "MSFT"]):
            db.upsert_analysis({
                "ticker": ticker, "trade_date": "2026-03-05",
                "run_timestamp": f"2026-03-05T09:0{i}:00",
                "config": "hybrid_haiku_tools", "decision": "BUY",
                "quality_score": 8.0, "cost_usd": 0.02, "elapsed_seconds": 40.0,
                "stop_loss": 180.0, "price_target": 200.0, "entry_price": 185.0,
                "position_size_pct": 3.0, "risk_reward": 2.0, "actionable": 1,
                "portfolio_equity": 100000.0, "held_at_analysis": 0,
                "held_shares": 0, "held_avg_cost": None, "result_file": None,
            })
        from src.admin.app import create_app
        app = create_app(daemon=None, db=db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/scheduler/history?days=30")
        data = resp.json()
        assert "runs" in data
        assert len(data["runs"]) >= 1
        run = data["runs"][0]
        assert run["tickers_processed"] == 2
        assert "decisions" in run
