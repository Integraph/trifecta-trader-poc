"""Tests for /accuracy/* endpoints (Task 016)."""

import os
import tempfile
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app(db=None):
    from src.admin.app import create_app
    return create_app(daemon=None, db=db or _make_db())


def _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01") -> int:
    return db.upsert_analysis({
        "ticker": ticker, "trade_date": trade_date,
        "run_timestamp": "2026-03-01T09:00:00",
        "config": "hybrid_haiku_tools", "decision": "BUY",
        "quality_score": 9.0, "cost_usd": 0.05, "elapsed_seconds": 50.0,
        "stop_loss": 180.0, "price_target": 220.0, "entry_price": 195.0,
        "position_size_pct": 5.0, "risk_reward": 2.5, "actionable": 1,
        "portfolio_equity": 100000.0, "held_at_analysis": 0,
        "held_shares": 0, "held_avg_cost": None, "result_file": None,
    })


# ── /accuracy/summary ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAccuracySummary:

    async def test_summary_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/accuracy/summary")
        assert resp.status_code == 200

    async def test_summary_has_required_fields(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/accuracy/summary")
        data = resp.json()
        assert "total_signals"  in data
        assert "by_decision"    in data
        assert "by_quality_tier" in data

    async def test_summary_respects_days_param(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/accuracy/summary?days=7")
        assert resp.json()["period_days"] == 7


# ── /accuracy/ticker/{ticker} ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAccuracyTicker:

    async def test_ticker_report_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/accuracy/ticker/AAPL")
        assert resp.status_code == 200

    async def test_ticker_report_uppercased(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/accuracy/ticker/aapl")
        assert resp.json()["ticker"] == "AAPL"

    async def test_ticker_report_empty_for_unknown_ticker(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/accuracy/ticker/ZZZZZ")
        assert resp.json()["total_signals"] == 0


# ── /accuracy/update ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAccuracyUpdate:

    async def test_update_returns_summary(self):
        db = _make_db()
        app = _make_app(db)
        with patch("src.accuracy.updater.fetch_outcome_prices", return_value={}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post("/accuracy/update")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_pending" in data

    async def test_update_with_ticker_filter(self):
        db = _make_db()
        app = _make_app(db)
        with patch("src.accuracy.updater.fetch_outcome_prices", return_value={}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.post("/accuracy/update", json={"ticker": "AAPL"})
        assert resp.status_code == 200


# ── /accuracy/backfill ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAccuracyBackfill:

    async def test_backfill_returns_202(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/accuracy/backfill", json={"days_back": 7})
        assert resp.status_code == 202

    async def test_backfill_returns_task_id(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/accuracy/backfill", json={"days_back": 7})
        data = resp.json()
        assert "task_id"  in data
        assert "days_back" in data
        assert data["days_back"] == 7
