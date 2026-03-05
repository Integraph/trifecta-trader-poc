"""Tests for /analyses/* endpoints (Task 016)."""

import os
import tempfile

import pytest
from httpx import AsyncClient, ASGITransport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app(db=None):
    from src.admin.app import create_app
    return create_app(daemon=None, db=db or _make_db())


def _seed_analysis(db, ticker="AAPL", trade_date="2026-03-05", decision="BUY"):
    return db.upsert_analysis({
        "ticker": ticker, "trade_date": trade_date,
        "run_timestamp": f"{trade_date}T09:00:00",
        "config": "hybrid_haiku_tools", "decision": decision,
        "quality_score": 8.0, "cost_usd": 0.03, "elapsed_seconds": 45.0,
        "stop_loss": 180.0, "price_target": 210.0, "entry_price": 190.0,
        "position_size_pct": 4.0, "risk_reward": 2.0, "actionable": 1,
        "portfolio_equity": 100000.0, "held_at_analysis": 0,
        "held_shares": 0, "held_avg_cost": None, "result_file": None,
    })


# ── /analyses/stats ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAnalysesStats:

    async def test_stats_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/stats")
        assert resp.status_code == 200

    async def test_stats_has_required_fields(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/stats")
        data = resp.json()
        for field in ("total_analyses", "total_cost_usd", "avg_quality_score",
                      "decision_breakdown", "unique_tickers"):
            assert field in data

    async def test_stats_counts_correctly(self):
        db = _make_db()
        _seed_analysis(db, "AAPL", "2026-03-05", "BUY")
        _seed_analysis(db, "MSFT", "2026-03-05", "SELL")
        _seed_analysis(db, "AAPL", "2026-03-06", "HOLD")

        app = _make_app(db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/stats")
        data = resp.json()
        assert data["total_analyses"]   == 3
        assert data["unique_tickers"]   == 2
        assert data["decision_breakdown"]["BUY"]  == 1
        assert data["decision_breakdown"]["SELL"] == 1
        assert data["decision_breakdown"]["HOLD"] == 1


# ── /analyses/recent ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAnalysesRecent:

    async def test_recent_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/recent")
        assert resp.status_code == 200

    async def test_recent_has_analyses_and_total(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/recent")
        data = resp.json()
        assert "analyses" in data
        assert "total"    in data

    async def test_recent_includes_outcome_status_column(self):
        db = _make_db()
        from datetime import date
        _seed_analysis(db, "AAPL", date.today().isoformat())
        app = _make_app(db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/recent?days=1")
        rows = resp.json()["analyses"]
        assert len(rows) >= 1
        assert "outcome_status" in rows[0]

    async def test_ticker_filter_works(self):
        db = _make_db()
        from datetime import date
        today = date.today().isoformat()
        _seed_analysis(db, "AAPL", today, "BUY")
        _seed_analysis(db, "MSFT", today, "SELL")

        app = _make_app(db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/recent?ticker=AAPL&days=7")
        rows = resp.json()["analyses"]
        assert all(r["ticker"] == "AAPL" for r in rows)


# ── /analyses/{id} ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAnalysisDetail:

    async def test_returns_404_for_unknown_id(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/analyses/99999")
        assert resp.status_code == 404

    async def test_returns_analysis_detail(self):
        db  = _make_db()
        aid = _seed_analysis(db)
        app = _make_app(db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/analyses/{aid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"]   == "AAPL"
        assert data["decision"] == "BUY"

    async def test_detail_includes_outcome_status(self):
        db  = _make_db()
        aid = _seed_analysis(db)
        app = _make_app(db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get(f"/analyses/{aid}")
        # outcome_status is NULL/None because no outcome was created
        assert "outcome_status" in resp.json()
