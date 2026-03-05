"""Tests for /queue/* endpoints (Task 016)."""

import json
import os
import tempfile

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app(tmp_queue: str, daemon_running: bool = True):
    from src.admin.app import create_app
    daemon = None
    if daemon_running:
        daemon = MagicMock()
        daemon._queue_reader = None
        daemon._scheduler    = None
        daemon._cfg = {
            "scheduler":    {"enabled": True},
            "queue_reader": {
                "enabled": True,
                "queue_dir": tmp_queue,
                "target_trader": "trifecta-trader",
                "max_retries": 2,
                "cooldown_seconds": 60,
                "poll_interval_seconds": 30,
            },
            "accuracy": {"enabled": True},
        }
    return create_app(daemon=daemon, db=_make_db())


def _make_candidate(ticker: str = "NVDA", priority: str = "high") -> dict:
    return {
        "scanner_id": "test_scanner",
        "timestamp":  "2026-03-05T14:30:00Z",
        "asset_type": "stock",
        "ticker":     ticker,
        "opportunity_score": 0.85,
        "catalysts":  ["volume_surge"],
        "signal_scores": {},
        "key_data":   {},
        "target_trader": "trifecta-trader",
        "priority":   priority,
        "status":     "pending",
        "retry_count": 0,
    }


# ── /queue/status ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestQueueStatus:

    async def test_status_returns_200(self, tmp_path):
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/queue/status")
        assert resp.status_code == 200

    async def test_status_has_counts(self, tmp_path):
        (tmp_path / "pending").mkdir()
        (tmp_path / "processing").mkdir()
        (tmp_path / "completed").mkdir()
        (tmp_path / "pending" / "test.json").write_text(json.dumps(_make_candidate()))

        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/queue/status")
        data = resp.json()
        assert data["counts"]["pending"] == 1
        assert data["counts"]["processing"] == 0


# ── /queue/pending ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestQueuePending:

    async def test_returns_empty_list_when_no_files(self, tmp_path):
        (tmp_path / "pending").mkdir()
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/queue/pending")
        assert resp.status_code == 200
        assert resp.json()["candidates"] == []

    async def test_returns_candidates_sorted_by_priority(self, tmp_path):
        pending = tmp_path / "pending"
        pending.mkdir()
        (pending / "a_low.json").write_text(json.dumps(_make_candidate("AAPL", "low")))
        (pending / "b_high.json").write_text(json.dumps(_make_candidate("NVDA", "high")))

        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/queue/pending")
        candidates = resp.json()["candidates"]
        assert len(candidates) == 2
        assert candidates[0]["priority"] == "high"
        assert candidates[1]["priority"] == "low"


# ── /queue/enqueue ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestQueueEnqueue:

    async def test_enqueue_creates_file(self, tmp_path):
        pending = tmp_path / "pending"
        pending.mkdir()
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/queue/enqueue", json={"ticker": "tsla", "priority": "high"})
        assert resp.status_code == 201
        files = list(pending.glob("*.json"))
        assert len(files) == 1

    async def test_enqueue_file_has_correct_format(self, tmp_path):
        pending = tmp_path / "pending"
        pending.mkdir()
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/queue/enqueue", json={"ticker": "nvda", "priority": "medium"})
        files = list(pending.glob("*.json"))
        msg   = json.loads(files[0].read_text())
        assert msg["ticker"]       == "NVDA"
        assert msg["target_trader"] == "trifecta-trader"
        assert msg["source"]       == "admin_api"
        assert msg["retry_count"]  == 0

    async def test_enqueue_ticker_uppercased(self, tmp_path):
        (tmp_path / "pending").mkdir()
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/queue/enqueue", json={"ticker": "aapl"})
        assert resp.json()["ticker"] == "AAPL"


# ── /queue/retry ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestQueueRetry:

    async def test_retry_moves_file_to_pending(self, tmp_path):
        for d in ("pending", "completed"):
            (tmp_path / d).mkdir()
        filename = "test_AAPL_20260305.json"
        msg = _make_candidate("AAPL")
        msg["retry_count"] = 2
        msg["status"]      = "completed"
        (tmp_path / "completed" / filename).write_text(json.dumps(msg))

        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(f"/queue/retry/{filename}")
        assert resp.status_code == 200
        assert (tmp_path / "pending" / filename).exists()
        assert not (tmp_path / "completed" / filename).exists()
        restored = json.loads((tmp_path / "pending" / filename).read_text())
        assert restored["retry_count"] == 0

    async def test_retry_404_unknown_file(self, tmp_path):
        for d in ("pending", "completed"):
            (tmp_path / d).mkdir()
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/queue/retry/nonexistent.json")
        assert resp.status_code == 404


# ── /queue/clear ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestQueueClear:

    async def test_clear_completed(self, tmp_path):
        completed = tmp_path / "completed"
        completed.mkdir()
        for i in range(3):
            (completed / f"file_{i}.json").write_text("{}")
        (tmp_path / "pending").mkdir()

        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.delete("/queue/clear?target=completed")
        assert resp.status_code == 200
        assert resp.json()["removed"] == 3
        assert len(list(completed.glob("*.json"))) == 0

    async def test_clear_invalid_target_returns_422(self, tmp_path):
        app = _make_app(str(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.delete("/queue/clear?target=invalid")
        assert resp.status_code == 422
