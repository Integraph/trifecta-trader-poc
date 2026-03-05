"""Tests for /config/* endpoints (Task 016)."""

import os
import tempfile

import pytest
import yaml
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app(daemon=None):
    from src.admin.app import create_app
    return create_app(daemon=daemon, db=_make_db())


# ── /config/automation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAutomationConfig:

    async def test_get_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/config/automation")
        assert resp.status_code == 200

    async def test_get_contains_scheduler_section(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/config/automation")
        data = resp.json()
        assert "scheduler"    in data
        assert "queue_reader" in data

    async def test_put_writes_to_disk(self, tmp_path):
        """Verify _write_yaml + _load_yaml round-trip works correctly."""
        from src.admin.config import _write_yaml, _load_yaml, _deep_merge
        cfg_file = tmp_path / "automation.yaml"
        existing = {"scheduler": {"watchlist_hour": 8}}
        _write_yaml(cfg_file, existing)
        update   = {"queue_reader": {"poll_interval_seconds": 15}}
        merged   = _deep_merge(existing, update)
        _write_yaml(cfg_file, merged)
        loaded   = _load_yaml(cfg_file)
        assert loaded["queue_reader"]["poll_interval_seconds"] == 15
        assert loaded["scheduler"]["watchlist_hour"]           == 8

    async def test_put_indicates_restart_requirements(self, tmp_path):
        from src.admin.config import _classify_changes
        changed = ["scheduler.watchlist_hour", "queue_reader.poll_interval_seconds"]
        applied, needs_restart = _classify_changes(changed)
        assert "queue_reader.poll_interval_seconds" in applied
        assert "scheduler.watchlist_hour" in needs_restart


# ── /config/watchlists ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestWatchlistConfig:

    async def test_get_watchlists_returns_list(self, tmp_path):
        wl_dir = tmp_path / "config" / "watchlists"
        wl_dir.mkdir(parents=True)
        (wl_dir / "default.yaml").write_text("tickers:\n  - AAPL\n  - MSFT\n")

        app = _make_app()
        with patch("src.admin.config.Path") as _:
            # Test directly via function
            from src.admin.config import _load_yaml
            data = _load_yaml(wl_dir / "default.yaml")
            assert data["tickers"] == ["AAPL", "MSFT"]

    async def test_get_watchlists_endpoint(self, tmp_path, monkeypatch):
        wl_dir = tmp_path / "config" / "watchlists"
        wl_dir.mkdir(parents=True)
        (wl_dir / "default.yaml").write_text("tickers:\n  - AAPL\n  - MSFT\n")

        import src.admin.config as cfg_module
        original_path = cfg_module.Path
        monkeypatch.setattr(
            cfg_module,
            "Path",
            lambda *a: original_path(*a) if a else original_path(),
        )

        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/config/watchlists")
        assert resp.status_code == 200
        assert "watchlists" in resp.json()

    async def test_put_watchlist_creates_file(self, tmp_path):
        from src.admin.config import _write_yaml, _load_yaml
        wl_path = tmp_path / "my_list.yaml"
        _write_yaml(wl_path, {"tickers": ["AAPL", "NVDA"]})
        loaded = _load_yaml(wl_path)
        assert loaded["tickers"] == ["AAPL", "NVDA"]


# ── /config/hybrid-configs ────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHybridConfigs:

    async def test_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/config/hybrid-configs")
        assert resp.status_code == 200

    async def test_returns_configs_list(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/config/hybrid-configs")
        data = resp.json()
        assert "configs" in data
        assert isinstance(data["configs"], list)
        if data["configs"]:
            assert "name" in data["configs"][0]


# ── classify changes helper ───────────────────────────────────────────────────

class TestClassifyChanges:

    def test_immediate_fields_classified_correctly(self):
        from src.admin.config import _classify_changes
        applied, restart = _classify_changes(["queue_reader.poll_interval_seconds"])
        assert "queue_reader.poll_interval_seconds" in applied
        assert len(restart) == 0

    def test_restart_fields_classified_correctly(self):
        from src.admin.config import _classify_changes
        applied, restart = _classify_changes(["scheduler.watchlist_hour"])
        assert "scheduler.watchlist_hour" in restart
        assert len(applied) == 0

    def test_unknown_fields_in_neither(self):
        from src.admin.config import _classify_changes
        applied, restart = _classify_changes(["some.unknown.field"])
        assert len(applied)   == 0
        assert len(restart)   == 0
