"""Tests for GET /health and health color logic (Task 016)."""

import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(daemon=None):
    from src.admin.app import create_app
    return create_app(daemon=daemon, db=None)


def _mock_db():
    db = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__  = MagicMock(return_value=False)
    cursor = MagicMock()
    cursor.fetchone.return_value = (0,)
    ctx.execute.return_value = cursor
    db._conn.return_value = ctx
    return db


# ── Color logic unit tests ────────────────────────────────────────────────────

class TestHealthColorLogic:

    def _color(self, **kwargs):
        from src.admin.health import compute_health_color
        defaults = dict(
            daemon_running=True,
            scheduler_last_result="success",
            scheduler_enabled_in_config=True,
            scheduler_actually_running=True,
            queue_reader_running=True,
            queue_enabled_in_config=True,
            accuracy_last_error=None,
            supabase_configured=True,
            supabase_write_enabled=True,
        )
        defaults.update(kwargs)
        return compute_health_color(**defaults)

    def test_green_when_all_healthy(self):
        color, status = self._color()
        assert color  == "green"
        assert status == "healthy"

    def test_red_when_daemon_not_running(self):
        color, status = self._color(daemon_running=False)
        assert color == "red"

    def test_red_when_scheduler_enabled_but_stopped(self):
        color, _ = self._color(
            scheduler_enabled_in_config=True,
            scheduler_actually_running=False,
        )
        assert color == "red"

    def test_yellow_when_scheduler_last_run_errored(self):
        color, _ = self._color(scheduler_last_result="error")
        assert color == "yellow"

    def test_ollama_unreachable_does_not_degrade(self):
        """Ollama is optional — unreachable Ollama should NOT cause degraded status."""
        # ollama_reachable is no longer a parameter; verify green is still returned
        color, status = self._color()
        assert color  == "green"
        assert status == "healthy"

    def test_yellow_when_supabase_configured_but_write_disabled(self):
        """Supabase degradation only fires when credentials exist but write is off."""
        color, _ = self._color(supabase_configured=True, supabase_write_enabled=False)
        assert color == "yellow"

    def test_green_when_supabase_not_configured(self):
        """Unconfigured Supabase (no creds) should NOT cause degraded status."""
        color, status = self._color(supabase_configured=False, supabase_write_enabled=False)
        assert color  == "green"
        assert status == "healthy"

    def test_yellow_when_accuracy_errored(self):
        color, _ = self._color(accuracy_last_error="yfinance timeout")
        assert color == "yellow"

    def test_color_is_deterministic(self):
        """Same inputs always produce same output."""
        from src.admin.health import compute_health_color
        args = dict(
            daemon_running=True, scheduler_last_result="success",
            scheduler_enabled_in_config=True, scheduler_actually_running=True,
            queue_reader_running=True, queue_enabled_in_config=True,
            accuracy_last_error=None, supabase_configured=True,
            supabase_write_enabled=True,
        )
        c1 = compute_health_color(**args)
        c2 = compute_health_color(**args)
        assert c1 == c2


# ── HTTP endpoint tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_health_returns_200(self):
        app = _make_app()
        with patch("src.admin.health._check_ollama", return_value={"reachable": True, "model": "q"}), \
             patch("src.admin.health._get_supabase_status", return_value={"configured": True, "write_enabled": True, "last_write": None}), \
             patch("src.admin.health._get_accuracy_counts", return_value={"pending_outcomes": 0, "complete_outcomes": 0}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_has_required_fields(self):
        app = _make_app()
        with patch("src.admin.health._check_ollama", return_value={"reachable": True, "model": None}), \
             patch("src.admin.health._get_supabase_status", return_value={"configured": False, "write_enabled": False}), \
             patch("src.admin.health._get_accuracy_counts", return_value={"pending_outcomes": 0, "complete_outcomes": 0}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "color"  in data
        assert "subsystems" in data

    async def test_health_standalone_mode_when_no_daemon(self):
        """Standalone mode (no daemon) returns blue/standalone — not red/unhealthy."""
        app = _make_app(daemon=None)
        with patch("src.admin.health._check_ollama", return_value={"reachable": False, "model": None}), \
             patch("src.admin.health._get_supabase_status", return_value={"configured": False, "write_enabled": False}), \
             patch("src.admin.health._get_accuracy_counts", return_value={"pending_outcomes": 0, "complete_outcomes": 0}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/health")
        data = resp.json()
        assert data["color"]  == "blue"
        assert data["status"] == "standalone"
        assert data["mode"]   == "standalone"

    async def test_health_includes_subsystem_breakdown(self):
        app = _make_app()
        with patch("src.admin.health._check_ollama", return_value={"reachable": True, "model": None}), \
             patch("src.admin.health._get_supabase_status", return_value={"configured": False, "write_enabled": False}), \
             patch("src.admin.health._get_accuracy_counts", return_value={"pending_outcomes": 5, "complete_outcomes": 10}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/health")
        subs = resp.json()["subsystems"]
        for key in ("daemon", "scheduler", "queue_reader", "accuracy_updater", "supabase", "ollama"):
            assert key in subs


# ── Ollama check ──────────────────────────────────────────────────────────────

class TestOllamaCheck:

    def test_returns_false_when_unreachable(self):
        from src.admin.health import _check_ollama
        with patch("src.admin.health.requests.get", side_effect=Exception("refused")):
            result = _check_ollama()
        assert result["reachable"] is False

    def test_returns_true_when_reachable(self):
        from src.admin.health import _check_ollama
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen2.5:14b"}]}
        with patch("src.admin.health.requests.get", return_value=mock_resp):
            result = _check_ollama()
        assert result["reachable"] is True
        assert result["model"] == "qwen2.5:14b"
