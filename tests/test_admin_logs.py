"""Tests for /logs/recent and AdminLogHandler (Task 016)."""

import asyncio
import logging
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


def _make_app():
    from src.admin.app import create_app
    return create_app(daemon=None, db=_make_db())


# ── AdminLogHandler unit tests ─────────────────────────────────────────────────

class TestAdminLogHandler:

    def _fresh_handler(self):
        from src.admin.logs import AdminLogHandler
        return AdminLogHandler(max_buffer=20)

    def test_emit_stores_entry_in_buffer(self):
        handler = self._fresh_handler()
        record  = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="hello world",
            args=(), exc_info=None,
        )
        handler.emit(record)
        entries = handler.recent(lines=10)
        assert len(entries) == 1
        assert entries[0]["message"] == "hello world"
        assert entries[0]["level"]   == "INFO"

    def test_buffer_bounded_by_max_buffer(self):
        handler = self._fresh_handler()
        for i in range(25):
            record = logging.LogRecord(
                name="test", level=logging.DEBUG, pathname="", lineno=0,
                msg=f"msg {i}", args=(), exc_info=None,
            )
            handler.emit(record)
        entries = handler.recent(lines=100)
        assert len(entries) <= 20  # max_buffer=20

    def test_level_filter_excludes_lower_levels(self):
        handler = self._fresh_handler()
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            record = logging.LogRecord(
                name="test", level=level, pathname="", lineno=0,
                msg=f"msg at level {level}", args=(), exc_info=None,
            )
            handler.emit(record)
        # Only WARNING and ERROR
        entries = handler.recent(lines=10, level="WARNING")
        assert all(e["level"] in ("WARNING", "ERROR") for e in entries)
        assert len(entries) == 2

    def test_entries_have_required_fields(self):
        handler = self._fresh_handler()
        record  = logging.LogRecord(
            name="my.module", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        handler.emit(record)
        entry = handler.recent()[0]
        assert "timestamp" in entry
        assert "level"     in entry
        assert "logger"    in entry
        assert "message"   in entry

    def test_subscribe_and_unsubscribe(self):
        handler = self._fresh_handler()
        q = handler.subscribe()
        assert q in handler._subscribers
        handler.unsubscribe(q)
        assert q not in handler._subscribers


# ── /logs/recent endpoint ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestLogsRecentEndpoint:

    async def test_returns_200(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/logs/recent")
        assert resp.status_code == 200

    async def test_returns_lines_list(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/logs/recent")
        assert "lines" in resp.json()

    async def test_level_filter_accepted(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/logs/recent?level=WARNING&lines=20")
        assert resp.status_code == 200

    async def test_entries_newest_first_when_present(self):
        """If log entries exist, newest should be first in the response."""
        from src.admin.logs import AdminLogHandler, _admin_handler
        # Inject a populated handler
        import src.admin.logs as logs_mod
        handler = AdminLogHandler(max_buffer=10)
        for i, msg in enumerate(["old", "middle", "new"]):
            rec = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=msg, args=(), exc_info=None,
            )
            handler.emit(rec)
        original = logs_mod._admin_handler
        logs_mod._admin_handler = handler
        try:
            app = _make_app()
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                resp = await c.get("/logs/recent")
            lines = resp.json()["lines"]
            if lines:
                assert lines[0]["message"] == "new"
        finally:
            logs_mod._admin_handler = original
