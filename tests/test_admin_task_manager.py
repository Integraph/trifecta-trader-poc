"""Tests for TaskManager (Task 016)."""

import asyncio
import os
import tempfile
import time

import pytest
from httpx import AsyncClient, ASGITransport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_manager():
    from src.admin.task_manager import TaskManager
    return TaskManager(max_workers=2)


def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _make_app():
    from src.admin.app import create_app
    return create_app(daemon=None, db=_make_db())


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestTaskManagerUnit:

    def test_submit_creates_running_task(self):
        mgr = _fresh_manager()
        status = mgr.submit("t1", lambda: 42)
        assert status["task_id"] == "t1"
        assert status["status"]  == "running"

    def test_get_status_returns_none_for_unknown(self):
        mgr = _fresh_manager()
        assert mgr.get_status("unknown_id") is None

    def test_task_completes_with_result(self):
        mgr = _fresh_manager()
        mgr.submit("t2", lambda: {"answer": 42})
        time.sleep(0.3)
        status = mgr.get_status("t2")
        assert status["status"] == "complete"
        assert status["result"] == {"answer": 42}

    def test_task_captures_error(self):
        mgr = _fresh_manager()
        def _fail():
            raise ValueError("boom")
        mgr.submit("t3", _fail)
        time.sleep(0.3)
        status = mgr.get_status("t3")
        assert status["status"] == "error"
        assert "boom" in status["error"]

    def test_completed_at_set_on_finish(self):
        mgr = _fresh_manager()
        mgr.submit("t4", lambda: None)
        time.sleep(0.3)
        status = mgr.get_status("t4")
        assert status["completed_at"] is not None

    def test_list_tasks_returns_newest_first(self):
        mgr = _fresh_manager()
        mgr.submit("first",  lambda: 1)
        mgr.submit("second", lambda: 2)
        time.sleep(0.3)
        tasks = mgr.list_tasks()
        assert tasks[0]["task_id"] == "second"
        assert tasks[1]["task_id"] == "first"

    def test_list_tasks_respects_limit(self):
        mgr = _fresh_manager()
        for i in range(10):
            mgr.submit(f"task_{i}", lambda: None)
        time.sleep(0.3)
        tasks = mgr.list_tasks(limit=3)
        assert len(tasks) <= 3

    def test_get_result_returns_none_while_running(self):
        mgr = _fresh_manager()
        mgr.submit("slow", lambda: time.sleep(5))
        assert mgr.get_result("slow") is None

    def test_get_result_returns_value_when_complete(self):
        mgr = _fresh_manager()
        mgr.submit("fast", lambda: 99)
        time.sleep(0.3)
        assert mgr.get_result("fast") == 99


# ── /tasks/* endpoints ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTasksEndpoints:

    async def test_list_tasks_endpoint(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/tasks")
        assert resp.status_code == 200
        assert "tasks" in resp.json()

    async def test_get_task_endpoint_404(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/tasks/completely_unknown_id")
        assert resp.status_code == 404

    async def test_task_appears_in_list_after_submission(self):
        """Submit a test run; verify it shows up in /tasks."""
        app = _make_app()
        from unittest.mock import patch
        with patch("src.admin.test_run._run_analysis_safe", return_value={
            "ticker": "AAPL", "decision": "BUY",
            "quality_score": {}, "trade_params": {}, "cost_breakdown": {},
            "elapsed_seconds": 1.0, "result_file": None,
        }):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                post = await c.post("/test-run", json={"ticker": "AAPL"})
                task_id = post.json()["task_id"]
                await asyncio.sleep(0.4)
                resp = await c.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == task_id
