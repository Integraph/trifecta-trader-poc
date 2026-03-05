"""
Async Task Manager — executes long-running admin operations in background threads
and provides polling-based status checks.

Used by: /scheduler/trigger, /test-run, /accuracy/backfill

Pattern:
    1. POST endpoint submits task → returns 202 with task_id
    2. Client polls GET /tasks/{task_id} or endpoint-specific poll URL
    3. TaskManager stores status + result keyed by task_id

Concurrency: ThreadPoolExecutor with max_workers=2 (one test run + one admin op).
"""

import logging
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_WORKERS    = 2
_MAX_STORED     = 100  # cap history to avoid unbounded memory


class TaskManager:
    """Manages background tasks for the Admin API."""

    def __init__(self, max_workers: int = _MAX_WORKERS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="admin_task")
        self._tasks: OrderedDict[str, dict] = OrderedDict()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, task_id: str, fn: Callable, *args, **kwargs) -> dict:
        """Submit a background task.

        Args:
            task_id: Unique ID for this task.
            fn: Callable to run in a background thread.
            *args / **kwargs: Forwarded to fn.

        Returns:
            Initial status dict (status='running').
        """
        status = {
            "task_id":    task_id,
            "status":     "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result":     None,
            "error":      None,
        }
        self._tasks[task_id] = status

        # Trim old tasks
        while len(self._tasks) > _MAX_STORED:
            self._tasks.popitem(last=False)

        # Capture initial state BEFORE submitting (thread may complete instantly)
        initial = dict(status)
        self._executor.submit(self._run, task_id, fn, *args, **kwargs)
        return initial

    def get_status(self, task_id: str) -> Optional[dict]:
        """Return current task status, or None if task_id is unknown."""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[Any]:
        """Return the result of a completed task, or None."""
        task = self._tasks.get(task_id)
        if task and task["status"] == "complete":
            return task["result"]
        return None

    def list_tasks(self, limit: int = 20) -> list:
        """Return recent tasks, newest first."""
        tasks = list(self._tasks.values())
        tasks.reverse()
        return tasks[:limit]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self, task_id: str, fn: Callable, *args, **kwargs) -> None:
        """Execute fn in the thread pool and update status."""
        try:
            result = fn(*args, **kwargs)
            self._tasks[task_id].update({
                "status":       "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "result":       result,
            })
        except Exception as e:
            logger.error("Background task %s failed: %s", task_id, e)
            self._tasks[task_id].update({
                "status":       "error",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error":        str(e),
                "traceback":    traceback.format_exc(),
            })


# Module-level singleton (shared across all API requests)
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
