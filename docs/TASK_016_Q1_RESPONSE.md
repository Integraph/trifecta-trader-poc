# Task 016 Q1 Response — Design Decisions

## Q1: Scheduler Last-Run State Tracking → Option A (Modify PipelineScheduler)

**Choice: Option A — Add `_last_run_detail` dict to `PipelineScheduler`.**

Option B (DB-only) has the critical flaw you identified: failed runs are invisible. The scheduler should own its own state. Add a `_last_run_detail` dict that captures:

```python
self._last_run_detail = {
    "timestamp": "...",
    "result": "success" | "error",
    "tickers_processed": 8,
    "elapsed_seconds": 142.5,
    "decisions": {"AAPL": "BUY", "MSFT": "HOLD"},
    "error": None  # or error string on failure
}
```

Populate it inside `_run_scan()` after the scan function returns (success or failure). This is a minimal change — one new attribute and ~10 lines in one method. The daemon's `health_check()` already reads `self._scheduler._last_run` and `self._scheduler._last_result`, so this is a natural extension of the existing pattern.

---

## Q2: Event Bus → Option A (Injectable Callback)

**Choice: Option A — Injectable `event_bus` callback in subsystems.**

Real-time events are the whole point of `/ws/events`. Polling-based synthetic events would feel sluggish and unreliable. The modification to existing classes is minimal:

- Add an optional `event_callback: Callable = None` parameter to `PipelineScheduler.__init__()`, `QueueReader.__init__()`, and `AccuracyUpdater.__init__()`
- Add ~3-5 `self._emit("event_type", data)` calls at key lifecycle points in each class
- `_emit` is a 2-line helper: `if self._event_callback: self._event_callback(event_type, data)`

When running without the API (CLI mode), `event_callback=None` means zero overhead. When the API is running, the daemon passes the live event bus. This is clean, testable, and truly real-time.

**Key lifecycle points to emit from:**
- `PipelineScheduler._run_scan()` — start and complete/error
- `QueueReader._process_candidate()` — picked up and complete/error
- `AccuracyUpdater.run_update()` — start and complete

---

## Q3: WebSocket Log Streaming → Option B (Custom logging.Handler)

**Choice: Option B — Custom `logging.Handler` injected into the root logger.**

The file-watching approach (Option A) is fragile — it breaks if the log file is rotated, if the daemon hasn't written to disk yet (buffering), or if someone changes the log path. A custom handler captures every log entry at the source, regardless of file configuration. It's also lower latency.

Implementation: Create an `AdminLogHandler(logging.Handler)` that pushes formatted log records into an `asyncio.Queue` (or a simple deque with a max size). The WebSocket endpoint reads from this queue and broadcasts to clients. The handler is only injected when the API server starts (`--api` flag), so zero impact on non-API usage.

The handler should keep a bounded buffer (last 500 entries) so `GET /logs/recent` can also read from it without touching the filesystem.

---

## Q4: Accuracy Backfill Response → Option B (Async Task Pattern)

**Choice: Option B — Async via TaskManager, same as test run.**

Consistency wins here. The admin dashboard will already have the polling UI pattern built for test runs and scheduler triggers. Backfill uses the exact same flow: POST returns 202 with task_id, client polls `GET /tasks/{task_id}`. One pattern, one UI component, three use cases.

This also means `/accuracy/update` (the daily update, not backfill) can stay synchronous since it's typically fast (a few seconds for 10-20 pending outcomes). But if you find it's slow in practice, you can trivially switch it to the async pattern later since the TaskManager is already there.

---

## Q5: API Standalone Mode → Option B (Graceful Degraded Mode)

**Choice: Option B — Graceful degradation with 503 for daemon-dependent endpoints.**

This is important for both testing and development workflow. The API should work in two modes:

**Full mode** (started via `run_daemon.py --api`): All endpoints operational, daemon instance shared.

**Standalone mode** (started directly for dev/testing): Endpoints that only need the database work fine — `/analyses/recent`, `/analyses/stats`, `/accuracy/summary`, `/accuracy/ticker/{ticker}`, `/config/*`, `/logs/recent`. Endpoints that need a live daemon return `503 Service Unavailable` with a clear message: `{"error": "daemon_not_running", "message": "This endpoint requires a running daemon. Start with: python -m src.run_daemon --api"}`.

This makes the API independently testable with `httpx.AsyncClient` — you can test all DB-backed endpoints without spinning up the full daemon. Add a standalone entry point:

```python
# python -m src.admin.app (standalone dev mode)
if __name__ == "__main__":
    import uvicorn
    app = create_app(daemon=None, db=PortfolioDatabase())
    uvicorn.run(app, host="0.0.0.0", port=8420)
```

---

## Summary

| Question | Choice | Rationale |
|----------|--------|-----------|
| Q1: Scheduler state | **Option A** — Modify PipelineScheduler | Failed runs must be visible; scheduler owns its state |
| Q2: Event bus | **Option A** — Injectable callback | Real-time is the point; minimal 3-5 line changes per class |
| Q3: Log streaming | **Option B** — Custom logging.Handler | More reliable than file-watching; captures all output |
| Q4: Backfill async | **Option B** — Async TaskManager | Consistency with test run and trigger patterns |
| Q5: Standalone mode | **Option B** — Graceful degradation | Essential for testing; DB endpoints work without daemon |
