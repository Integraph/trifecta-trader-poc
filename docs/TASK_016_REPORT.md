# Task 016 Report: Admin API (FastAPI)

**Date:** 2026-03-05  
**Status:** Complete  
**Tests:** 105 new passing / 40 pre-existing failures (unchanged)  
**Total:** 497 passing across the full suite

---

## Summary

Built a complete FastAPI REST Admin API exposing full operational state of the Trifecta Trader pipeline. The API serves both the Platform UI health badge and the upcoming Admin Dashboard (Task 017). It shares the live `PipelineDaemon` instance and `PortfolioDatabase` via dependency injection, and degrades gracefully in standalone mode.

---

## Design Decisions (from Q1 response)

| Q | Choice | Implementation |
|---|--------|----------------|
| Q1: Scheduler state | Option A — `_last_run_detail` dict in `PipelineScheduler` | Added to `_run_scan()`, captures result/tickers/elapsed/decisions/error |
| Q2: Event bus | Option A — injectable `event_callback` | Added to `PipelineScheduler`, `QueueReader`, `AccuracyUpdater` with `_emit()` helper |
| Q3: Log streaming | Option B — `AdminLogHandler(logging.Handler)` | Injects into root logger on startup; 500-entry ring buffer |
| Q4: Backfill | Option B — async TaskManager | Same 202+poll pattern as test run and scheduler trigger |
| Q5: Standalone mode | Option B — graceful degradation | DB endpoints work without daemon; daemon-required → 503 |

---

## Deliverables

### Modified Existing Classes

**`src/automation/scheduler.py`**
- Added `event_callback: Optional[Callable] = None` to `PipelineScheduler.__init__()`
- Added `_last_run_detail: Optional[dict] = None` attribute
- Added `_emit()` helper (2 lines, no-op if callback is None)
- Modified `_run_scan()` to populate `_last_run_detail` and emit `scheduler.run_started` + `scheduler.run_completed` events

**`src/automation/queue_reader.py`**
- Added `event_callback: Optional[Callable] = None` to `QueueReader.__init__()`
- Added `_emit()` helper
- Emits `queue.candidate_picked`, `queue.analysis_completed`, `queue.analysis_failed` from `_process_candidate()`

**`src/accuracy/updater.py`**
- Added `event_callback: Optional[Callable] = None` to `AccuracyUpdater.__init__()`
- Added `_emit()` helper
- Emits `accuracy.update_started` + `accuracy.update_completed` from `run_update()`

### New `src/admin/` Package (12 files)

**`dependencies.py`** — DI layer. `init_dependencies(daemon, db)` wires shared instances. `require_daemon()` raises 503 if daemon is None.

**`task_manager.py`** — `TaskManager` with `ThreadPoolExecutor(max_workers=2)`. Ordered dict of task statuses. `submit()` → `get_status()` → `get_result()` → `list_tasks()`. Module-level singleton via `get_task_manager()`.

**`events.py`** — `EventBus` with asyncio fan-out to WebSocket subscribers. `publish()` is thread-safe via `asyncio.run_coroutine_threadsafe()`. 200-event bounded history. `make_event_callback()` returns a callable for subsystem injection. `/ws/events` WebSocket endpoint.

**`logs.py`** — `AdminLogHandler(logging.Handler)` with 500-entry `deque` buffer. `install_admin_handler()` injects into root logger at startup. `GET /logs/recent` reads buffer (falls back to tailing `logs/daemon.log` in standalone mode). `/ws/logs` WebSocket with level filter.

**`health.py`** — `GET /health` with deterministic color logic (`compute_health_color()` is a pure function). Checks: Ollama reachability (1s timeout), Supabase config/credentials, queue file counts, accuracy outcome counts. 

**`scheduler.py`** — `GET /scheduler/status`, `POST /scheduler/trigger` (202+async), `GET /scheduler/trigger/{task_id}`, `GET /scheduler/history` (aggregated from DB by trade_date).

**`queue.py`** — `GET /queue/status`, `GET /queue/pending` (sorted by priority), `GET /queue/completed`, `POST /queue/enqueue` (Scanner-format JSON), `POST /queue/retry/{filename}`, `DELETE /queue/clear`.

**`accuracy.py`** — `GET /accuracy/summary`, `GET /accuracy/ticker/{ticker}`, `POST /accuracy/update` (synchronous), `POST /accuracy/backfill` (async, TaskManager).

**`config.py`** — `GET/PUT /config/automation`, `GET/PUT /config/supabase`, `GET/PUT /config/watchlists/{name}`, `GET /config/hybrid-configs`. PUT responses classify changes as immediate vs. restart-required.

**`test_run.py`** — `POST /test-run` (202+async, publish defaults to False), `GET /test-run/{task_id}`.

**`analyses.py`** — `GET /analyses/recent` (LEFT JOIN signal_outcomes for outcome_status), `GET /analyses/{id}`, `GET /analyses/stats`.

**`tasks.py`** — `GET /tasks`, `GET /tasks/{task_id}` — shared task status across all async operations.

**`app.py`** — `create_app(daemon, db)` factory. Mounts all 10 routers. CORS allows all origins. Generic exception handler. Startup event installs `AdminLogHandler` and wires asyncio loop into `EventBus`. Standalone `__main__` entry point.

### Modified Files

**`src/run_daemon.py`** — Added `--api` (default: True), `--no-api`, `--api-port` (default: 8420), `--api-host` flags. `_start_api_server()` starts uvicorn in a daemon background thread before `daemon.start()` (so health endpoint is available during init).

**`config/automation.yaml`** — Added `admin_api:` block with `enabled: true`, `port: 8420`, `host: "0.0.0.0"`.

**`pyproject.toml`** — Added `fastapi>=0.109.0`, `uvicorn[standard]>=0.27.0`, `websockets>=12.0`, `httpx>=0.27.0`. Added `asyncio_mode = "auto"` to pytest options (enables pytest-asyncio for all async tests).

---

## Exit Criteria Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `src/admin/` package with all 12 source files | ✅ |
| 2 | `GET /health` returns correct status with green/yellow/red | ✅ |
| 3 | Health checks Ollama (1s timeout) | ✅ |
| 4 | Health checks Supabase config | ✅ |
| 5 | Health reports queue file counts | ✅ |
| 6 | `GET /scheduler/status` returns config, next_run, last_run | ✅ |
| 7 | `POST /scheduler/trigger` returns 202, runs in background | ✅ |
| 8 | Trigger result retrievable via `GET /scheduler/trigger/{task_id}` | ✅ |
| 9 | `GET /scheduler/history` groups by trade_date | ✅ |
| 10 | `GET /queue/status` accurate pending/processing/completed | ✅ |
| 11 | `GET /queue/pending` sorted by priority | ✅ |
| 12 | `POST /queue/enqueue` creates Scanner-format JSON | ✅ |
| 13 | `POST /queue/retry/{filename}` resets retry_count | ✅ |
| 14 | `GET /accuracy/summary` returns AccuracyReporter data | ✅ |
| 15 | `GET /accuracy/ticker/{ticker}` ticker-specific report | ✅ |
| 16 | `POST /accuracy/update` triggers update cycle | ✅ |
| 17 | `GET /config/automation` returns merged config | ✅ |
| 18 | `PUT /config/automation` writes to disk, indicates restart | ✅ |
| 19 | `GET /config/watchlists` lists watchlist files with tickers | ✅ |
| 20 | `GET /config/hybrid-configs` lists all CONFIGS | ✅ |
| 21 | `POST /test-run` returns 202 with task_id | ✅ |
| 22 | Test run result includes decision, quality_score, trade_params | ✅ |
| 23 | Test run defaults to `publish: false` | ✅ |
| 24 | `GET /logs/recent` returns parsed log entries with level filter | ✅ |
| 25 | `WebSocket /ws/logs` streams log entries | ✅ |
| 26 | `GET /analyses/recent` with outcome_status LEFT JOIN | ✅ |
| 27 | `GET /analyses/stats` correct aggregates | ✅ |
| 28 | `WebSocket /ws/events` broadcasts subsystem events | ✅ |
| 29 | `TaskManager` max 2 concurrent workers | ✅ |
| 30 | FastAPI app mounts all routers with CORS | ✅ |
| 31 | `run_daemon.py` starts API in background thread | ✅ |
| 32 | `config/automation.yaml` has `admin_api` block | ✅ |
| 33 | 105 new tests pass | ✅ |
| 34 | 497 total passing (was 392 after Task 015) | ✅ |
| 35 | Zero vendor modifications | ✅ |

---

## Test Results

```
tests/test_admin_health.py       13 passed
tests/test_admin_scheduler.py    10 passed
tests/test_admin_queue.py        13 passed
tests/test_admin_accuracy.py     10 passed
tests/test_admin_config.py       12 passed
tests/test_admin_test_run.py      9 passed
tests/test_admin_logs.py          9 passed
tests/test_admin_analyses.py     12 passed
tests/test_admin_task_manager.py 12 passed
tests/test_admin_events.py        5 passed (+ 6 unit tests)
                                  ─────────
Total new                        105 passed / 0 failed

Full suite: 497 passed, 40 failed (pre-existing), 8 skipped
```

---

## Files Created/Modified

**New:**
- `src/admin/__init__.py`
- `src/admin/dependencies.py`
- `src/admin/task_manager.py`
- `src/admin/events.py`
- `src/admin/logs.py`
- `src/admin/health.py`
- `src/admin/scheduler.py`
- `src/admin/queue.py`
- `src/admin/accuracy.py`
- `src/admin/config.py`
- `src/admin/test_run.py`
- `src/admin/analyses.py`
- `src/admin/tasks.py`
- `src/admin/app.py`
- `tests/test_admin_health.py`
- `tests/test_admin_scheduler.py`
- `tests/test_admin_queue.py`
- `tests/test_admin_accuracy.py`
- `tests/test_admin_config.py`
- `tests/test_admin_test_run.py`
- `tests/test_admin_logs.py`
- `tests/test_admin_analyses.py`
- `tests/test_admin_task_manager.py`
- `tests/test_admin_events.py`
- `docs/TASK_016_REPORT.md`

**Modified:**
- `src/automation/scheduler.py` — `_last_run_detail` + `event_callback`
- `src/automation/queue_reader.py` — `event_callback`
- `src/accuracy/updater.py` — `event_callback`
- `src/run_daemon.py` — `--api` / `--no-api` / `--api-port` flags
- `config/automation.yaml` — `admin_api:` block
- `pyproject.toml` — new dependencies, `asyncio_mode = "auto"`
