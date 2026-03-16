# Task 016: Admin API (FastAPI)

**Priority:** HIGH — Operational visibility is essential before scaling pipeline usage
**Depends on:** Task 014 (daemon/scheduler/queue), Task 015 (accuracy tracker)
**Enables:** Task 017 (Admin Frontend), Platform UI health badge

---

## Objective

Build a FastAPI REST API that exposes the full operational state of the Trifecta Trader pipeline. This API serves two audiences:

1. **Admin Dashboard (Task 017):** Full operational control — daemon status, scheduler management, queue inspection, accuracy reports, configuration, test runs, and log streaming.
2. **Platform UI health badge:** A single lightweight `/health` endpoint that returns green/yellow/red status so traders immediately know whether the pipeline is functioning.

The API runs as a companion process alongside the daemon, served on `localhost:8420`. No authentication is required (local-only access).

---

## Architecture

```
┌────────────────────────────────────┐
│  Admin API  (FastAPI :8420)        │
│                                    │
│  /health          → Overall status │
│  /scheduler/*     → Scheduler ops  │
│  /queue/*         → Queue ops      │
│  /accuracy/*      → Accuracy data  │
│  /config/*        → Config R/W     │
│  /test-run        → One-off test   │
│  /logs/stream     → WebSocket logs │
│  /analyses/*      → History query  │
│  /ws/events       → Live events    │
├────────────────────────────────────┤
│  PipelineDaemon (shared instance)  │
│  PortfolioDatabase (shared conn)   │
│  YAML config (read / write-back)   │
└────────────────────────────────────┘
```

The key design principle: the Admin API **shares** the same `PipelineDaemon` instance that's already running. It does NOT start a second daemon. The entry point (`src/run_daemon.py`) is modified to optionally start the FastAPI server in a background thread alongside the existing daemon loop.

---

## Deliverable 1: Health Endpoint (`src/admin/health.py`)

This is the single most important endpoint — it powers the Platform UI health badge.

### `GET /health`

Returns the overall pipeline health status plus per-subsystem breakdown.

```json
{
  "status": "healthy",
  "color": "green",
  "timestamp": "2026-03-05T14:30:00Z",
  "uptime_seconds": 21600,
  "subsystems": {
    "daemon": {
      "status": "running",
      "pid": 12345,
      "start_time": "2026-03-05T08:30:00Z"
    },
    "scheduler": {
      "status": "running",
      "last_run": "2026-03-05T08:30:00Z",
      "last_run_result": "success",
      "last_run_tickers": 8,
      "last_run_elapsed_seconds": 142.5,
      "next_run": "2026-03-06T08:30:00Z"
    },
    "queue_reader": {
      "status": "running",
      "pending_count": 2,
      "processing_count": 0,
      "completed_today": 5,
      "last_poll": "2026-03-05T14:29:30Z"
    },
    "accuracy_updater": {
      "status": "idle",
      "last_run": "2026-03-04T17:00:00Z",
      "next_run": "2026-03-05T17:00:00Z",
      "pending_outcomes": 12,
      "complete_outcomes": 45
    },
    "supabase": {
      "configured": true,
      "write_enabled": true,
      "last_write": "2026-03-05T08:35:12Z"
    },
    "ollama": {
      "reachable": true,
      "model": "qwen2.5:14b"
    }
  }
}
```

### Health Color Logic

| Color    | Condition |
|----------|-----------|
| `green`  | Daemon running AND scheduler last run succeeded AND queue reader polling |
| `yellow` | Daemon running BUT one of: scheduler last run failed, OR accuracy updater errored, OR Supabase write_enabled=false, OR Ollama unreachable |
| `red`    | Daemon not running OR scheduler is disabled while enabled in config OR queue reader stopped unexpectedly |

The color logic must be deterministic and tested. The Platform UI will only read `status` and `color` from this endpoint.

### Implementation Notes

- Import `os.getpid()` for the PID
- Check Ollama reachability with a quick HTTP GET to `http://localhost:11434/api/tags` (1-second timeout, catch all exceptions → `reachable: false`)
- Check Supabase config from `config/supabase.yaml`
- Queue counts: use `Path("queue/pending").glob("*.json")` and `Path("queue/completed").glob("*.json")` with a today-only filter on completed
- `pending_outcomes` and `complete_outcomes`: query `signal_outcomes` table

---

## Deliverable 2: Scheduler Endpoints (`src/admin/scheduler.py`)

### `GET /scheduler/status`

Returns current scheduler state.

```json
{
  "enabled": true,
  "is_running": true,
  "schedule": "08:30 US/Eastern weekdays",
  "next_run": "2026-03-06T08:30:00-05:00",
  "last_run": {
    "timestamp": "2026-03-05T08:30:00-05:00",
    "result": "success",
    "tickers_processed": 8,
    "elapsed_seconds": 142.5,
    "decisions": {"AAPL": "BUY", "MSFT": "HOLD", "TSLA": "SELL"}
  },
  "config": {
    "watchlist_hour": 8,
    "watchlist_minute": 30,
    "timezone": "US/Eastern",
    "weekdays_only": true,
    "hybrid_config": "hybrid_haiku_tools",
    "publish": true,
    "watchlist": "default"
  }
}
```

### `POST /scheduler/trigger`

Trigger an immediate watchlist scan. Returns the scan result when complete.

**Request body (optional):**
```json
{
  "watchlist": "default",
  "publish": false
}
```

**Important:** This calls `PipelineScheduler.run_now()` which is blocking. The endpoint should run the scan in a background thread and return a `202 Accepted` with a task ID. The client polls `GET /scheduler/trigger/{task_id}` for the result.

```json
{
  "task_id": "scan_20260305_143500",
  "status": "running",
  "started_at": "2026-03-05T14:35:00Z"
}
```

### `GET /scheduler/trigger/{task_id}`

Poll for the result of a triggered scan.

```json
{
  "task_id": "scan_20260305_143500",
  "status": "complete",
  "started_at": "2026-03-05T14:35:00Z",
  "completed_at": "2026-03-05T14:37:22Z",
  "result": {
    "tickers_processed": 8,
    "elapsed_seconds": 142.5,
    "decisions": {"AAPL": "BUY", "MSFT": "HOLD"}
  }
}
```

### `GET /scheduler/history`

**Query params:** `?days=7&limit=50`

Returns recent scheduler run history from the `analyses` table, grouped by `trade_date`.

```json
{
  "runs": [
    {
      "trade_date": "2026-03-05",
      "tickers_processed": 8,
      "decisions": {"BUY": 3, "SELL": 1, "HOLD": 4},
      "avg_quality_score": 7.2,
      "total_elapsed_seconds": 142.5,
      "total_cost_usd": 0.23
    }
  ]
}
```

---

## Deliverable 3: Queue Endpoints (`src/admin/queue.py`)

### `GET /queue/status`

Returns queue reader state and file counts.

```json
{
  "enabled": true,
  "is_running": true,
  "poll_interval_seconds": 30,
  "last_poll": "2026-03-05T14:29:30Z",
  "counts": {
    "pending": 2,
    "processing": 0,
    "completed": 15,
    "failed": 1
  },
  "config": {
    "queue_dir": "queue",
    "target_trader": "trifecta-trader",
    "max_retries": 2,
    "cooldown_seconds": 60
  }
}
```

### `GET /queue/pending`

Returns the contents of all pending queue files, sorted by priority.

```json
{
  "candidates": [
    {
      "filename": "scanner_20260305_143000_NVDA.json",
      "ticker": "NVDA",
      "priority": "high",
      "opportunity_score": 0.85,
      "catalysts": ["volume_surge", "price_breakout"],
      "asset_type": "stock",
      "retry_count": 0,
      "queued_at": "2026-03-05T14:30:00Z"
    }
  ]
}
```

### `GET /queue/completed`

**Query params:** `?days=1&limit=20`

Returns completed queue files with their analysis results.

### `POST /queue/enqueue`

Manually add a ticker to the queue for analysis.

**Request body:**
```json
{
  "ticker": "NVDA",
  "priority": "high",
  "reason": "Manual admin request"
}
```

Creates a JSON file in `queue/pending/` with the standard Scanner message format (using sensible defaults for fields the Scanner would normally populate).

### `POST /queue/retry/{filename}`

Move a failed/completed file back to `queue/pending/` with `retry_count` reset to 0.

### `DELETE /queue/clear`

**Query param:** `?target=completed` (one of: `pending`, `completed`, `all`)

Remove files from the specified queue subdirectory. Returns count of files removed.

---

## Deliverable 4: Accuracy Endpoints (`src/admin/accuracy.py`)

### `GET /accuracy/summary`

**Query params:** `?days=30`

Returns the same data as `AccuracyReporter.summary()` but as JSON.

```json
{
  "period_days": 30,
  "total_signals": 45,
  "pending_outcomes": 12,
  "by_decision": {
    "BUY": {
      "count": 20,
      "direction_correct_t5": 0.65,
      "avg_return_t5_pct": 2.3,
      "target_hit_rate": 0.45,
      "stop_hit_rate": 0.25
    }
  },
  "by_quality_tier": {
    "high (8-10)": {
      "count": 12,
      "direction_correct_t5": 0.83,
      "avg_return_t5_pct": 4.1
    }
  },
  "best_signals": [],
  "worst_signals": []
}
```

### `GET /accuracy/ticker/{ticker}`

Returns `AccuracyReporter.ticker_report()` as JSON.

### `POST /accuracy/update`

Trigger an immediate accuracy update cycle (price fetch + scoring).

**Request body (optional):**
```json
{
  "ticker": "AAPL"
}
```

Returns the `run_update()` summary dict.

### `POST /accuracy/backfill`

**Request body:**
```json
{
  "days_back": 30
}
```

Trigger a backfill. Returns the backfill summary.

---

## Deliverable 5: Configuration Endpoints (`src/admin/config.py`)

### `GET /config/automation`

Returns the current `config/automation.yaml` as JSON (merged with defaults).

### `PUT /config/automation`

Write updated configuration back to `config/automation.yaml`.

**Request body:** Full or partial config dict. Merges with existing config (deep merge), writes to disk, and reloads the daemon config.

**Important:** Some config changes require a daemon restart to take effect (e.g., changing `watchlist_hour`). The response should indicate which changes take effect immediately vs. require restart.

```json
{
  "applied": {
    "queue_reader.poll_interval_seconds": 15
  },
  "requires_restart": [
    "scheduler.watchlist_hour"
  ]
}
```

### `GET /config/supabase`

Returns `config/supabase.yaml` as JSON.

### `PUT /config/supabase`

Update Supabase config. Same pattern as automation config.

### `GET /config/watchlists`

List available watchlist files and their tickers.

```json
{
  "watchlists": [
    {
      "name": "default",
      "path": "config/watchlists/default.yaml",
      "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"]
    },
    {
      "name": "small_cap",
      "path": "config/watchlists/small_cap.yaml",
      "tickers": ["PLTR", "SOFI", "RKLB"]
    }
  ]
}
```

### `PUT /config/watchlists/{name}`

Update a watchlist file.

**Request body:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"]
}
```

### `GET /config/hybrid-configs`

List all available hybrid LLM configurations from `CONFIGS` in `hybrid_llm.py`.

```json
{
  "configs": [
    {
      "name": "hybrid_haiku_tools",
      "tool_provider": "anthropic",
      "tool_model": "claude-haiku-4-5-20251001",
      "reasoning_quick_provider": "ollama",
      "reasoning_quick_model": "qwen2.5:14b",
      "reasoning_deep_provider": "anthropic",
      "reasoning_deep_model": "claude-sonnet-4-5-20250929"
    }
  ],
  "active": "hybrid_haiku_tools"
}
```

---

## Deliverable 6: Test Run Endpoint (`src/admin/test_run.py`)

This is the admin's "does it still work?" button. Runs a single-ticker analysis and returns the full result.

### `POST /test-run`

**Request body:**
```json
{
  "ticker": "AAPL",
  "hybrid_config": "hybrid_haiku_tools",
  "publish": false,
  "trade_date": "2026-03-05"
}
```

- `hybrid_config` defaults to the scheduler's configured value
- `publish` defaults to `false` (safety: test runs should not publish by default)
- `trade_date` defaults to today

**Behavior:** This is a long-running operation (30-120 seconds). Use the same async task pattern as `/scheduler/trigger`:

1. `POST /test-run` returns `202 Accepted` with a `task_id`
2. `GET /test-run/{task_id}` polls for the result

**Response when complete:**
```json
{
  "task_id": "test_AAPL_20260305_143500",
  "status": "complete",
  "elapsed_seconds": 45.2,
  "result": {
    "ticker": "AAPL",
    "decision": "BUY",
    "quality_score": {
      "composite": 8.2,
      "research": 8.5,
      "analysis": 7.9,
      "recommendation": 8.3
    },
    "trade_params": {
      "entry_price": 178.50,
      "stop_loss": 172.30,
      "price_target": 195.00,
      "position_pct": 3.5,
      "risk_reward_ratio": 2.66,
      "confidence": "high"
    },
    "cost_breakdown": {
      "total_usd": 0.028,
      "by_provider": {}
    },
    "published": false,
    "agents_summary": "Research → Analysis → Trade Decision"
  }
}
```

The test run calls `run_analysis()` directly with the provided parameters. It captures and returns the full result dict including quality scores, trade params, cost breakdown, and decision reasoning.

---

## Deliverable 7: Log Streaming (`src/admin/logs.py`)

### `GET /logs/recent`

**Query params:** `?lines=100&level=INFO`

Returns recent log entries from `logs/daemon.log`.

```json
{
  "lines": [
    {
      "timestamp": "2026-03-05T14:30:00",
      "level": "INFO",
      "logger": "src.automation.scheduler",
      "message": "Watchlist scan starting: 8 tickers  config=hybrid_haiku_tools  date=2026-03-05"
    }
  ]
}
```

### `WebSocket /ws/logs`

Real-time log streaming via WebSocket. Tails `logs/daemon.log` and pushes new entries to connected clients.

**Query params:** `?level=INFO` (minimum level filter)

The WebSocket handler uses a file watcher on `logs/daemon.log` (or a custom `logging.Handler` injected into the root logger) to push entries in real-time.

---

## Deliverable 8: Analysis History (`src/admin/analyses.py`)

### `GET /analyses/recent`

**Query params:** `?days=7&ticker=AAPL&limit=50`

Returns recent analyses from the database with pagination.

```json
{
  "analyses": [
    {
      "id": 42,
      "ticker": "AAPL",
      "trade_date": "2026-03-05",
      "decision": "BUY",
      "quality_score": 8.2,
      "entry_price": 178.50,
      "stop_loss": 172.30,
      "price_target": 195.00,
      "cost_usd": 0.028,
      "elapsed_seconds": 45.2,
      "config": "hybrid_haiku_tools",
      "outcome_status": "pending"
    }
  ],
  "total": 156
}
```

The `outcome_status` field is a LEFT JOIN to `signal_outcomes` to show whether accuracy tracking has been completed.

### `GET /analyses/{id}`

Full detail for a single analysis, including its signal outcome if available.

### `GET /analyses/stats`

Aggregate stats across all analyses.

```json
{
  "total_analyses": 156,
  "total_cost_usd": 4.23,
  "avg_quality_score": 7.1,
  "decision_breakdown": {"BUY": 45, "SELL": 22, "HOLD": 89},
  "avg_elapsed_seconds": 38.5,
  "analyses_today": 8,
  "unique_tickers": 24
}
```

---

## Deliverable 9: Live Events WebSocket (`src/admin/events.py`)

### `WebSocket /ws/events`

Pushes real-time events to the admin dashboard for live updates without polling.

Event types:
```json
{"event": "scheduler.run_started", "data": {"tickers": 8, "timestamp": "..."}}
{"event": "scheduler.run_completed", "data": {"tickers_processed": 8, "elapsed": 142.5}}
{"event": "queue.candidate_picked", "data": {"ticker": "NVDA", "priority": "high"}}
{"event": "queue.analysis_completed", "data": {"ticker": "NVDA", "decision": "BUY"}}
{"event": "accuracy.update_started", "data": {"pending": 12}}
{"event": "accuracy.update_completed", "data": {"updated": 8, "newly_complete": 3}}
{"event": "test_run.started", "data": {"ticker": "AAPL", "task_id": "..."}}
{"event": "test_run.completed", "data": {"ticker": "AAPL", "decision": "BUY"}}
{"event": "health.status_changed", "data": {"from": "green", "to": "yellow", "reason": "..."}}
```

Implementation: Use a simple in-process event bus (asyncio Queue or similar). Each subsystem publishes events; the WebSocket handler broadcasts to all connected clients.

---

## Deliverable 10: App Server & Daemon Integration (`src/admin/app.py`, `src/run_daemon.py` modified)

### `src/admin/app.py`

Creates the FastAPI application, mounts all routers, and configures CORS (allow `localhost:*` origins for the admin frontend).

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Trifecta Trader Admin API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Mount routers
app.include_router(health_router, tags=["Health"])
app.include_router(scheduler_router, prefix="/scheduler", tags=["Scheduler"])
app.include_router(queue_router, prefix="/queue", tags=["Queue"])
app.include_router(accuracy_router, prefix="/accuracy", tags=["Accuracy"])
app.include_router(config_router, prefix="/config", tags=["Configuration"])
app.include_router(test_run_router, prefix="/test-run", tags=["Test Run"])
app.include_router(logs_router, prefix="/logs", tags=["Logs"])
app.include_router(analyses_router, prefix="/analyses", tags=["Analyses"])
```

### `src/run_daemon.py` modification

Add a `--api` flag (default: `true`) and an `--api-port` flag (default: `8420`).

When `--api` is enabled, start the FastAPI server in a background thread using `uvicorn.run()` with the shared `PipelineDaemon` instance injected into the app state.

```python
if args.api:
    import threading
    import uvicorn
    from src.admin.app import create_app

    admin_app = create_app(daemon=daemon, db=db)
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(admin_app,),
        kwargs={"host": "0.0.0.0", "port": args.api_port, "log_level": "warning"},
        daemon=True,
    )
    api_thread.start()
    logger.info("Admin API started on port %d", args.api_port)
```

### `config/automation.yaml` addition

```yaml
admin_api:
  enabled: true
  port: 8420
  host: "0.0.0.0"
```

---

## Deliverable 11: Async Task Manager (`src/admin/task_manager.py`)

Several endpoints (test run, scheduler trigger, accuracy backfill) are long-running. A shared task manager handles these consistently.

```python
class TaskManager:
    """Manages background tasks for the Admin API."""

    def submit(self, task_id: str, fn: Callable, *args) -> dict:
        """Submit a background task. Returns status dict."""

    def get_status(self, task_id: str) -> dict:
        """Get task status: pending/running/complete/error."""

    def get_result(self, task_id: str) -> Optional[dict]:
        """Get completed task result."""

    def list_tasks(self, limit: int = 20) -> list:
        """List recent tasks with status."""
```

Implementation: `concurrent.futures.ThreadPoolExecutor` with a max of 2 workers (one test run + one admin operation at a time). Tasks are stored in a dict keyed by `task_id` with their status and result.

### `GET /tasks`

List recent background tasks (test runs, triggered scans, backfills).

### `GET /tasks/{task_id}`

Poll a specific task for its result.

---

## Dependencies

Add to `requirements.txt` (or `pyproject.toml`):

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0
```

---

## File Structure

```
src/admin/
├── __init__.py
├── app.py              # FastAPI app factory, CORS, router mounting
├── health.py           # GET /health
├── scheduler.py        # /scheduler/* endpoints
├── queue.py            # /queue/* endpoints
├── accuracy.py         # /accuracy/* endpoints
├── config.py           # /config/* endpoints
├── test_run.py         # /test-run endpoints
├── logs.py             # /logs/* + WebSocket /ws/logs
├── analyses.py         # /analyses/* endpoints
├── events.py           # WebSocket /ws/events, event bus
├── task_manager.py     # Async task execution
└── dependencies.py     # FastAPI dependency injection (daemon, db instances)

tests/
├── test_admin_health.py
├── test_admin_scheduler.py
├── test_admin_queue.py
├── test_admin_accuracy.py
├── test_admin_config.py
├── test_admin_test_run.py
├── test_admin_logs.py
├── test_admin_analyses.py
├── test_admin_task_manager.py
└── test_admin_events.py
```

---

## Testing Strategy

### Health Tests (`test_admin_health.py`)
- Green status when all subsystems running
- Yellow status when scheduler last run failed
- Yellow when Ollama unreachable
- Red when daemon not running
- Health color logic is deterministic (same input → same color)
- Supabase status reflects config/supabase.yaml
- Queue counts match actual files on disk

### Scheduler Tests (`test_admin_scheduler.py`)
- GET /scheduler/status returns correct config and next_run
- POST /scheduler/trigger returns 202 with task_id
- GET /scheduler/trigger/{task_id} returns result when complete
- History endpoint groups by trade_date correctly

### Queue Tests (`test_admin_queue.py`)
- GET /queue/status returns correct counts
- GET /queue/pending returns sorted by priority
- POST /queue/enqueue creates valid JSON file in pending/
- POST /queue/retry moves file back to pending with reset retry_count
- DELETE /queue/clear removes only specified target

### Accuracy Tests (`test_admin_accuracy.py`)
- GET /accuracy/summary returns AccuracyReporter data
- GET /accuracy/ticker/{ticker} returns ticker-specific data
- POST /accuracy/update triggers update and returns summary
- POST /accuracy/backfill triggers backfill with correct days_back

### Config Tests (`test_admin_config.py`)
- GET /config/automation returns merged config
- PUT /config/automation writes to disk correctly
- PUT response indicates which fields require restart
- GET /config/watchlists lists all .yaml files with tickers
- PUT /config/watchlists/{name} creates/updates watchlist file
- GET /config/hybrid-configs lists all CONFIGS entries

### Test Run Tests (`test_admin_test_run.py`)
- POST /test-run returns 202 with task_id
- Task result includes decision, quality_score, trade_params
- Default publish=false (never accidentally publishes)
- Invalid ticker returns appropriate error

### Logs Tests (`test_admin_logs.py`)
- GET /logs/recent returns parsed log entries
- Level filter works correctly
- Log entries are sorted newest-first

### Analysis History Tests (`test_admin_analyses.py`)
- GET /analyses/recent returns correct fields
- Ticker filter and date range work
- GET /analyses/{id} includes outcome data via LEFT JOIN
- GET /analyses/stats aggregates correctly

### Task Manager Tests (`test_admin_task_manager.py`)
- Submit creates a task in running state
- Get status transitions: running → complete
- Error tasks have error message and traceback
- Max 2 concurrent workers enforced
- List tasks returns newest first

### Event Bus Tests (`test_admin_events.py`)
- Events are broadcast to all connected WebSocket clients
- Event types match documented schema
- Disconnected clients don't cause errors

Use `httpx.AsyncClient` with `app` for testing FastAPI endpoints directly (no actual server needed). Use `unittest.mock` for daemon/db dependencies.

---

## Implementation Notes

### Dependency Injection

Use FastAPI's dependency injection to provide shared instances:

```python
# src/admin/dependencies.py
from src.automation.daemon import PipelineDaemon
from src.portfolio.database import PortfolioDatabase

_daemon: PipelineDaemon = None
_db: PortfolioDatabase = None

def get_daemon() -> PipelineDaemon:
    return _daemon

def get_db() -> PortfolioDatabase:
    return _db

def init_dependencies(daemon: PipelineDaemon, db: PortfolioDatabase):
    global _daemon, _db
    _daemon = daemon
    _db = db
```

### Config Write-Back

When writing config changes via PUT:
1. Read existing YAML from disk
2. Deep-merge the incoming changes
3. Write back to disk with `yaml.safe_dump()`
4. Update the in-memory config on the daemon instance
5. Return which changes take effect immediately vs. require restart

Changes that take effect immediately:
- `queue_reader.poll_interval_seconds`
- `queue_reader.max_retries`
- `queue_reader.cooldown_seconds`
- `accuracy.backfill_on_first_run`
- `supabase.write_enabled`
- `supabase.signal_ttl_hours`

Changes that require daemon restart:
- `scheduler.watchlist_hour` / `watchlist_minute` / `timezone`
- `scheduler.hybrid_config`
- `queue_reader.queue_dir`
- `queue_reader.target_trader`
- `accuracy.update_hour` / `update_minute`

### Queue Enqueue Format

When manually enqueueing via `POST /queue/enqueue`, create a file that matches the Scanner's message format:

```json
{
  "scanner_id": "admin_manual",
  "timestamp": "2026-03-05T14:35:00Z",
  "asset_type": "stock",
  "ticker": "NVDA",
  "opportunity_score": 0.0,
  "catalysts": [],
  "signal_scores": {},
  "key_data": {},
  "target_trader": "trifecta-trader",
  "priority": "high",
  "status": "pending",
  "source": "admin_api",
  "reason": "Manual admin request"
}
```

Filename format: `admin_{ticker}_{timestamp}.json`

### Error Handling

All endpoints should return consistent error responses:

```json
{
  "error": "scheduler_not_running",
  "message": "Cannot trigger scan: scheduler is not running",
  "status_code": 409
}
```

Use FastAPI exception handlers for consistency.

---

## Exit Criteria

1. `src/admin/` package with all 12 source files created
2. `GET /health` returns correct status with green/yellow/red color logic
3. Health endpoint checks Ollama reachability (1s timeout)
4. Health endpoint checks Supabase config status
5. Health endpoint reports queue file counts accurately
6. `GET /scheduler/status` returns config, next_run, last_run details
7. `POST /scheduler/trigger` returns 202 and runs scan in background
8. Trigger task result is retrievable via `GET /scheduler/trigger/{task_id}`
9. `GET /scheduler/history` groups analyses by trade_date
10. `GET /queue/status` returns accurate pending/processing/completed counts
11. `GET /queue/pending` returns candidates sorted by priority
12. `POST /queue/enqueue` creates valid Scanner-format JSON in queue/pending/
13. `POST /queue/retry/{filename}` resets retry_count and moves to pending
14. `GET /accuracy/summary` returns AccuracyReporter data as JSON
15. `GET /accuracy/ticker/{ticker}` returns ticker-specific report
16. `POST /accuracy/update` triggers update cycle and returns summary
17. `GET /config/automation` returns merged config as JSON
18. `PUT /config/automation` writes to disk and indicates restart requirements
19. `GET /config/watchlists` lists all watchlist files with tickers
20. `GET /config/hybrid-configs` lists all CONFIGS from hybrid_llm.py
21. `POST /test-run` accepts ticker + config, returns 202 with task_id
22. Test run result includes decision, quality_score, trade_params, cost
23. Test run defaults to `publish: false`
24. `GET /logs/recent` returns parsed log entries with level filtering
25. `WebSocket /ws/logs` streams log entries in real-time
26. `GET /analyses/recent` returns analyses with outcome_status join
27. `GET /analyses/stats` returns correct aggregate statistics
28. `WebSocket /ws/events` broadcasts subsystem events to clients
29. `TaskManager` handles concurrent background tasks (max 2 workers)
30. FastAPI app mounts all routers with correct prefixes and CORS
31. `src/run_daemon.py` starts API server in background thread with `--api` flag
32. `config/automation.yaml` has `admin_api` block
33. All new tests pass (target: 80+ tests)
34. All existing tests still pass (392+ from Task 015)
35. Zero vendor modifications (FastAPI is additive, not modifying existing classes)
