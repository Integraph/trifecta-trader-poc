# Task 014: Pipeline Automation — Scheduler + Queue Reader + Daemon Mode

**Priority:** HIGH — Turns the POC from a manual tool into an always-on signal generator
**Depends on:** Task 013 (complete — signal adapter + Supabase writer)
**Parallel with:** Market Scanner S2 (building the queue writer side)

---

## Objective

Add three capabilities to the POC that make it run autonomously:

1. **Watchlist Scheduler** — Runs the default watchlist analysis on a daily schedule before market open, auto-publishing signals to Supabase
2. **Queue Reader** — Watches the Scanner's `queue/pending/` directory for new candidate JSON files and runs deep analysis on them
3. **Daemon Mode** — A `--daemon` flag that keeps the process running, handling both scheduled watchlist scans and incoming Scanner candidates

After Task 014, the full autonomous pipeline is: Scanner finds opportunities → drops candidates into queue → POC picks them up → runs multi-agent analysis → publishes signals to Supabase → Platform UI displays them.

---

## Background

Currently, running the pipeline requires manual invocation:

```bash
python -m src.run_analysis --ticker AAPL --hybrid hybrid_haiku_tools --publish
python -m src.run_batch --tickers AAPL,MSFT --hybrid hybrid_haiku_tools --publish
```

The Market Scanner (separate repo) is completing S2, which adds a file-based JSON queue. The Scanner writes candidate files to `queue/pending/`. Our POC needs to read from that queue, analyze the candidates, and write results back to `queue/completed/`.

---

## Deliverable 1: Watchlist Scheduler (`src/automation/scheduler.py`)

### Overview

A scheduler that runs the default watchlist through `run_batch` on a configurable schedule. Uses APScheduler (same library the Scanner uses for consistency).

### Class Design

```python
"""
Pipeline scheduler — runs watchlist analysis on a daily schedule.

Uses APScheduler for job scheduling. Runs during a configurable window
(default: 8:30 AM ET, 60 minutes before market open).
"""

import logging
from datetime import datetime
from typing import Optional, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Schedules watchlist analysis runs."""

    def __init__(self,
                 watchlist_scan_fn: Callable,
                 schedule_hour: int = 8,
                 schedule_minute: int = 30,
                 timezone: str = "US/Eastern",
                 weekdays_only: bool = True):
        """
        Args:
            watchlist_scan_fn: Function to call for watchlist scans.
                               Signature: fn() -> dict (batch results)
            schedule_hour: Hour to run (24h format, in timezone)
            schedule_minute: Minute to run
            timezone: Timezone for scheduling (default: US/Eastern)
            weekdays_only: If True, only run Mon-Fri (default: True)
        """

    def start(self) -> None:
        """Start the scheduler. Non-blocking."""

    def stop(self) -> None:
        """Stop the scheduler gracefully."""

    def run_now(self) -> dict:
        """Trigger an immediate watchlist scan (for testing/manual use)."""

    def next_run_time(self) -> Optional[datetime]:
        """Return the next scheduled run time, or None if not scheduled."""

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is active."""
```

### Configuration

Add to `config/automation.yaml`:

```yaml
scheduler:
  enabled: true
  watchlist_hour: 8          # 8:30 AM ET = 60 min before market open
  watchlist_minute: 30
  timezone: "US/Eastern"
  weekdays_only: true        # Skip Sat/Sun
  hybrid_config: "hybrid_haiku_tools"
  publish: true              # Auto-publish to Supabase
  watchlist: "default"       # config/watchlists/default.yaml

queue_reader:
  enabled: true
  poll_interval_seconds: 30  # How often to check for new candidates
  queue_dir: "../trifecta-market-scanner/queue"   # Path to Scanner's queue
  target_trader: "trifecta-trader"                 # Only pick up stock candidates
  hybrid_config: "hybrid_haiku_tools"
  publish: true
  max_concurrent_analyses: 1  # Sequential for now (LLM rate limits)
  cooldown_seconds: 60        # Min wait between analyses
```

### Watchlist Scan Function

The scheduler calls a function that wraps the existing `run_batch` logic:

```python
def create_watchlist_scan_fn(hybrid_config: str, watchlist: str, publish: bool) -> Callable:
    """
    Create a watchlist scan function for the scheduler.

    Returns a function that:
    1. Loads the watchlist YAML
    2. Runs run_batch logic for each ticker
    3. Publishes to Supabase if publish=True
    4. Returns the batch result dict
    """
```

This function should reuse the existing `run_batch.py` logic — don't duplicate the pipeline invocation code. Import and call the same functions.

### Implementation Notes

- **APScheduler BackgroundScheduler** — runs in a background thread, doesn't block the main thread
- **CronTrigger** — `day_of_week='mon-fri'` for weekdays_only, `hour=8, minute=30, timezone='US/Eastern'`
- If a scan is already running when the next trigger fires, **skip** it (APScheduler's `max_instances=1` handles this)
- Log start time, end time, and result summary for each scan
- All errors caught — a failed scan doesn't crash the scheduler

---

## Deliverable 2: Queue Reader (`src/automation/queue_reader.py`)

### Overview

A polling loop that watches the Scanner's queue directory for new candidate JSON files. When a file appears, the reader picks it up, runs deep analysis, publishes the signal to Supabase, and writes the result back to the queue.

### Queue Message Format (from Scanner S2 spec)

The Scanner writes JSON files to `queue/pending/` with this format:

```json
{
    "scanner_id": "scan_20260304_143000",
    "timestamp": "2026-03-04T14:30:00Z",
    "asset_type": "stock",
    "ticker": "AAPL",
    "opportunity_score": 72.5,
    "catalysts": ["Volume surge detected", "Price breakout confirmed"],
    "signal_scores": {
        "volume_surge": 0.80,
        "price_breakout": 0.70,
        "momentum_shift": 0.35,
        "earnings_catalyst": null
    },
    "key_data": {
        "current_price": 263.75,
        "volume": 45000000,
        "market_cap": 2100000000000,
        "sector": "Technology",
        "name": "Apple Inc."
    },
    "target_trader": "trifecta-trader",
    "priority": "medium",
    "status": "pending"
}
```

File naming convention: `{YYYYMMDD_HHMMSS}_{TICKER}.json`

### Class Design

```python
"""
Queue Reader — polls the Scanner's queue for new candidates and runs
deep analysis on them.

Lifecycle:
1. Poll queue/pending/ for JSON files with target_trader="trifecta-trader"
2. Move file to queue/processing/ (atomic rename)
3. Run deep analysis via run_analysis pipeline
4. Publish signal to Supabase (if publish=True)
5. Write result to queue/completed/ with analysis results appended
6. If analysis fails, move file back to queue/pending/ with retry count
"""

import json
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class QueueReader:
    """Reads Scanner candidates from the file-based queue and runs analysis."""

    def __init__(self,
                 queue_dir: str,
                 analyze_fn: Callable,
                 target_trader: str = "trifecta-trader",
                 poll_interval: int = 30,
                 max_retries: int = 2,
                 cooldown_seconds: int = 60):
        """
        Args:
            queue_dir: Path to the Scanner's queue directory (contains pending/, processing/, completed/)
            analyze_fn: Function to call for analysis.
                        Signature: fn(ticker: str, scanner_context: dict) -> dict
                        Returns the analysis result dict (same as run_analysis output)
            target_trader: Only process candidates with this target_trader value
            poll_interval: Seconds between queue polls
            max_retries: Max times to retry a failed analysis before giving up
            cooldown_seconds: Min seconds between analyses (respects LLM rate limits)
        """

    def poll_once(self) -> list[dict]:
        """
        Single poll cycle:
        1. List JSON files in pending/
        2. Filter by target_trader
        3. Sort by priority (high > medium > low), then by timestamp (oldest first)
        4. Process each candidate sequentially

        Returns list of processed results.
        """

    def start(self) -> None:
        """Start the polling loop. Blocks the calling thread."""

    def stop(self) -> None:
        """Signal the polling loop to stop after current cycle completes."""

    @property
    def is_running(self) -> bool:
        """Whether the reader is actively polling."""

    def _process_candidate(self, pending_path: Path) -> Optional[dict]:
        """
        Process a single candidate:
        1. Read JSON from pending/
        2. Move to processing/ (atomic rename via shutil.move)
        3. Call analyze_fn(ticker, scanner_context)
        4. On success: write enriched JSON to completed/
        5. On failure: move back to pending/ with incremented retry_count

        Returns result dict or None on failure.
        """

    def _should_skip(self, message: dict) -> bool:
        """
        Skip if:
        - target_trader doesn't match
        - retry_count >= max_retries
        - asset_type is "crypto" (this trader handles stocks only)
        """
```

### Analyze Function

The queue reader calls a function that wraps the existing `run_analysis` logic:

```python
def create_analyze_fn(hybrid_config: str, publish: bool) -> Callable:
    """
    Create an analysis function for the queue reader.

    Returns a function that:
    1. Takes (ticker: str, scanner_context: dict) as args
    2. Runs the full multi-agent pipeline via run_analysis logic
    3. Publishes to Supabase if publish=True
    4. Returns the analysis result dict

    The scanner_context (opportunity_score, catalysts, signal_scores) can be
    injected into the portfolio context to give the pipeline extra information
    about why this ticker was flagged. This is informational — it doesn't
    change the pipeline's decision-making.
    """
```

### Queue Lifecycle

```
Scanner writes → queue/pending/20260304_143000_AAPL.json
                        ↓
QueueReader polls, finds file
                        ↓
Move to → queue/processing/20260304_143000_AAPL.json
                        ↓
Run deep analysis (run_analysis pipeline)
                        ↓
On success: write to → queue/completed/20260304_143000_AAPL.json
            (original message + analysis_result appended)
                        ↓
On failure: move back to → queue/pending/20260304_143000_AAPL.json
            (retry_count incremented)
```

### Completed Message Format

When analysis succeeds, the completed JSON contains the original Scanner message plus the analysis results:

```json
{
    "scanner_id": "scan_20260304_143000",
    "timestamp": "2026-03-04T14:30:00Z",
    "asset_type": "stock",
    "ticker": "AAPL",
    "opportunity_score": 72.5,
    "catalysts": ["Volume surge detected", "Price breakout confirmed"],
    "signal_scores": { ... },
    "key_data": { ... },
    "target_trader": "trifecta-trader",
    "priority": "medium",
    "status": "completed",
    "analysis_result": {
        "decision": "BUY",
        "quality_score": 9.4,
        "entry_price": 264.72,
        "stop_loss": 238.0,
        "price_target": 285.0,
        "elapsed_seconds": 1056.0,
        "cost_usd": 0.0626,
        "signal_published": true,
        "signal_id": "a1b2c3d4-..."
    },
    "completed_at": "2026-03-04T15:12:30Z"
}
```

### Implementation Notes

- **Atomic file moves** — use `shutil.move()` to prevent race conditions if the Scanner is writing while we're reading
- **Priority sorting** — process high-priority candidates first: `{"high": 0, "medium": 1, "low": 2}`
- **Cooldown** — after each analysis, wait `cooldown_seconds` before the next one. This prevents overloading the LLM APIs. 60 seconds default.
- **Retry logic** — on failure, move back to pending with `retry_count` incremented in the JSON. After `max_retries`, leave in pending but skip on future polls (log a warning).
- **Crypto filtering** — `target_trader: "trifecta-trader"` means stocks only. Crypto candidates (target: `"trifecta-crypto-trader"`) are ignored by this reader. The Crypto Trader will have its own queue reader eventually.
- **Scanner context injection** — pass the Scanner's `opportunity_score`, `catalysts`, and `signal_scores` into the analysis as informational context. The pipeline can reference this in its reasoning but it doesn't change the decision logic.

---

## Deliverable 3: Daemon Mode (`src/automation/daemon.py`)

### Overview

A `--daemon` flag on the POC that starts both the scheduler and the queue reader, keeps the process running, and handles graceful shutdown.

### Entry Point

Add to `run_analysis.py` (or create `src/run_daemon.py`):

```
python -m src.run_daemon
python -m src.run_daemon --config config/automation.yaml
python -m src.run_daemon --no-scheduler    # Queue reader only
python -m src.run_daemon --no-queue        # Scheduler only
python -m src.run_daemon --run-now         # Immediate watchlist scan then exit
```

### Class Design

```python
"""
Daemon mode — runs the pipeline scheduler and queue reader as a long-lived process.

Handles:
- Signal handling (SIGTERM, SIGINT) for graceful shutdown
- Logging to file with rotation
- Health status reporting
- Startup validation (checks config, queue dir, Supabase connectivity)
"""

import signal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineDaemon:
    """Manages the scheduler and queue reader as a unified service."""

    def __init__(self, config_path: str = "config/automation.yaml"):
        """Load config and initialize scheduler + queue reader."""

    def start(self) -> None:
        """
        Start the daemon:
        1. Validate config (check queue dir exists, Supabase is reachable)
        2. Set up signal handlers (SIGTERM, SIGINT → graceful stop)
        3. Start scheduler (if enabled)
        4. Start queue reader polling loop (if enabled)
        5. Block main thread until stop signal received
        """

    def stop(self) -> None:
        """
        Graceful shutdown:
        1. Stop queue reader (finish current analysis if running)
        2. Stop scheduler
        3. Log shutdown summary (scans completed, signals published)
        """

    def health_check(self) -> dict:
        """
        Return health status:
        {
            "status": "running" | "stopped",
            "scheduler": {"enabled": bool, "next_run": datetime | None, "last_run": datetime | None},
            "queue_reader": {"enabled": bool, "pending_count": int, "last_poll": datetime | None},
            "uptime_seconds": float
        }
        """
```

### Signal Handling

```python
def _setup_signal_handlers(self):
    """Register SIGTERM and SIGINT handlers for graceful shutdown."""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
```

### Logging

Configure file-based logging with rotation when in daemon mode:

```python
import logging.handlers

def _setup_logging(self):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_dir / "daemon.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(handler)
```

### Startup Validation

Before starting, check:

1. `config/automation.yaml` exists and is valid YAML
2. If queue reader is enabled, the queue directory exists and is accessible
3. If publish is enabled, `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` env vars are set
4. The hybrid config specified in config exists in `CONFIGS`
5. If scheduler is enabled, the watchlist YAML exists

Log warnings for missing optional config, errors for missing required config.

---

## New Files

```
src/automation/
    __init__.py
    scheduler.py              # Deliverable 1
    queue_reader.py           # Deliverable 2
    daemon.py                 # Deliverable 3
config/
    automation.yaml           # Scheduler + queue reader configuration
src/run_daemon.py             # CLI entry point for daemon mode
tests/
    test_scheduler.py         # Scheduler unit tests
    test_queue_reader.py      # Queue reader unit tests (with fixture queue dirs)
    test_daemon.py            # Daemon integration tests
```

## Modified Files

```
requirements.txt              # Add apscheduler, pytz (if not already present)
.env.example                  # Document SUPABASE_URL, SUPABASE_SERVICE_KEY
```

**No modifications to:** `run_analysis.py`, `run_batch.py`, or any existing pipeline code. The daemon wraps the existing functions — it doesn't change them.

---

## Exit Criteria

1. `PipelineScheduler` starts and stops without errors
2. Scheduler triggers watchlist scan at configured time
3. `run_now()` triggers an immediate scan and returns results
4. Weekday-only mode skips Saturday and Sunday
5. `QueueReader` polls the queue directory on the configured interval
6. Reader picks up only files with matching `target_trader`
7. Reader moves files through the lifecycle: pending → processing → completed
8. Failed analyses move back to pending with incremented retry_count
9. Reader skips files that exceed max_retries
10. Reader respects cooldown_seconds between analyses
11. Completed JSON includes original Scanner message + analysis results
12. `PipelineDaemon` starts both scheduler and queue reader
13. `--no-scheduler` and `--no-queue` flags work correctly
14. `--run-now` triggers immediate scan then exits
15. SIGTERM/SIGINT cause graceful shutdown (finish current analysis)
16. Daemon logs to `logs/daemon.log` with rotation
17. Startup validation catches missing config/env vars
18. Health check returns correct status
19. All new tests pass
20. All existing tests still pass (224+ from Task 013)
21. Zero vendor modifications

---

## Testing Strategy

### Scheduler Tests (`test_scheduler.py`)
- Mock APScheduler to verify CronTrigger configuration
- Test weekday-only vs all-days scheduling
- Test run_now() calls the scan function
- Test that concurrent runs are prevented (max_instances=1)
- Test start/stop lifecycle

### Queue Reader Tests (`test_queue_reader.py`)
- Create temp queue directory with fixture JSON files
- Test poll_once() picks up matching files
- Test priority sorting (high before medium before low)
- Test file lifecycle (pending → processing → completed)
- Test retry logic (failure → back to pending with retry_count)
- Test max_retries skip behavior
- Test crypto candidates are ignored
- Test cooldown timing
- Test empty queue returns empty list

### Daemon Tests (`test_daemon.py`)
- Test startup validation (missing config, missing queue dir)
- Test --no-scheduler mode starts only queue reader
- Test --no-queue mode starts only scheduler
- Test health_check returns correct structure
- Test signal handling triggers stop

---

## Dependencies

```
apscheduler>=3.10.0
pytz>=2024.1
```

Install: `pip install apscheduler pytz --break-system-packages`

---

## Notes

- **Queue directory location:** The automation config uses a relative path `../trifecta-market-scanner/queue` by default. This assumes the Scanner repo is cloned alongside the POC repo. The path is configurable via `automation.yaml`.
- **Scanner S2 timing:** The Scanner agent is building S2 now (queue writer + concurrent fetching + scheduler). Task 014 builds the reader side. When both ship, the full automated pipeline is operational. If the Scanner isn't ready yet, the daemon still works — the scheduler runs watchlist scans on schedule, and the queue reader polls an empty directory with no errors.
- **Rate limits:** The cooldown_seconds (60s default) prevents overloading Claude/Ollama APIs. With hybrid_haiku_tools, each analysis takes ~60-120 seconds anyway, so the cooldown is effectively a minimum gap between analyses.
- **No watchlist conflicts:** The scheduler and queue reader may try to analyze the same ticker. This is fine — the Supabase signal adapter uses upsert with deduplication, so the second analysis simply replaces the first if it runs on the same day.
- **Hardware:** Running on MacBook Pro M3 Max 128GB. Ollama qwen2.5:14b runs locally. The daemon should work fine as a background process. When the Mac Mini M4 Pro 64GB arrives, the daemon could move there as a dedicated signal server.
