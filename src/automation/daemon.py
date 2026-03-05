"""
Pipeline Daemon — manages the scheduler and queue reader as a unified long-lived process.

Handles:
- Startup validation (config, queue dir, env vars, hybrid config)
- Signal handlers (SIGTERM, SIGINT) for graceful shutdown
- File-based logging with rotation (logs/daemon.log)
- Health status reporting
"""

import logging
import logging.handlers
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DEFAULTS = {
    "scheduler": {
        "enabled":          True,
        "watchlist_hour":   8,
        "watchlist_minute": 30,
        "timezone":         "US/Eastern",
        "weekdays_only":    True,
        "hybrid_config":    "hybrid_haiku_tools",
        "publish":          True,
        "watchlist":        "default",
    },
    "queue_reader": {
        "enabled":                True,
        "poll_interval_seconds":  30,
        "queue_dir":              "queue",
        "target_trader":          "trifecta-trader",
        "hybrid_config":          "hybrid_haiku_tools",
        "publish":                True,
        "max_concurrent_analyses": 1,
        "cooldown_seconds":       60,
        "max_retries":            2,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class PipelineDaemon:
    """Manages the PipelineScheduler and QueueReader as a unified service.

    Usage:
        daemon = PipelineDaemon("config/automation.yaml")
        daemon.start()   # blocks until SIGTERM/SIGINT
    """

    def __init__(self, config_path: str = "config/automation.yaml"):
        self._config_path  = config_path
        self._cfg          = self._load_config(config_path)
        self._scheduler    = None
        self._queue_reader = None
        self._start_time: Optional[datetime] = None
        self._stop_event   = threading.Event()
        self._scans_completed  = 0
        self._signals_published = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        enable_scheduler: bool = True,
        enable_queue: bool = True,
        run_now: bool = False,
    ) -> None:
        """Start the daemon.

        Args:
            enable_scheduler: Whether to start the watchlist scheduler.
            enable_queue: Whether to start the queue reader.
            run_now: Trigger an immediate watchlist scan then exit.
        """
        self._setup_logging()
        logger.info("=" * 60)
        logger.info("Pipeline Daemon starting  config=%s", self._config_path)

        errors = self.validate(enable_scheduler, enable_queue)
        if errors:
            for err in errors:
                logger.error("Startup validation: %s", err)
            raise RuntimeError(f"Daemon startup aborted: {len(errors)} validation error(s). "
                               f"First: {errors[0]}")

        self._setup_signal_handlers()
        self._start_time = datetime.now(timezone.utc)

        sched_cfg = self._cfg["scheduler"]
        qr_cfg    = self._cfg["queue_reader"]

        # ── Scheduler ─────────────────────────────────────────────────────────
        if enable_scheduler and sched_cfg.get("enabled", True):
            from src.automation.scheduler import PipelineScheduler, create_watchlist_scan_fn

            scan_fn = create_watchlist_scan_fn(
                hybrid_config=sched_cfg["hybrid_config"],
                watchlist=sched_cfg.get("watchlist", "default"),
                publish=sched_cfg.get("publish", True),
            )
            self._scheduler = PipelineScheduler(
                watchlist_scan_fn=scan_fn,
                schedule_hour=sched_cfg.get("watchlist_hour", 8),
                schedule_minute=sched_cfg.get("watchlist_minute", 30),
                timezone=sched_cfg.get("timezone", "US/Eastern"),
                weekdays_only=sched_cfg.get("weekdays_only", True),
            )

            if run_now:
                logger.info("--run-now: triggering immediate watchlist scan.")
                result = self._scheduler.run_now()
                logger.info("--run-now complete: %s", result)
                return

            self._scheduler.start()
            logger.info("Scheduler active. Next run: %s", self._scheduler.next_run_time())
        else:
            if run_now:
                logger.warning("--run-now requested but scheduler is disabled.")
                return

        # ── Queue Reader ───────────────────────────────────────────────────────
        if enable_queue and qr_cfg.get("enabled", True):
            from src.automation.queue_reader import QueueReader, create_analyze_fn

            analyze_fn = create_analyze_fn(
                hybrid_config=qr_cfg["hybrid_config"],
                publish=qr_cfg.get("publish", True),
            )
            self._queue_reader = QueueReader(
                queue_dir=qr_cfg.get("queue_dir", "queue"),
                analyze_fn=analyze_fn,
                target_trader=qr_cfg.get("target_trader", "trifecta-trader"),
                poll_interval=qr_cfg.get("poll_interval_seconds", 30),
                max_retries=qr_cfg.get("max_retries", 2),
                cooldown_seconds=qr_cfg.get("cooldown_seconds", 60),
            )
            logger.info("Queue reader active: dir=%s", qr_cfg.get("queue_dir", "queue"))
            # Blocks main thread — this is intentional
            self._queue_reader.start()
        else:
            # No queue reader — keep main thread alive via event
            logger.info("Queue reader disabled. Running scheduler only.")
            self._stop_event.wait()

        logger.info("Daemon exiting.")

    def stop(self) -> None:
        """Graceful shutdown: stop queue reader, then scheduler."""
        logger.info("Daemon shutting down...")
        if self._queue_reader is not None:
            self._queue_reader.stop()
        if self._scheduler is not None:
            self._scheduler.stop()
        self._stop_event.set()
        uptime = self._uptime_seconds()
        logger.info(
            "Daemon stopped. Uptime: %.0fs", uptime if uptime else 0
        )

    def health_check(self) -> dict:
        """Return current health status dict."""
        sched_status = {
            "enabled":   self._scheduler is not None,
            "next_run":  (
                self._scheduler.next_run_time()
                if self._scheduler and self._scheduler.is_running
                else None
            ),
            "last_run":  getattr(self._scheduler, "_last_run", None),
            "is_running": self._scheduler.is_running if self._scheduler else False,
        }
        qr_status = {
            "enabled":       self._queue_reader is not None,
            "is_running":    self._queue_reader.is_running if self._queue_reader else False,
            "pending_count": self._queue_reader.pending_count if self._queue_reader else 0,
            "last_poll":     self._queue_reader.last_poll if self._queue_reader else None,
        }
        overall = "running" if (sched_status["is_running"] or qr_status["is_running"]) else "stopped"

        return {
            "status":          overall,
            "scheduler":       sched_status,
            "queue_reader":    qr_status,
            "uptime_seconds":  self._uptime_seconds(),
        }

    def validate(
        self,
        check_scheduler: bool = True,
        check_queue: bool = True,
    ) -> list:
        """Run startup validation checks. Returns list of error strings (empty = OK)."""
        errors = []
        sched_cfg = self._cfg.get("scheduler", {})
        qr_cfg    = self._cfg.get("queue_reader", {})

        # Hybrid config must exist
        from src.hybrid_llm import CONFIGS
        for label, cfg_section in [("scheduler", sched_cfg), ("queue_reader", qr_cfg)]:
            hc = cfg_section.get("hybrid_config", "hybrid_haiku_tools")
            if hc not in CONFIGS:
                errors.append(f"{label}.hybrid_config '{hc}' not found in CONFIGS. "
                              f"Available: {list(CONFIGS)[:5]}")

        if check_scheduler and sched_cfg.get("enabled", True):
            wl = sched_cfg.get("watchlist", "default")
            wl_path = Path(f"config/watchlists/{wl}.yaml")
            if not wl_path.exists():
                errors.append(f"Watchlist file not found: {wl_path}")

        if check_queue and qr_cfg.get("enabled", True):
            queue_dir = Path(qr_cfg.get("queue_dir", "queue"))
            if not queue_dir.exists():
                errors.append(
                    f"Queue directory not found: {queue_dir}. "
                    "Create it or update queue_reader.queue_dir in automation.yaml."
                )

        # Supabase credentials check (warn only)
        publish_any = (
            sched_cfg.get("publish", True) or qr_cfg.get("publish", True)
        )
        if publish_any:
            if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_SERVICE_KEY"):
                logger.warning(
                    "publish=true but SUPABASE_URL / SUPABASE_SERVICE_KEY not set. "
                    "Signals will not be written to Supabase. Set these env vars or "
                    "set publish: false in automation.yaml."
                )

        return errors

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> dict:
        """Load automation.yaml, merging with defaults."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            logger.warning("Config not found at %s — using defaults.", path)
            return _CONFIG_DEFAULTS

        with open(cfg_path) as f:
            loaded = yaml.safe_load(f) or {}

        return _deep_merge(_CONFIG_DEFAULTS, loaded)

    def _setup_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT for graceful shutdown."""
        def handler(signum, frame):
            logger.info("Received signal %d — initiating graceful shutdown.", signum)
            self.stop()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _setup_logging(self) -> None:
        """Add a rotating file handler to the root logger."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "daemon.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _uptime_seconds(self) -> Optional[float]:
        if self._start_time is None:
            return None
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
