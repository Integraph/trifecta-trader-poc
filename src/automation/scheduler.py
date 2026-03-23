"""
Pipeline Scheduler — runs the default watchlist through run_batch on a daily cron schedule.

Uses APScheduler BackgroundScheduler so the main thread remains free for other work
(e.g., the queue reader polling loop).  The scheduler is purely additive: it calls the
same run_batch logic used by the CLI, so no pipeline code is duplicated.
"""

import logging
from datetime import datetime
from typing import Callable, Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def create_watchlist_scan_fn(
    hybrid_config: str,
    watchlist: str = "default",
    publish: bool = True,
    trade_date: Optional[str] = None,
) -> Callable:
    """Return a zero-argument callable that runs run_batch for the watchlist.

    Args:
        hybrid_config: Name of the hybrid LLM config (e.g., 'hybrid_haiku_tools').
        watchlist: Watchlist name (maps to config/watchlists/<name>.yaml).
        publish: Publish signals to Supabase after each ticker.
        trade_date: Override trade date (defaults to today when called).

    Returns:
        A function  () -> dict  that runs the watchlist and returns batch results.
    """
    def _scan_fn() -> dict:
        from src.run_batch import run_batch, load_watchlist  # lazy: avoids pipeline import at module level
        date = trade_date or datetime.now(pytz.utc).strftime("%Y-%m-%d")
        watchlist_path = f"config/watchlists/{watchlist}.yaml"

        try:
            watchlist_display_name, tickers = load_watchlist(watchlist_path)
        except Exception as e:
            logger.error("Failed to load watchlist %s: %s", watchlist_path, e)
            return {"error": str(e), "tickers_processed": 0}

        logger.info(
            "Watchlist scan starting: %d tickers  config=%s  date=%s  name=%s",
            len(tickers), hybrid_config, date, watchlist_display_name,
        )
        start = datetime.now()

        try:
            results = run_batch(
                tickers=tickers,
                watchlist_name=watchlist_display_name,
                hybrid=hybrid_config,
                trade_date=date,
                execute=False,
                dry_run=False,
                publish=publish,
                use_cache=True,
                cost_breakdown=False,
            )
        except Exception as e:
            logger.error("Watchlist scan failed: %s", e)
            return {"error": str(e), "tickers_processed": 0}

        elapsed = (datetime.now() - start).total_seconds()
        decisions = {r.get("ticker", "?"): r.get("decision", "?") for r in results}
        logger.info(
            "Watchlist scan complete: %d/%d tickers in %.0fs — %s",
            len(results), len(tickers), elapsed, decisions,
        )
        return {"results": results, "tickers_processed": len(results), "elapsed_seconds": elapsed}

    return _scan_fn


class PipelineScheduler:
    """Schedules watchlist analysis runs using APScheduler.

    The scheduler runs in a background thread — start() is non-blocking.
    The main thread remains free for the queue reader polling loop.
    """

    def __init__(
        self,
        watchlist_scan_fn: Callable,
        schedule_hour: int = 8,
        schedule_minute: int = 30,
        timezone: str = "US/Eastern",
        weekdays_only: bool = True,
        event_callback: Optional[Callable] = None,
    ):
        """
        Args:
            watchlist_scan_fn: Zero-argument function to call for watchlist scans.
            schedule_hour: Hour to run (24h, in timezone).
            schedule_minute: Minute to run.
            timezone: Timezone for scheduling (default: US/Eastern).
            weekdays_only: If True, only run Mon-Fri.
            event_callback: Optional fn(event_type: str, data: dict) called at
                lifecycle points. Pass the admin event bus when the API is running.
        """
        self._scan_fn        = watchlist_scan_fn
        self._hour           = schedule_hour
        self._minute         = schedule_minute
        self._timezone       = timezone
        self._weekdays       = weekdays_only
        self._event_callback = event_callback
        self._scheduler  = BackgroundScheduler(timezone=pytz.timezone(timezone))
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[dict]  = None
        self._last_run_detail: Optional[dict] = None

    def start(self) -> None:
        """Start the background scheduler. Non-blocking."""
        day_of_week = "mon-fri" if self._weekdays else "*"
        trigger = CronTrigger(
            hour=self._hour,
            minute=self._minute,
            day_of_week=day_of_week,
            timezone=self._timezone,
        )
        self._scheduler.add_job(
            self._run_scan,
            trigger=trigger,
            max_instances=1,           # skip if previous scan still running
            coalesce=True,
            id="watchlist_scan",
            name="Daily watchlist scan",
        )
        self._scheduler.start()
        logger.info(
            "Scheduler started: %02d:%02d %s  weekdays_only=%s",
            self._hour, self._minute, self._timezone, self._weekdays,
        )

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")

    def run_now(self) -> dict:
        """Trigger an immediate watchlist scan (blocking — waits for completion)."""
        logger.info("run_now() triggered.")
        result = self._run_scan()
        return result or {}

    def next_run_time(self) -> Optional[datetime]:
        """Return the next scheduled run time, or None if not scheduled."""
        jobs = self._scheduler.get_jobs()
        if not jobs:
            return None
        return jobs[0].next_run_time

    @property
    def is_running(self) -> bool:
        """Whether the APScheduler is active."""
        return self._scheduler.running

    def _emit(self, event_type: str, data: dict) -> None:
        """Publish a lifecycle event to the admin event bus (if connected)."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception:
                pass  # never let event delivery break the scheduler

    def _run_scan(self) -> Optional[dict]:
        """Internal: invoke the scan function and record timing."""
        self._last_run = datetime.now(pytz.utc)
        tickers_count = 0
        self._emit("scheduler.run_started", {"timestamp": self._last_run.isoformat()})
        try:
            result = self._scan_fn()
            self._last_result = result
            elapsed   = result.get("elapsed_seconds", 0)
            tickers   = result.get("tickers_processed", 0)
            decisions = {
                r.get("ticker", "?"): r.get("decision", "?")
                for r in result.get("results", [])
            } if "results" in result else {}
            self._last_run_detail = {
                "timestamp":        self._last_run.isoformat(),
                "result":           "error" if "error" in result else "success",
                "tickers_processed": tickers,
                "elapsed_seconds":  elapsed,
                "decisions":        decisions,
                "error":            result.get("error"),
            }
            self._emit("scheduler.run_completed", {
                "tickers_processed": tickers,
                "elapsed_seconds":   elapsed,
                "decisions":         decisions,
            })
            return result
        except Exception as e:
            logger.error("Scheduled scan raised: %s", e)
            self._last_result = {"error": str(e)}
            self._last_run_detail = {
                "timestamp":        self._last_run.isoformat(),
                "result":           "error",
                "tickers_processed": 0,
                "elapsed_seconds":  0,
                "decisions":        {},
                "error":            str(e),
            }
            self._emit("scheduler.run_completed", {"error": str(e), "tickers_processed": 0})
            return self._last_result
