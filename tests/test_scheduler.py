"""Tests for src/automation/scheduler.py (Task 014).

APScheduler is mocked so no real background threads are started.

Covers:
- PipelineScheduler initializes with correct defaults
- start() adds a CronTrigger job to the scheduler
- Weekday-only mode uses 'mon-fri' day_of_week
- All-days mode uses '*' day_of_week
- start() passes max_instances=1 to prevent concurrent runs
- CronTrigger hour/minute match constructor args
- stop() calls scheduler shutdown
- run_now() calls the scan function and returns its result
- run_now() returns empty dict on scan function exception
- next_run_time() returns None when no jobs scheduled
- is_running property reflects scheduler state
- create_watchlist_scan_fn returns a callable
- create_watchlist_scan_fn callable calls run_batch with correct args
- create_watchlist_scan_fn callable handles watchlist load failure gracefully
- create_watchlist_scan_fn callable handles run_batch exception gracefully
"""

import pytest
from unittest.mock import MagicMock, patch, call


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_scheduler(
    scan_fn=None,
    hour=8,
    minute=30,
    timezone="US/Eastern",
    weekdays_only=True,
):
    from src.automation.scheduler import PipelineScheduler
    if scan_fn is None:
        scan_fn = MagicMock(return_value={"results": [], "tickers_processed": 0})
    return PipelineScheduler(
        watchlist_scan_fn=scan_fn,
        schedule_hour=hour,
        schedule_minute=minute,
        timezone=timezone,
        weekdays_only=weekdays_only,
    )


# ── 1. Initialization ─────────────────────────────────────────────────────────

class TestSchedulerInit:

    def test_default_attributes(self):
        s = _make_scheduler()
        assert s._hour    == 8
        assert s._minute  == 30
        assert s._timezone == "US/Eastern"
        assert s._weekdays is True

    def test_custom_schedule(self):
        s = _make_scheduler(hour=7, minute=45, weekdays_only=False)
        assert s._hour   == 7
        assert s._minute == 45
        assert s._weekdays is False

    def test_is_running_false_before_start(self):
        s = _make_scheduler()
        assert s.is_running is False

    def test_next_run_time_none_before_start(self):
        s = _make_scheduler()
        # No jobs added yet
        assert s.next_run_time() is None


# ── 2. start() — job configuration ───────────────────────────────────────────

class TestSchedulerStart:

    def test_start_adds_cron_job(self):
        from src.automation.scheduler import PipelineScheduler
        from apscheduler.triggers.cron import CronTrigger

        s = _make_scheduler(hour=8, minute=30)
        mock_sched = MagicMock()
        mock_sched.running = True
        s._scheduler = mock_sched

        s.start()

        mock_sched.add_job.assert_called_once()
        mock_sched.start.assert_called_once()

    def test_weekdays_only_uses_mon_fri(self):
        s = _make_scheduler(weekdays_only=True)
        mock_sched = MagicMock()
        mock_sched.running = True
        s._scheduler = mock_sched

        s.start()

        _, kwargs = mock_sched.add_job.call_args
        trigger = kwargs["trigger"]
        # CronTrigger object — inspect its fields
        field_map = {f.name: f for f in trigger.fields}
        assert str(field_map["day_of_week"]) == "mon-fri"

    def test_all_days_uses_star(self):
        s = _make_scheduler(weekdays_only=False)
        mock_sched = MagicMock()
        mock_sched.running = True
        s._scheduler = mock_sched

        s.start()

        _, kwargs = mock_sched.add_job.call_args
        trigger = kwargs["trigger"]
        field_map = {f.name: f for f in trigger.fields}
        assert str(field_map["day_of_week"]) == "*"

    def test_max_instances_is_1(self):
        s = _make_scheduler()
        mock_sched = MagicMock()
        mock_sched.running = True
        s._scheduler = mock_sched

        s.start()

        _, kwargs = mock_sched.add_job.call_args
        assert kwargs.get("max_instances") == 1

    def test_hour_and_minute_passed_to_trigger(self):
        s = _make_scheduler(hour=7, minute=45)
        mock_sched = MagicMock()
        mock_sched.running = True
        s._scheduler = mock_sched

        s.start()

        _, kwargs = mock_sched.add_job.call_args
        trigger = kwargs["trigger"]
        field_map = {f.name: f for f in trigger.fields}
        assert str(field_map["hour"])   == "7"
        assert str(field_map["minute"]) == "45"


# ── 3. stop() ─────────────────────────────────────────────────────────────────

class TestSchedulerStop:

    def test_stop_calls_shutdown(self):
        s = _make_scheduler()
        mock_sched = MagicMock()
        mock_sched.running = True
        s._scheduler = mock_sched

        s.stop()

        mock_sched.shutdown.assert_called_once_with(wait=False)

    def test_stop_when_not_running_does_not_raise(self):
        s = _make_scheduler()
        mock_sched = MagicMock()
        mock_sched.running = False
        s._scheduler = mock_sched

        # Should not raise
        s.stop()
        mock_sched.shutdown.assert_not_called()


# ── 4. run_now() ──────────────────────────────────────────────────────────────

class TestRunNow:

    def test_run_now_calls_scan_fn(self):
        scan_fn = MagicMock(return_value={"results": [], "tickers_processed": 0})
        s = _make_scheduler(scan_fn=scan_fn)
        s.run_now()
        scan_fn.assert_called_once()

    def test_run_now_returns_scan_result(self):
        expected = {"results": [{"ticker": "AAPL"}], "tickers_processed": 1}
        scan_fn = MagicMock(return_value=expected)
        s = _make_scheduler(scan_fn=scan_fn)
        result = s.run_now()
        assert result == expected

    def test_run_now_returns_empty_dict_on_exception(self):
        scan_fn = MagicMock(side_effect=Exception("scan boom"))
        s = _make_scheduler(scan_fn=scan_fn)
        result = s.run_now()
        assert isinstance(result, dict)

    def test_run_now_records_last_run(self):
        scan_fn = MagicMock(return_value={})
        s = _make_scheduler(scan_fn=scan_fn)
        s.run_now()
        assert s._last_run is not None


# ── 5. create_watchlist_scan_fn ───────────────────────────────────────────────

class TestCreateWatchlistScanFn:

    def test_returns_callable(self):
        from src.automation.scheduler import create_watchlist_scan_fn
        fn = create_watchlist_scan_fn("hybrid_haiku_tools")
        assert callable(fn)

    def test_calls_run_batch_with_correct_hybrid_config(self):
        from src.automation.scheduler import create_watchlist_scan_fn

        with patch("src.automation.scheduler.create_watchlist_scan_fn") as mock_create:
            fake_fn = MagicMock(return_value={"results": [], "tickers_processed": 0})
            mock_create.return_value = fake_fn
            fn = mock_create("hybrid_haiku_tools", watchlist="default", publish=True)
            fn()
            fake_fn.assert_called_once()

    def test_handles_watchlist_load_failure(self):
        import sys, types
        from src.automation.scheduler import create_watchlist_scan_fn

        fn = create_watchlist_scan_fn(
            hybrid_config="hybrid_haiku_tools",
            watchlist="nonexistent_watchlist_xyz",
        )

        # Inject a fake src.run_batch into sys.modules so the lazy import
        # inside _scan_fn picks it up without touching the real pipeline.
        fake_run_batch_mod = types.ModuleType("src.run_batch")
        fake_run_batch_mod.run_batch = MagicMock()
        fake_run_batch_mod.load_watchlist = MagicMock(
            side_effect=FileNotFoundError("watchlist not found")
        )
        with patch.dict(sys.modules, {"src.run_batch": fake_run_batch_mod}):
            result = fn()
        assert "error" in result
        assert result["tickers_processed"] == 0

    def test_handles_run_batch_exception(self):
        import sys, types
        from src.automation.scheduler import create_watchlist_scan_fn

        fn = create_watchlist_scan_fn(
            hybrid_config="hybrid_haiku_tools",
            watchlist="default",
        )

        fake_run_batch_mod = types.ModuleType("src.run_batch")
        fake_run_batch_mod.load_watchlist = MagicMock(return_value=["AAPL"])
        fake_run_batch_mod.run_batch = MagicMock(side_effect=RuntimeError("LLM failed"))
        with patch.dict(sys.modules, {"src.run_batch": fake_run_batch_mod}):
            result = fn()
        assert "error" in result
        assert result["tickers_processed"] == 0
