"""Tests for EventBus and make_event_callback (Task 016)."""

import asyncio
import pytest
from unittest.mock import MagicMock


# ── EventBus unit tests ───────────────────────────────────────────────────────

class TestEventBus:

    def _fresh_bus(self):
        from src.admin.events import EventBus
        bus  = EventBus()
        loop = asyncio.new_event_loop()
        bus.set_loop(loop)
        return bus, loop

    def test_subscribe_returns_queue(self):
        from src.admin.events import EventBus
        bus = EventBus()
        q   = bus.subscribe()
        assert q is not None
        assert q in bus._subscribers

    def test_unsubscribe_removes_queue(self):
        from src.admin.events import EventBus
        bus = EventBus()
        q   = bus.subscribe()
        bus.unsubscribe(q)
        assert q not in bus._subscribers

    def test_unsubscribe_unknown_queue_is_noop(self):
        from src.admin.events import EventBus
        bus = EventBus()
        bus.unsubscribe(asyncio.Queue())  # should not raise

    def test_publish_stores_in_history(self):
        from src.admin.events import EventBus
        bus = EventBus()
        bus.publish("test.event", {"key": "value"})
        history = bus.recent_events()
        assert len(history) == 1
        assert history[0]["event"] == "test.event"

    def test_history_bounded_by_max(self):
        from src.admin.events import EventBus
        bus = EventBus()
        bus._max_history = 5
        for i in range(10):
            bus.publish(f"event.{i}", {})
        assert len(bus.recent_events()) <= 5

    def test_recent_events_returns_last_n(self):
        from src.admin.events import EventBus
        bus = EventBus()
        for i in range(10):
            bus.publish(f"evt.{i}", {"i": i})
        recent = bus.recent_events(limit=3)
        assert len(recent) == 3
        assert recent[-1]["data"]["i"] == 9  # last event

    def test_publish_to_subscribers_via_loop(self):
        from src.admin.events import EventBus

        async def _run():
            bus  = EventBus()
            loop = asyncio.get_event_loop()
            bus.set_loop(loop)
            q = bus.subscribe()
            # Publish from within the loop (no threadsafe needed)
            await bus._broadcast({"event": "x", "data": {}, "timestamp": "t"})
            assert not q.empty()
            item = await q.get()
            assert item["event"] == "x"

        asyncio.run(_run())


# ── make_event_callback ───────────────────────────────────────────────────────

class TestMakeEventCallback:

    def test_callback_publishes_to_bus(self):
        import src.admin.events as ev_mod
        from src.admin.events import EventBus
        original = ev_mod._event_bus
        try:
            bus = EventBus()
            ev_mod._event_bus = bus
            callback = ev_mod.make_event_callback()
            callback("scheduler.run_completed", {"tickers_processed": 5})
            history = bus.recent_events()
            assert len(history) == 1
            assert history[0]["event"] == "scheduler.run_completed"
        finally:
            ev_mod._event_bus = original

    def test_callback_does_not_raise_on_bus_error(self):
        from src.admin.events import EventBus, make_event_callback
        import src.admin.events as ev_mod
        original = ev_mod._event_bus
        try:
            broken_bus = MagicMock()
            broken_bus.publish.side_effect = RuntimeError("bus error")
            ev_mod._event_bus = broken_bus
            cb = make_event_callback()
            cb("test.event", {})  # should not raise
        finally:
            ev_mod._event_bus = original


# ── Event types match documented schema ───────────────────────────────────────

class TestEventTypes:
    """Verify documented event types are produced by subsystem emit calls."""

    def test_scheduler_events_have_correct_type_names(self):
        # Verifies the event type strings in scheduler._emit() match the spec
        expected = {"scheduler.run_started", "scheduler.run_completed"}
        # Read the scheduler source and check strings are present
        import inspect
        import src.automation.scheduler as sched_mod
        source = inspect.getsource(sched_mod.PipelineScheduler._emit)
        # _emit itself just calls callback — check the call sites
        source_full = inspect.getsource(sched_mod.PipelineScheduler._run_scan)
        assert "scheduler.run_started"   in source_full
        assert "scheduler.run_completed" in source_full

    def test_queue_reader_events_have_correct_type_names(self):
        import inspect
        import src.automation.queue_reader as qr_mod
        source = inspect.getsource(qr_mod.QueueReader._process_candidate)
        assert "queue.candidate_picked"    in source
        assert "queue.analysis_completed" in source

    def test_accuracy_updater_events_have_correct_type_names(self):
        import inspect
        import src.accuracy.updater as upd_mod
        source = inspect.getsource(upd_mod.AccuracyUpdater.run_update)
        assert "accuracy.update_started"   in source
        assert "accuracy.update_completed" in source
