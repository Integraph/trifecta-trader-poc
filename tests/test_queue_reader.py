"""Tests for src/automation/queue_reader.py (Task 014).

Uses pytest's tmp_path fixture to create a real temporary queue directory —
no mocking of the filesystem, giving full lifecycle coverage.

Covers:
- poll_once() returns empty list for empty queue
- poll_once() picks up matching candidate files
- poll_once() ignores non-matching target_trader
- poll_once() ignores crypto candidates
- poll_once() skips files that exceed max_retries
- poll_once() processes candidates in priority order (high > medium > low)
- poll_once() processes oldest-first within same priority
- _process_candidate() moves file pending → processing → completed on success
- _process_candidate() moves file back to pending (retry_count++) on failure
- Completed JSON contains original scanner fields + analysis_result
- Failed JSON contains incremented retry_count and last_error
- _should_skip() returns True for wrong target_trader
- _should_skip() returns True for crypto asset_type
- _should_skip() returns True when retry_count >= max_retries
- _should_skip() returns False for valid candidates
- create_analyze_fn returns a callable
- pending_count property reflects queue state
- is_running property
"""

import json
import time
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidate(
    ticker="AAPL",
    priority="medium",
    target_trader="trifecta-trader",
    asset_type="stock",
    retry_count=0,
    opportunity_score=72.5,
    filename=None,
) -> dict:
    msg = {
        "scanner_id":       f"scan_20260304_143000_{ticker}",
        "timestamp":        "2026-03-04T14:30:00Z",
        "asset_type":       asset_type,
        "ticker":           ticker,
        "opportunity_score": opportunity_score,
        "catalysts":        ["Volume surge detected"],
        "signal_scores":    {"volume_surge": 0.80},
        "key_data":         {"current_price": 263.75},
        "target_trader":    target_trader,
        "priority":         priority,
        "status":           "pending",
    }
    if retry_count > 0:
        msg["retry_count"] = retry_count
    return msg


def _write_candidate(queue_dir: Path, msg: dict, filename: str = None) -> Path:
    ticker = msg.get("ticker", "AAPL")
    fname  = filename or f"20260304_143000_{ticker}.json"
    path   = queue_dir / "pending" / fname
    path.write_text(json.dumps(msg))
    return path


def _make_reader(tmp_path, analyze_fn=None, max_retries=2, cooldown=0):
    from src.automation.queue_reader import QueueReader
    if analyze_fn is None:
        analyze_fn = MagicMock(return_value={
            "decision":    "BUY",
            "quality_score": {"composite": 9.4},
            "cost_breakdown": {"total_usd": 0.06},
            "elapsed_seconds": 120.0,
        })
    (tmp_path / "pending").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processing").mkdir(parents=True, exist_ok=True)
    (tmp_path / "completed").mkdir(parents=True, exist_ok=True)
    return QueueReader(
        queue_dir=str(tmp_path),
        analyze_fn=analyze_fn,
        target_trader="trifecta-trader",
        poll_interval=1,
        max_retries=max_retries,
        cooldown_seconds=cooldown,
    )


# ── 1. Empty queue ────────────────────────────────────────────────────────────

class TestEmptyQueue:

    def test_poll_once_returns_empty_list(self, tmp_path):
        reader = _make_reader(tmp_path)
        result = reader.poll_once()
        assert result == []

    def test_pending_count_zero(self, tmp_path):
        reader = _make_reader(tmp_path)
        assert reader.pending_count == 0


# ── 2. Filtering ──────────────────────────────────────────────────────────────

class TestFiltering:

    def test_picks_up_matching_candidate(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"))
        results = reader.poll_once()
        assert len(results) == 1

    def test_ignores_wrong_target_trader(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate(target_trader="other-trader"))
        results = reader.poll_once()
        assert results == []

    def test_ignores_crypto_asset_type(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate(asset_type="crypto"))
        results = reader.poll_once()
        assert results == []

    def test_skips_max_retries_exceeded(self, tmp_path):
        reader = _make_reader(tmp_path, max_retries=2)
        _write_candidate(tmp_path, _make_candidate(retry_count=2))
        results = reader.poll_once()
        assert results == []

    def test_processes_candidate_below_max_retries(self, tmp_path):
        reader = _make_reader(tmp_path, max_retries=3)
        _write_candidate(tmp_path, _make_candidate(retry_count=2))
        results = reader.poll_once()
        assert len(results) == 1


# ── 3. _should_skip() ─────────────────────────────────────────────────────────

class TestShouldSkip:

    def test_skip_wrong_trader(self, tmp_path):
        reader = _make_reader(tmp_path)
        msg = _make_candidate(target_trader="crypto-trader")
        assert reader._should_skip(msg) is True

    def test_skip_crypto(self, tmp_path):
        reader = _make_reader(tmp_path)
        msg = _make_candidate(asset_type="crypto")
        assert reader._should_skip(msg) is True

    def test_skip_max_retries(self, tmp_path):
        reader = _make_reader(tmp_path, max_retries=2)
        msg = _make_candidate(retry_count=2)
        assert reader._should_skip(msg) is True

    def test_not_skip_valid_candidate(self, tmp_path):
        reader = _make_reader(tmp_path)
        msg = _make_candidate()
        assert reader._should_skip(msg) is False

    def test_not_skip_first_retry(self, tmp_path):
        reader = _make_reader(tmp_path, max_retries=2)
        msg = _make_candidate(retry_count=1)
        assert reader._should_skip(msg) is False


# ── 4. Priority sorting ───────────────────────────────────────────────────────

class TestPrioritySorting:

    def test_high_before_medium_before_low(self, tmp_path):
        processed_order = []

        def _analyze(ticker, ctx):
            processed_order.append(ticker)
            return {"decision": "BUY", "quality_score": {"composite": 8.0},
                    "cost_breakdown": {"total_usd": 0.05}, "elapsed_seconds": 10.0}

        reader = _make_reader(tmp_path, analyze_fn=_analyze)
        _write_candidate(tmp_path, _make_candidate("LOW",  priority="low"),    "c_LOW.json")
        _write_candidate(tmp_path, _make_candidate("HIGH", priority="high"),   "a_HIGH.json")
        _write_candidate(tmp_path, _make_candidate("MED",  priority="medium"), "b_MED.json")
        reader.poll_once()
        assert processed_order == ["HIGH", "MED", "LOW"]

    def test_same_priority_oldest_first(self, tmp_path):
        processed_order = []

        def _analyze(ticker, ctx):
            processed_order.append(ticker)
            return {"decision": "BUY", "quality_score": {"composite": 8.0},
                    "cost_breakdown": {"total_usd": 0.05}, "elapsed_seconds": 10.0}

        reader = _make_reader(tmp_path, analyze_fn=_analyze)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_080000_AAPL.json")
        _write_candidate(tmp_path, _make_candidate("MSFT"), "20260304_070000_MSFT.json")
        reader.poll_once()
        assert processed_order == ["MSFT", "AAPL"]


# ── 5. File lifecycle ─────────────────────────────────────────────────────────

class TestFileLifecycle:

    def test_success_removes_from_pending(self, tmp_path):
        reader = _make_reader(tmp_path)
        path = _write_candidate(tmp_path, _make_candidate("AAPL"))
        reader.poll_once()
        assert not path.exists()

    def test_success_creates_completed_file(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        completed = tmp_path / "completed" / "20260304_143000_AAPL.json"
        assert completed.exists()

    def test_completed_file_has_status_completed(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        data = json.loads((tmp_path / "completed" / "20260304_143000_AAPL.json").read_text())
        assert data["status"] == "completed"

    def test_completed_file_contains_analysis_result(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        data = json.loads((tmp_path / "completed" / "20260304_143000_AAPL.json").read_text())
        assert "analysis_result" in data
        assert data["analysis_result"]["decision"] == "BUY"

    def test_completed_file_preserves_scanner_fields(self, tmp_path):
        reader = _make_reader(tmp_path)
        msg = _make_candidate("AAPL", opportunity_score=88.5)
        _write_candidate(tmp_path, msg, "20260304_143000_AAPL.json")
        reader.poll_once()
        data = json.loads((tmp_path / "completed" / "20260304_143000_AAPL.json").read_text())
        assert data["opportunity_score"] == 88.5
        assert data["ticker"] == "AAPL"
        assert "completed_at" in data

    def test_processing_dir_empty_after_success(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        processing_files = list((tmp_path / "processing").glob("*.json"))
        assert processing_files == []


# ── 6. Failure / retry ────────────────────────────────────────────────────────

class TestRetryLogic:

    def test_failure_moves_file_back_to_pending(self, tmp_path):
        failing_fn = MagicMock(side_effect=Exception("LLM timeout"))
        reader = _make_reader(tmp_path, analyze_fn=failing_fn)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        pending = tmp_path / "pending" / "20260304_143000_AAPL.json"
        assert pending.exists()

    def test_failure_increments_retry_count(self, tmp_path):
        failing_fn = MagicMock(side_effect=Exception("timeout"))
        reader = _make_reader(tmp_path, analyze_fn=failing_fn)
        _write_candidate(tmp_path, _make_candidate("AAPL", retry_count=0),
                         "20260304_143000_AAPL.json")
        reader.poll_once()
        data = json.loads((tmp_path / "pending" / "20260304_143000_AAPL.json").read_text())
        assert data["retry_count"] == 1

    def test_failure_records_last_error(self, tmp_path):
        failing_fn = MagicMock(side_effect=Exception("specific error msg"))
        reader = _make_reader(tmp_path, analyze_fn=failing_fn)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        data = json.loads((tmp_path / "pending" / "20260304_143000_AAPL.json").read_text())
        assert "specific error msg" in data.get("last_error", "")

    def test_failure_clears_processing_dir(self, tmp_path):
        failing_fn = MagicMock(side_effect=Exception("fail"))
        reader = _make_reader(tmp_path, analyze_fn=failing_fn)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        reader.poll_once()
        processing_files = list((tmp_path / "processing").glob("*.json"))
        assert processing_files == []


# ── 7. is_running property ────────────────────────────────────────────────────

class TestIsRunning:

    def test_is_running_false_before_start(self, tmp_path):
        reader = _make_reader(tmp_path)
        assert reader.is_running is False

    def test_stop_sets_running_false(self, tmp_path):
        reader = _make_reader(tmp_path)
        reader._running = True
        reader.stop()
        assert reader.is_running is False


# ── 8. pending_count ──────────────────────────────────────────────────────────

class TestPendingCount:

    def test_count_zero_initially(self, tmp_path):
        reader = _make_reader(tmp_path)
        assert reader.pending_count == 0

    def test_count_increments_with_files(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "a.json")
        _write_candidate(tmp_path, _make_candidate("MSFT"), "b.json")
        assert reader.pending_count == 2

    def test_count_decrements_after_processing(self, tmp_path):
        reader = _make_reader(tmp_path)
        _write_candidate(tmp_path, _make_candidate("AAPL"), "20260304_143000_AAPL.json")
        assert reader.pending_count == 1
        reader.poll_once()
        assert reader.pending_count == 0


# ── 9. create_analyze_fn ─────────────────────────────────────────────────────

class TestCreateAnalyzeFn:

    def test_returns_callable(self):
        from src.automation.queue_reader import create_analyze_fn
        fn = create_analyze_fn("hybrid_haiku_tools", publish=False)
        assert callable(fn)

    def test_callable_accepts_ticker_and_context(self):
        import sys, types
        from src.automation.queue_reader import create_analyze_fn

        mock_result = {
            "decision": "BUY",
            "quality_score": {"composite": 9.0},
            "cost_breakdown": {"total_usd": 0.05},
            "elapsed_seconds": 60.0,
        }

        # Inject fake src.run_analysis to avoid triggering full pipeline import
        fake_ra_mod = types.ModuleType("src.run_analysis")
        fake_ra_mod.run_analysis = MagicMock(return_value=mock_result)
        fake_ra_mod.build_portfolio_context = MagicMock(return_value={"held": False})
        fake_ra_mod._publish_signal = MagicMock()

        with patch.dict(sys.modules, {"src.run_analysis": fake_ra_mod}):
            fn = create_analyze_fn("hybrid_haiku_tools", publish=False)
            result = fn("AAPL", {"opportunity_score": 72.5, "catalysts": []})

        assert result["decision"] == "BUY"

    def test_scanner_context_injected_into_portfolio_context(self):
        import sys, types
        from src.automation.queue_reader import create_analyze_fn

        captured_ctx = {}

        def _fake_run_analysis(ticker, **kwargs):
            captured_ctx.update(kwargs.get("portfolio_context", {}))
            return {"decision": "BUY", "quality_score": {"composite": 9.0},
                    "cost_breakdown": {}, "elapsed_seconds": 1.0}

        fake_ra_mod = types.ModuleType("src.run_analysis")
        fake_ra_mod.run_analysis = MagicMock(side_effect=_fake_run_analysis)
        fake_ra_mod.build_portfolio_context = MagicMock(return_value={"held": False})
        fake_ra_mod._publish_signal = MagicMock()

        with patch.dict(sys.modules, {"src.run_analysis": fake_ra_mod}):
            fn = create_analyze_fn("hybrid_haiku_tools", publish=False)
            fn("AAPL", {"opportunity_score": 88.0, "catalysts": ["Volume surge"]})

        assert "scanner_context" in captured_ctx
        assert captured_ctx["scanner_context"]["opportunity_score"] == 88.0
