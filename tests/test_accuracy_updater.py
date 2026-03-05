"""Tests for src/accuracy/updater.py (Task 015).

yfinance is mocked throughout — no network calls.

Covers:
- run_update() processes pending outcomes
- run_update() marks outcomes complete when T+10 available
- run_update() updates partial when only T+1 available
- run_update() handles fetch errors (marks error, continues)
- run_update() handles empty price dict (skips, continues)
- run_update() summary dict has correct counts
- run_update() ticker filter limits to one ticker
- backfill() creates outcome records for analyses without one
- backfill() does not duplicate existing outcome records
- backfill() calls run_update() after creating records
- CLI --report calls AccuracyReporter.print_summary
- CLI --backfill N calls updater.backfill with correct days
"""

import types
import pytest
import tempfile
import os
import pandas as pd
from datetime import date, timedelta
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01",
                     decision="BUY", quality_score=9.0,
                     entry_price=195.0, stop_loss=180.0, price_target=220.0) -> int:
    row = {
        "ticker": ticker, "trade_date": trade_date,
        "run_timestamp": "2026-03-01T09:00:00",
        "config": "hybrid_haiku_tools", "decision": decision,
        "quality_score": quality_score, "cost_usd": 0.06, "elapsed_seconds": 60.0,
        "stop_loss": stop_loss, "price_target": price_target, "entry_price": entry_price,
        "position_size_pct": 5.0, "risk_reward": 2.5, "actionable": 1,
        "portfolio_equity": 100000.0, "held_at_analysis": 0,
        "held_shares": 0, "held_avg_cost": None, "result_file": None,
    }
    return db.upsert_analysis(row)


def _make_updater(db):
    from src.accuracy.updater import AccuracyUpdater
    return AccuracyUpdater(db=db)


def _full_prices():
    """Return a complete set of prices including T+10."""
    return {
        "price_at_signal": 194.0,
        "price_t1": 196.0,
        "price_t5": 200.0,
        "price_t10": 205.0,
        "high_t10": 210.0,
        "low_t10": 190.0,
        "daily_highs": [197.0] * 10,
        "daily_lows":  [192.0] * 10,
    }


def _partial_prices():
    """Return only T+1 (T+5 and T+10 not yet available)."""
    return {
        "price_at_signal": 194.0,
        "price_t1": 196.0,
    }


# ── 1. run_update() ───────────────────────────────────────────────────────────

class TestRunUpdate:

    def test_marks_complete_when_t10_available(self):
        db = _make_db()
        aid = _insert_analysis(db)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY",
                                        entry_price=195.0, stop_loss=180.0, price_target=220.0)
        with patch("src.accuracy.updater.fetch_outcome_prices", return_value=_full_prices()):
            result = updater.run_update()

        outcome = updater.tracker.get_outcome(aid)
        assert outcome["status"] == "complete"
        assert result["newly_complete"] == 1

    def test_partial_update_when_only_t1_available(self):
        db = _make_db()
        aid = _insert_analysis(db)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")

        with patch("src.accuracy.updater.fetch_outcome_prices", return_value=_partial_prices()):
            result = updater.run_update()

        outcome = updater.tracker.get_outcome(aid)
        assert outcome["status"] == "partial"
        assert outcome["price_t1"] == 196.0
        assert result["newly_complete"] == 0
        assert result["updated"] == 1

    def test_marks_error_on_fetch_exception(self):
        db = _make_db()
        aid = _insert_analysis(db)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")

        with patch("src.accuracy.updater.fetch_outcome_prices", side_effect=Exception("delisted")):
            result = updater.run_update()

        outcome = updater.tracker.get_outcome(aid)
        assert outcome["status"] == "error"
        assert result["errors"] == 1

    def test_skips_on_empty_prices(self):
        db = _make_db()
        aid = _insert_analysis(db)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")

        with patch("src.accuracy.updater.fetch_outcome_prices", return_value={}):
            result = updater.run_update()

        assert result["skipped"] == 1
        assert result["updated"]  == 0

    def test_summary_counts_are_correct(self):
        db = _make_db()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02", quality_score=8.0)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid1, "AAPL", "2026-03-01", "BUY")
        updater.tracker.create_outcome(aid2, "MSFT", "2026-03-02", "SELL")

        prices = [_full_prices(), _partial_prices()]
        with patch("src.accuracy.updater.fetch_outcome_prices", side_effect=prices):
            result = updater.run_update()

        assert result["total_pending"]  == 2
        assert result["updated"]        == 2
        assert result["newly_complete"] == 1  # only AAPL has T+10

    def test_ticker_filter(self):
        db = _make_db()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02", quality_score=8.0)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid1, "AAPL", "2026-03-01", "BUY")
        updater.tracker.create_outcome(aid2, "MSFT", "2026-03-02", "SELL")

        with patch("src.accuracy.updater.fetch_outcome_prices", return_value=_full_prices()):
            result = updater.run_update(ticker="AAPL")

        assert result["total_pending"] == 1  # only AAPL processed
        # MSFT still pending
        assert updater.tracker.get_outcome(aid2)["status"] == "pending"

    def test_scores_applied_after_t10_available(self):
        db = _make_db()
        aid = _insert_analysis(db, entry_price=195.0)
        updater = _make_updater(db)
        updater.tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY",
                                        entry_price=195.0, stop_loss=180.0, price_target=220.0)

        prices = {**_full_prices(), "price_t5": 200.0}
        with patch("src.accuracy.updater.fetch_outcome_prices", return_value=prices):
            updater.run_update()

        outcome = updater.tracker.get_outcome(aid)
        # direction_correct_t5: 200 > 195 → 1
        assert outcome["direction_correct_t5"] == 1
        assert outcome["return_t5_pct"] is not None


# ── 2. backfill() ─────────────────────────────────────────────────────────────

class TestBackfill:

    def test_creates_records_for_existing_analyses(self):
        db = _make_db()
        # Analysis from 5 days ago (within backfill window)
        past_date = (date.today() - timedelta(days=5)).isoformat()
        aid = _insert_analysis(db, trade_date=past_date)
        updater = _make_updater(db)

        with patch("src.accuracy.updater.fetch_outcome_prices", return_value={}):
            result = updater.backfill(days_back=10)

        assert result["created"] == 1
        outcome = updater.tracker.get_outcome(aid)
        assert outcome is not None

    def test_does_not_duplicate_existing_outcomes(self):
        db = _make_db()
        past_date = (date.today() - timedelta(days=5)).isoformat()
        aid = _insert_analysis(db, trade_date=past_date)
        updater = _make_updater(db)
        # Pre-create the outcome
        updater.tracker.create_outcome(aid, "AAPL", past_date, "BUY")

        with patch("src.accuracy.updater.fetch_outcome_prices", return_value={}):
            result = updater.backfill(days_back=10)

        # Should create 0 new records (already exists)
        assert result["created"] == 0

    def test_ignores_analyses_outside_window(self):
        db = _make_db()
        old_date = (date.today() - timedelta(days=60)).isoformat()
        _insert_analysis(db, trade_date=old_date)
        updater = _make_updater(db)

        with patch("src.accuracy.updater.fetch_outcome_prices", return_value={}):
            result = updater.backfill(days_back=30)

        assert result["created"] == 0


# ── 3. CLI ────────────────────────────────────────────────────────────────────

class TestCLI:

    def test_report_flag_calls_print_summary(self):
        mock_reporter = MagicMock()
        mock_reporter_cls = MagicMock(return_value=mock_reporter)

        with patch("sys.argv", ["updater", "--report"]):
            with patch("src.accuracy.updater.AccuracyUpdater") as MockUpdater:
                mock_updater_inst = MagicMock()
                mock_updater_inst.tracker._db = MagicMock()
                MockUpdater.return_value = mock_updater_inst
                # AccuracyReporter is lazily imported inside main(), so
                # patch where it is defined
                with patch("src.accuracy.reporter.AccuracyReporter", mock_reporter_cls):
                    from src.accuracy.updater import main
                    main()

        mock_reporter.print_summary.assert_called_once()

    def test_backfill_flag_calls_backfill_with_days(self):
        with patch("sys.argv", ["updater", "--backfill", "30"]):
            with patch("src.accuracy.updater.AccuracyUpdater") as MockUpdater:
                mock_inst = MagicMock()
                mock_inst.backfill.return_value = {"created": 5}
                MockUpdater.return_value = mock_inst
                from src.accuracy.updater import main
                main()

        mock_inst.backfill.assert_called_once_with(days_back=30)

    def test_default_runs_update(self):
        with patch("sys.argv", ["updater"]):
            with patch("src.accuracy.updater.AccuracyUpdater") as MockUpdater:
                mock_inst = MagicMock()
                mock_inst.run_update.return_value = {"total_pending": 0}
                MockUpdater.return_value = mock_inst
                from src.accuracy.updater import main
                main()

        mock_inst.run_update.assert_called_once()
