"""Tests for src/accuracy/reporter.py (Task 015).

Uses in-memory SQLite and pre-seeded outcome rows.

Covers:
- summary() with no complete outcomes returns zeroed dict
- summary() correctly aggregates direction accuracy
- summary() correctly aggregates target/stop hit rates
- summary() correctly aggregates avg return percentages
- summary() breaks down by quality tier (high/medium/low)
- summary() best_signals / worst_signals sorted correctly
- ticker_report() returns signals for correct ticker
- ticker_report() separates complete vs pending
- print_summary() does not raise with complete data
- print_summary() handles zero complete signals gracefully
- _aggregate() helper computes correct averages
- Quality tier breakdown: high-quality signals have better accuracy
"""

import io
import pytest
import tempfile
import os
from unittest.mock import patch


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


def _insert_complete_outcome(db, analysis_id, ticker, signal_date, decision="BUY",
                              entry_price=195.0, stop_loss=180.0, price_target=220.0,
                              price_t5=200.0, price_t10=210.0,
                              direction_correct_t5=1, direction_correct_t10=1,
                              target_hit=1, stop_hit=0, return_t5_pct=2.56,
                              return_t10_pct=7.69, max_favorable_pct=10.0,
                              max_adverse_pct=-5.0):
    """Insert a fully scored, complete outcome row."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO signal_outcomes (
            analysis_id, ticker, signal_date, decision,
            entry_price, stop_loss, price_target,
            price_at_signal, price_t1, price_t5, price_t10,
            high_t10, low_t10,
            direction_correct_t1, direction_correct_t5, direction_correct_t10,
            target_hit, stop_hit,
            return_t1_pct, return_t5_pct, return_t10_pct,
            max_favorable_pct, max_adverse_pct,
            status, last_updated
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'complete',?)
    """
    with db._conn() as conn:
        conn.execute(sql, (
            analysis_id, ticker, signal_date, decision,
            entry_price, stop_loss, price_target,
            entry_price, entry_price * 1.01, price_t5, price_t10,
            price_t10 + 5, entry_price * 0.95,
            1, direction_correct_t5, direction_correct_t10,
            target_hit, stop_hit,
            1.03, return_t5_pct, return_t10_pct,
            max_favorable_pct, max_adverse_pct,
            now,
        ))


def _make_reporter(db=None):
    from src.accuracy.reporter import AccuracyReporter
    if db is None:
        db = _make_db()
    return AccuracyReporter(db), db


# ── 1. summary() ─────────────────────────────────────────────────────────────

class TestSummary:

    def test_empty_database_returns_zero_totals(self):
        reporter, _ = _make_reporter()
        result = reporter.summary(days=30)
        assert result["total_signals"] == 0
        assert result["by_decision"]   == {}

    def test_counts_complete_outcomes(self):
        reporter, db = _make_reporter()
        aid = _insert_analysis(db)
        _insert_complete_outcome(db, aid, "AAPL", "2026-03-01")
        result = reporter.summary(days=30)
        assert result["total_signals"] == 1

    def test_excludes_pending_outcomes(self):
        reporter, db = _make_reporter()
        aid = _insert_analysis(db)
        # Only create pending (no scores)
        from src.accuracy.price_tracker import PriceTracker
        tracker = PriceTracker(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        result = reporter.summary(days=30)
        assert result["total_signals"] == 0

    def test_aggregates_by_decision(self):
        reporter, db = _make_reporter()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01", decision="BUY")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02",
                                  decision="SELL", quality_score=8.0)
        _insert_complete_outcome(db, aid1, "AAPL", "2026-03-01", decision="BUY")
        _insert_complete_outcome(db, aid2, "MSFT", "2026-03-02", decision="SELL")
        result = reporter.summary(days=30)
        assert "BUY"  in result["by_decision"]
        assert "SELL" in result["by_decision"]

    def test_direction_accuracy_aggregation(self):
        reporter, db = _make_reporter()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02", quality_score=8.0)
        _insert_complete_outcome(db, aid1, "AAPL", "2026-03-01", direction_correct_t5=1)
        _insert_complete_outcome(db, aid2, "MSFT", "2026-03-02", direction_correct_t5=0)
        result = reporter.summary(days=30)
        # 50% accuracy for BUY (1 correct, 1 wrong)
        assert result["by_decision"]["BUY"]["direction_correct_t5"] == 0.5

    def test_quality_tier_breakdown(self):
        reporter, db = _make_reporter()
        aid_high = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01",
                                     quality_score=9.0)
        aid_low  = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02",
                                     quality_score=4.0)
        _insert_complete_outcome(db, aid_high, "AAPL", "2026-03-01", direction_correct_t5=1)
        _insert_complete_outcome(db, aid_low,  "MSFT", "2026-03-02", direction_correct_t5=0)
        result = reporter.summary(days=30)
        tiers  = result["by_quality_tier"]
        assert "high (8-10)"  in tiers
        assert "low (0-6)"    in tiers
        assert tiers["high (8-10)"]["direction_correct_t5"] == 1.0
        assert tiers["low (0-6)"]["direction_correct_t5"]   == 0.0

    def test_best_and_worst_signals(self):
        reporter, db = _make_reporter()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02", quality_score=8.0)
        _insert_complete_outcome(db, aid1, "AAPL", "2026-03-01", return_t10_pct=15.0)
        _insert_complete_outcome(db, aid2, "MSFT", "2026-03-02", return_t10_pct=-5.0)
        result = reporter.summary(days=30)
        assert result["best_signals"][0]["ticker"]          == "AAPL"
        assert result["worst_signals"][0]["ticker"]         == "MSFT"

    def test_respects_days_filter(self):
        from datetime import date, timedelta
        reporter, db = _make_reporter()
        old_date    = (date.today() - timedelta(days=60)).isoformat()
        recent_date = (date.today() - timedelta(days=5)).isoformat()
        aid_old    = _insert_analysis(db, ticker="AAPL", trade_date=old_date)
        aid_recent = _insert_analysis(db, ticker="MSFT", trade_date=recent_date, quality_score=8.0)
        _insert_complete_outcome(db, aid_old,    "AAPL", old_date)
        _insert_complete_outcome(db, aid_recent, "MSFT", recent_date)
        # 30-day window should exclude old_date
        result = reporter.summary(days=30)
        assert result["total_signals"] == 1


# ── 2. ticker_report() ────────────────────────────────────────────────────────

class TestTickerReport:

    def test_returns_only_ticker_signals(self):
        reporter, db = _make_reporter()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02", quality_score=8.0)
        _insert_complete_outcome(db, aid1, "AAPL", "2026-03-01")
        _insert_complete_outcome(db, aid2, "MSFT", "2026-03-02")
        report = reporter.ticker_report("AAPL")
        assert report["ticker"] == "AAPL"
        assert report["total_signals"] == 1
        assert all(s["ticker"] == "AAPL" for s in report["signals"])

    def test_separates_complete_and_pending(self):
        reporter, db = _make_reporter()
        from src.accuracy.price_tracker import PriceTracker
        tracker = PriceTracker(db)
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-02", quality_score=8.5)
        _insert_complete_outcome(db, aid1, "AAPL", "2026-03-01")
        tracker.create_outcome(aid2, "AAPL", "2026-03-02", "BUY")
        report = reporter.ticker_report("AAPL")
        assert report["complete"] == 1
        assert report["pending"]  == 1
        assert report["total_signals"] == 2

    def test_empty_ticker_returns_zero_counts(self):
        reporter, _ = _make_reporter()
        report = reporter.ticker_report("ZZZZ")
        assert report["total_signals"] == 0
        assert report["complete"]      == 0


# ── 3. print_summary() ───────────────────────────────────────────────────────

class TestPrintSummary:

    def test_does_not_raise_with_data(self, capsys):
        reporter, db = _make_reporter()
        aid = _insert_analysis(db)
        _insert_complete_outcome(db, aid, "AAPL", "2026-03-01")
        reporter.print_summary(days=30)  # Should not raise
        captured = capsys.readouterr()
        assert "TRIFECTA TRADER" in captured.out

    def test_handles_zero_complete_signals(self, capsys):
        reporter, _ = _make_reporter()
        reporter.print_summary(days=30)
        captured = capsys.readouterr()
        assert "No complete signals" in captured.out


# ── 4. _aggregate() helper ────────────────────────────────────────────────────

class TestAggregate:

    def test_correct_averages(self):
        from src.accuracy.reporter import _aggregate
        rows = [
            {"direction_correct_t5": 1, "return_t5_pct": 5.0, "target_hit": 1, "stop_hit": 0},
            {"direction_correct_t5": 0, "return_t5_pct": -2.0, "target_hit": 0, "stop_hit": 1},
        ]
        agg = _aggregate(rows)
        assert agg["count"]                 == 2
        assert agg["direction_correct_t5"]  == 0.5
        assert abs(agg["avg_return_t5_pct"] - 1.5) < 0.001
        assert agg["target_hit_rate"]       == 0.5
        assert agg["stop_hit_rate"]         == 0.5

    def test_handles_none_values(self):
        from src.accuracy.reporter import _aggregate
        rows = [
            {"direction_correct_t5": 1, "return_t5_pct": None},
            {"direction_correct_t5": None, "return_t5_pct": 3.0},
        ]
        agg = _aggregate(rows)
        # direction_correct_t5: only 1 non-None value, 1/1 = 1.0
        assert agg["direction_correct_t5"] == 1.0
        # avg_return_t5_pct: only 1 non-None value = 3.0
        assert agg["avg_return_t5_pct"]   == 3.0

    def test_empty_list_returns_zero_count(self):
        from src.accuracy.reporter import _aggregate
        agg = _aggregate([])
        assert agg["count"] == 0
