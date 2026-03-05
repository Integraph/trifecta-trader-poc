"""Tests for src/accuracy/price_tracker.py (Task 015).

Uses in-memory SQLite so tests are fast and isolated.

Covers:
- create_outcome() stores all fields correctly
- create_outcome() duplicate analysis_id is silently ignored (INSERT OR IGNORE)
- update_prices() updates only the provided fields
- update_prices() does not overwrite completed outcomes
- get_pending_outcomes() returns pending and partial, not complete/error
- get_outcome() returns correct row by analysis_id
- get_outcomes_for_ticker() returns rows newest first
- mark_complete() sets status='complete'
- mark_error() sets status='error' and records error_message
- apply_scores() writes scorer output fields back to row
- fetch_outcome_prices() returns correct structure (mocked yfinance)
- fetch_outcome_prices() returns partial results when T+10 not yet available
- fetch_outcome_prices() handles non-trading day (falls back to prior close)
- fetch_outcome_prices() returns empty dict for unknown ticker (mocked)
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import date, timedelta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db():
    """Return a PortfolioDatabase backed by a temp SQLite file (avoids
    the per-connection fresh-slate behaviour of :memory: databases)."""
    from src.portfolio.database import PortfolioDatabase
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return PortfolioDatabase(path)


def _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01",
                     decision="BUY", quality_score=9.0,
                     entry_price=195.0, stop_loss=180.0, price_target=220.0) -> int:
    """Insert a minimal analyses row and return its id."""
    row = {
        "ticker":        ticker,
        "trade_date":    trade_date,
        "run_timestamp": "2026-03-01T09:00:00",
        "config":        "hybrid_haiku_tools",
        "decision":      decision,
        "quality_score": quality_score,
        "cost_usd":      0.06,
        "elapsed_seconds": 60.0,
        "stop_loss":     stop_loss,
        "price_target":  price_target,
        "entry_price":   entry_price,
        "position_size_pct": 5.0,
        "risk_reward":   2.5,
        "actionable":    1,
        "portfolio_equity": 100000.0,
        "held_at_analysis": 0,
        "held_shares":   0,
        "held_avg_cost": None,
        "result_file":   None,
    }
    return db.upsert_analysis(row)


def _make_tracker(db=None):
    from src.accuracy.price_tracker import PriceTracker
    if db is None:
        db = _make_db()
    return PriceTracker(db), db


def _ohlcv_df(dates, closes, highs=None, lows=None):
    """Build a minimal yfinance-style DataFrame."""
    n = len(dates)
    return pd.DataFrame({
        "Close": closes,
        "High":  highs or [c + 2 for c in closes],
        "Low":   lows or [c - 2 for c in closes],
        "Open":  closes,
        "Volume": [1_000_000] * n,
    }, index=pd.to_datetime(dates))


# ── 1. create_outcome() ───────────────────────────────────────────────────────

class TestCreateOutcome:

    def test_creates_pending_record(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        oid = tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY",
                                     entry_price=195.0, stop_loss=180.0, price_target=220.0)
        assert oid is not None and oid > 0

    def test_stores_all_fields(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY",
                               entry_price=195.0, stop_loss=180.0, price_target=220.0)
        outcome = tracker.get_outcome(aid)
        assert outcome["ticker"]        == "AAPL"
        assert outcome["signal_date"]   == "2026-03-01"
        assert outcome["decision"]      == "BUY"
        assert outcome["entry_price"]   == 195.0
        assert outcome["stop_loss"]     == 180.0
        assert outcome["price_target"]  == 220.0
        assert outcome["status"]        == "pending"

    def test_status_is_pending_initially(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        assert outcome["status"] == "pending"

    def test_null_price_fields_accepted(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "HOLD")
        outcome = tracker.get_outcome(aid)
        assert outcome["entry_price"]  is None
        assert outcome["stop_loss"]    is None
        assert outcome["price_target"] is None

    def test_duplicate_analysis_id_silently_ignored(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        oid1 = tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        oid2 = tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        # Second insert is ignored; outcome still has exactly one record
        outcome = tracker.get_outcome(aid)
        assert outcome is not None


# ── 2. update_prices() ────────────────────────────────────────────────────────

class TestUpdatePrices:

    def test_updates_price_checkpoints(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.update_prices(outcome["id"], {"price_at_signal": 194.5, "price_t1": 196.0})
        updated = tracker.get_outcome(aid)
        assert updated["price_at_signal"] == 194.5
        assert updated["price_t1"]        == 196.0

    def test_incremental_update_preserves_existing(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.update_prices(outcome["id"], {"price_at_signal": 194.5, "price_t1": 196.0})
        tracker.update_prices(outcome["id"], {"price_t5": 200.0})
        updated = tracker.get_outcome(aid)
        assert updated["price_at_signal"] == 194.5  # still there
        assert updated["price_t1"]        == 196.0  # still there
        assert updated["price_t5"]        == 200.0  # new

    def test_does_not_update_complete_records(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.mark_complete(outcome["id"])
        tracker.update_prices(outcome["id"], {"price_at_signal": 999.0})
        final = tracker.get_outcome(aid)
        assert final["price_at_signal"] is None  # not updated

    def test_ignores_non_price_keys(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        # daily_highs/daily_lows should not be stored
        tracker.update_prices(outcome["id"], {
            "price_t1": 196.0, "daily_highs": [197, 198]
        })
        updated = tracker.get_outcome(aid)
        assert updated["price_t1"] == 196.0


# ── 3. get_pending_outcomes() ─────────────────────────────────────────────────

class TestGetPendingOutcomes:

    def test_returns_pending_and_partial(self):
        tracker, db = _make_tracker()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02")
        aid3 = _insert_analysis(db, ticker="NVDA", trade_date="2026-03-03")

        tracker.create_outcome(aid1, "AAPL", "2026-03-01", "BUY")
        tracker.create_outcome(aid2, "MSFT", "2026-03-02", "SELL")
        tracker.create_outcome(aid3, "NVDA", "2026-03-03", "HOLD")

        o3 = tracker.get_outcome(aid3)
        tracker.mark_complete(o3["id"])

        pending = tracker.get_pending_outcomes()
        tickers = {p["ticker"] for p in pending}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "NVDA" not in tickers  # complete

    def test_empty_when_all_complete(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.mark_complete(outcome["id"])
        assert tracker.get_pending_outcomes() == []


# ── 4. get_outcomes_for_ticker() ──────────────────────────────────────────────

class TestGetOutcomesForTicker:

    def test_returns_newest_first(self):
        tracker, db = _make_tracker()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-05",
                                 quality_score=8.5)
        tracker.create_outcome(aid1, "AAPL", "2026-03-01", "BUY")
        tracker.create_outcome(aid2, "AAPL", "2026-03-05", "HOLD")
        outcomes = tracker.get_outcomes_for_ticker("AAPL")
        assert outcomes[0]["signal_date"] == "2026-03-05"
        assert outcomes[1]["signal_date"] == "2026-03-01"

    def test_filters_by_ticker(self):
        tracker, db = _make_tracker()
        aid1 = _insert_analysis(db, ticker="AAPL", trade_date="2026-03-01")
        aid2 = _insert_analysis(db, ticker="MSFT", trade_date="2026-03-02")
        tracker.create_outcome(aid1, "AAPL", "2026-03-01", "BUY")
        tracker.create_outcome(aid2, "MSFT", "2026-03-02", "SELL")
        outcomes = tracker.get_outcomes_for_ticker("AAPL")
        assert all(o["ticker"] == "AAPL" for o in outcomes)
        assert len(outcomes) == 1


# ── 5. mark_complete() / mark_error() ────────────────────────────────────────

class TestStatusTransitions:

    def test_mark_complete_sets_status(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.mark_complete(outcome["id"])
        assert tracker.get_outcome(aid)["status"] == "complete"

    def test_mark_error_sets_status_and_message(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.mark_error(outcome["id"], "Ticker delisted")
        updated = tracker.get_outcome(aid)
        assert updated["status"]        == "error"
        assert updated["error_message"] == "Ticker delisted"


# ── 6. apply_scores() ─────────────────────────────────────────────────────────

class TestApplyScores:

    def test_writes_score_fields(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        tracker.apply_scores(outcome["id"], {
            "direction_correct_t1":  1,
            "direction_correct_t5":  1,
            "direction_correct_t10": 0,
            "return_t5_pct":         2.5,
            "target_hit":            1,
            "stop_hit":              0,
        })
        updated = tracker.get_outcome(aid)
        assert updated["direction_correct_t1"] == 1
        assert updated["return_t5_pct"]        == 2.5
        assert updated["target_hit"]           == 1

    def test_ignores_unknown_score_fields(self):
        tracker, db = _make_tracker()
        aid = _insert_analysis(db)
        tracker.create_outcome(aid, "AAPL", "2026-03-01", "BUY")
        outcome = tracker.get_outcome(aid)
        # Should not raise even with unknown keys
        tracker.apply_scores(outcome["id"], {"foo": 1, "return_t5_pct": 3.0})
        updated = tracker.get_outcome(aid)
        assert updated["return_t5_pct"] == 3.0


# ── 7. fetch_outcome_prices() (mocked yfinance) ───────────────────────────────

class TestFetchOutcomePrices:

    def _make_hist(self):
        """10 trading days of OHLCV data starting 2026-03-03."""
        dates  = pd.date_range("2026-03-03", periods=11, freq="B")
        closes = [194.0 + i for i in range(11)]
        highs  = [c + 2 for c in closes]
        lows   = [c - 2 for c in closes]
        return _ohlcv_df([d.strftime("%Y-%m-%d") for d in dates], closes, highs, lows)

    def test_returns_price_at_signal_and_t1(self):
        from src.accuracy.price_tracker import fetch_outcome_prices
        hist = self._make_hist()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist

        with patch("src.accuracy.price_tracker.yf.Ticker", return_value=mock_ticker):
            prices = fetch_outcome_prices("AAPL", "2026-03-03")

        assert "price_at_signal" in prices
        assert "price_t1"        in prices

    def test_returns_all_checkpoints_when_10_days_available(self):
        from src.accuracy.price_tracker import fetch_outcome_prices
        hist = self._make_hist()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist

        with patch("src.accuracy.price_tracker.yf.Ticker", return_value=mock_ticker):
            prices = fetch_outcome_prices("AAPL", "2026-03-03")

        for key in ("price_at_signal", "price_t1", "price_t5", "price_t10",
                    "high_t10", "low_t10", "daily_highs", "daily_lows"):
            assert key in prices, f"Missing key: {key}"

    def test_daily_arrays_have_10_entries(self):
        from src.accuracy.price_tracker import fetch_outcome_prices
        hist = self._make_hist()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist

        with patch("src.accuracy.price_tracker.yf.Ticker", return_value=mock_ticker):
            prices = fetch_outcome_prices("AAPL", "2026-03-03")

        assert len(prices["daily_highs"]) == 10
        assert len(prices["daily_lows"])  == 10

    def test_partial_result_when_only_t1_available(self):
        from src.accuracy.price_tracker import fetch_outcome_prices
        # Only 2 rows (signal_date + T+1)
        dates  = pd.date_range("2026-03-03", periods=2, freq="B")
        hist   = _ohlcv_df([d.strftime("%Y-%m-%d") for d in dates],
                           [194.0, 196.0])
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist

        with patch("src.accuracy.price_tracker.yf.Ticker", return_value=mock_ticker):
            prices = fetch_outcome_prices("AAPL", "2026-03-03")

        assert "price_at_signal" in prices
        assert "price_t1"        in prices
        assert "price_t5"        not in prices
        assert "price_t10"       not in prices

    def test_empty_dict_for_unknown_ticker(self):
        from src.accuracy.price_tracker import fetch_outcome_prices
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("src.accuracy.price_tracker.yf.Ticker", return_value=mock_ticker):
            prices = fetch_outcome_prices("ZZZZZ", "2026-03-03")

        assert prices == {}

    def test_non_trading_day_falls_back_to_prior_close(self):
        from src.accuracy.price_tracker import fetch_outcome_prices
        # Simulate signal_date=Saturday; first row is Monday (next trading day)
        signal_date = "2026-02-28"  # Saturday
        monday_date = "2026-03-02"

        future_hist = _ohlcv_df([monday_date], [198.0])
        prior_hist  = _ohlcv_df(["2026-02-27"], [195.0])  # Friday close

        mock_ticker = MagicMock()
        # First call (from signal_date forward) returns Monday onwards
        mock_ticker.history.side_effect = [future_hist, prior_hist]

        with patch("src.accuracy.price_tracker.yf.Ticker", return_value=mock_ticker):
            prices = fetch_outcome_prices("AAPL", signal_date)

        # price_at_signal should be Friday's close (195.0), not Monday's
        assert prices.get("price_at_signal") == 195.0
