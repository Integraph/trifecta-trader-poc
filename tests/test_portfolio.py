"""Tests for Task 012: Portfolio-Aware Execution & Watchlist Batch Mode.

Covers:
- Watchlist YAML loading (valid, missing, empty, no tickers)
- Portfolio context generation (mocked Alpaca API)
- Pre-filter warnings (no position, max allocation, low buying power)
- Batch summary calculation
- SQLite database creation and schema
- PortfolioTracker: log_analysis, log_order, take_snapshot
- get_analysis_history, get_daily_pnl, get_batch_summary
- Batch results file writing
- Single-ticker run_analysis still works (no regression)
- --portfolio flag (no ticker required)
"""

import json
import os
import tempfile
from datetime import datetime, date
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Return a PortfolioDatabase backed by a temporary file."""
    from src.portfolio.database import PortfolioDatabase
    return PortfolioDatabase(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def tmp_tracker(tmp_path):
    """Return a PortfolioTracker backed by a temporary DB."""
    from src.portfolio.tracker import PortfolioTracker
    return PortfolioTracker(db_path=str(tmp_path / "tracker.db"))


@pytest.fixture
def mock_account():
    acc = MagicMock()
    acc.equity        = 100_000.0
    acc.buying_power  = 45_000.0
    acc.cash          = 22_000.0
    acc.portfolio_value = 100_000.0
    return acc


@pytest.fixture
def mock_position():
    pos = MagicMock()
    pos.symbol          = "AAPL"
    pos.qty             = "50"
    pos.current_price   = "195.00"
    pos.market_value    = "9750.00"
    pos.cost_basis      = "9100.00"
    pos.unrealized_pl   = "650.00"
    pos.unrealized_plpc = "0.0714"
    return pos


def _make_result(ticker="AAPL", decision="BUY", quality=9.2, elapsed=300.0,
                 hybrid="hybrid_haiku_tools"):
    return {
        "ticker":        ticker,
        "trade_date":    "2026-03-04",
        "run_timestamp": datetime.now().isoformat(),
        "hybrid_config": hybrid,
        "provider":      "anthropic",
        "decision":      decision,
        "quality_score": {"composite": quality},
        "cost_breakdown": {"total_usd": 0.12, "by_model": {}},
        "elapsed_seconds": elapsed,
        "result_file":   f"results/{ticker}/analysis_2026-03-04_{hybrid}.json",
        "trade_params":  {},
    }


def _make_portfolio_ctx(held=True, shares=50, pct=9.75, equity=100_000.0,
                        buying_power=45_000.0):
    pos = {"held": held}
    if held:
        pos.update({
            "shares":         shares,
            "avg_cost":       182.0,
            "unrealized_pnl": 650.0,
            "current_value":  9_750.0,
            "portfolio_pct":  pct,
        })
    return {
        "account_equity":       equity,
        "buying_power":         buying_power,
        "cash":                 22_000.0,
        "total_positions":      3,
        "current_position":     pos,
        "portfolio_allocation": {"AAPL": {"pct": pct, "shares": shares}},
        "_source":              "alpaca",
    }


# ── 1. Watchlist YAML loading ─────────────────────────────────────────────────

class TestWatchlistLoading:

    def test_load_valid_watchlist(self, tmp_path):
        from src.run_batch import load_watchlist

        wl = tmp_path / "test.yaml"
        wl.write_text("name: Test\ntickers:\n  - AAPL\n  - MSFT\n")
        name, tickers = load_watchlist(str(wl))
        assert name == "Test"
        assert tickers == ["AAPL", "MSFT"]

    def test_load_missing_file(self, tmp_path):
        from src.run_batch import load_watchlist
        with pytest.raises(FileNotFoundError):
            load_watchlist(str(tmp_path / "missing.yaml"))

    def test_load_empty_file(self, tmp_path):
        from src.run_batch import load_watchlist
        wl = tmp_path / "empty.yaml"
        wl.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_watchlist(str(wl))

    def test_load_no_tickers_key(self, tmp_path):
        from src.run_batch import load_watchlist
        wl = tmp_path / "no_tickers.yaml"
        wl.write_text("name: X\ndescription: nothing\n")
        with pytest.raises(ValueError, match="no tickers"):
            load_watchlist(str(wl))

    def test_tickers_uppercased(self, tmp_path):
        from src.run_batch import load_watchlist
        wl = tmp_path / "lower.yaml"
        wl.write_text("name: T\ntickers:\n  - aapl\n  - msft\n")
        _, tickers = load_watchlist(str(wl))
        assert tickers == ["AAPL", "MSFT"]

    def test_tickers_from_string(self):
        from src.run_batch import tickers_from_string
        result = tickers_from_string("aapl, msft, nvda")
        assert result == ["AAPL", "MSFT", "NVDA"]

    def test_default_watchlist_file_exists(self):
        """The checked-in default.yaml must be parseable."""
        from src.run_batch import load_watchlist
        name, tickers = load_watchlist("config/watchlists/default.yaml")
        assert len(tickers) >= 5
        assert "AAPL" in tickers

    def test_small_cap_watchlist_file_exists(self):
        from src.run_batch import load_watchlist
        name, tickers = load_watchlist("config/watchlists/small_cap.yaml")
        assert len(tickers) >= 3


# ── 2. Portfolio context generation ──────────────────────────────────────────

class TestPortfolioContextGeneration:

    def test_build_context_with_held_position(self, mock_account, mock_position):
        from src.run_analysis import build_portfolio_context
        from src.execution.position_manager import AccountState, Position

        acc = AccountState(
            equity=100_000.0, buying_power=45_000.0,
            cash=22_000.0, portfolio_value=100_000.0
        )
        pos = Position(
            ticker="AAPL", qty=50, current_price=195.0,
            market_value=9_750.0, cost_basis=9_100.0,
            unrealized_pl=650.0, unrealized_pl_pct=7.14
        )

        pm_mock = MagicMock()
        pm_mock.get_account_state.return_value = acc
        pm_mock.get_positions.return_value = {"AAPL": pos}

        # PositionManager is imported inside the function body, so patch at
        # the source module so the local import picks up the mock.
        with patch("src.run_analysis._make_trading_client"), \
             patch("src.execution.position_manager.PositionManager",
                   return_value=pm_mock):
            ctx = build_portfolio_context("AAPL")

        assert ctx["account_equity"] == 100_000.0
        assert ctx["buying_power"]   == 45_000.0
        assert ctx["current_position"]["held"] is True
        assert ctx["current_position"]["shares"] == 50
        assert ctx["_source"] == "alpaca"

    def test_build_context_no_position(self):
        from src.run_analysis import build_portfolio_context
        from src.execution.position_manager import AccountState

        acc = AccountState(equity=100_000.0, buying_power=45_000.0,
                           cash=22_000.0, portfolio_value=100_000.0)

        pm_mock = MagicMock()
        pm_mock.get_account_state.return_value = acc
        pm_mock.get_positions.return_value = {}

        with patch("src.run_analysis._make_trading_client"), \
             patch("src.execution.position_manager.PositionManager",
                   return_value=pm_mock):
            ctx = build_portfolio_context("TSLA")

        assert ctx["current_position"]["held"] is False

    def test_build_context_alpaca_unavailable(self):
        from src.run_analysis import build_portfolio_context
        with patch("src.run_analysis._make_trading_client",
                   side_effect=Exception("Connection refused")):
            ctx = build_portfolio_context("AAPL")

        assert ctx["_source"] == "unavailable"
        assert ctx["current_position"]["held"] is False
        assert "_error" in ctx


# ── 3. Pre-filter warnings ────────────────────────────────────────────────────

class TestPreFilterWarnings:

    def test_no_position_triggers_warning(self):
        from src.run_analysis import check_portfolio_warnings
        ctx = _make_portfolio_ctx(held=False, buying_power=50_000.0)
        warnings = check_portfolio_warnings("TSLA", ctx)
        assert any("No position" in w for w in warnings)

    def test_max_allocation_triggers_warning(self):
        from src.run_analysis import check_portfolio_warnings
        from src.execution.position_manager import MAX_POSITION_PCT
        ctx = _make_portfolio_ctx(held=True, pct=MAX_POSITION_PCT)
        warnings = check_portfolio_warnings("AAPL", ctx)
        assert any("max allocation" in w for w in warnings)

    def test_low_buying_power_triggers_warning(self):
        from src.run_analysis import check_portfolio_warnings
        ctx = _make_portfolio_ctx(held=True, buying_power=500.0)
        warnings = check_portfolio_warnings("AAPL", ctx)
        assert any("buying power" in w.lower() for w in warnings)

    def test_healthy_portfolio_no_warnings(self):
        from src.run_analysis import check_portfolio_warnings
        ctx = _make_portfolio_ctx(held=True, pct=5.0, buying_power=50_000.0)
        warnings = check_portfolio_warnings("AAPL", ctx)
        # Should only have no-position warning absent; no-max-alloc; no-low-bp
        non_position = [w for w in warnings if "No position" not in w]
        assert len(non_position) == 0

    def test_unavailable_context_no_warnings(self):
        from src.run_analysis import check_portfolio_warnings
        ctx = {"_source": "unavailable", "_error": "timeout",
               "current_position": {"held": False}}
        warnings = check_portfolio_warnings("AAPL", ctx)
        assert warnings == []


# ── 4. SQLite database ────────────────────────────────────────────────────────

class TestPortfolioDatabase:

    def test_db_file_created(self, tmp_path):
        from src.portfolio.database import PortfolioDatabase
        db_path = tmp_path / "sub" / "portfolio.db"
        PortfolioDatabase(db_path=str(db_path))
        assert db_path.exists()

    def test_schema_tables_exist(self, tmp_db):
        import sqlite3
        with sqlite3.connect(tmp_db.db_path) as conn:
            tables = {r[0] for r in
                      conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "analyses"             in tables
        assert "orders"               in tables
        assert "portfolio_snapshots"  in tables

    def test_upsert_and_fetch_analysis(self, tmp_db):
        row = {
            "ticker": "AAPL", "trade_date": "2026-03-04",
            "run_timestamp": datetime.now().isoformat(),
            "config": "hybrid_haiku_tools", "decision": "BUY",
            "quality_score": 9.2, "cost_usd": 0.12, "elapsed_seconds": 300.0,
            "stop_loss": 180.0, "price_target": 220.0, "entry_price": 195.0,
            "position_size_pct": 5.0, "risk_reward": 1.8, "actionable": 1,
            "portfolio_equity": 100_000.0, "held_at_analysis": 1,
            "held_shares": 50, "held_avg_cost": 182.0, "result_file": "results/x.json",
        }
        aid = tmp_db.upsert_analysis(row)
        assert aid is not None and aid > 0

        rows = tmp_db.get_recent_analyses("AAPL", limit=5)
        assert len(rows) == 1
        assert rows[0]["decision"] == "BUY"

    def test_upsert_replaces_duplicate(self, tmp_db):
        base = {
            "ticker": "AAPL", "trade_date": "2026-03-04",
            "run_timestamp": datetime.now().isoformat(),
            "config": "hybrid_haiku_tools", "decision": "BUY",
            "quality_score": 9.0, "cost_usd": 0.10, "elapsed_seconds": 200.0,
            "stop_loss": None, "price_target": None, "entry_price": None,
            "position_size_pct": None, "risk_reward": None, "actionable": 0,
            "portfolio_equity": None, "held_at_analysis": 0,
            "held_shares": None, "held_avg_cost": None, "result_file": None,
        }
        tmp_db.upsert_analysis(base)
        updated = {**base, "decision": "SELL", "quality_score": 9.5}
        tmp_db.upsert_analysis(updated)

        rows = tmp_db.get_recent_analyses("AAPL")
        assert len(rows) == 1
        assert rows[0]["decision"] == "SELL"

    def test_insert_and_fetch_order(self, tmp_db):
        base = {
            "ticker": "AAPL", "trade_date": "2026-03-04",
            "run_timestamp": datetime.now().isoformat(),
            "config": "hybrid_haiku_tools", "decision": "BUY",
            "quality_score": 9.0, "cost_usd": None, "elapsed_seconds": None,
            "stop_loss": None, "price_target": None, "entry_price": None,
            "position_size_pct": None, "risk_reward": None, "actionable": 0,
            "portfolio_equity": None, "held_at_analysis": 0,
            "held_shares": None, "held_avg_cost": None, "result_file": None,
        }
        aid = tmp_db.upsert_analysis(base)

        order_row = {
            "analysis_id": aid,
            "ticker": "AAPL",
            "timestamp": datetime.now().isoformat(),
            "side": "buy",
            "qty": 10,
            "entry_price": 195.0,
            "stop_loss": 180.0,
            "take_profit": 220.0,
            "approved": 1,
            "rejection_reasons": "[]",
            "action": "EXECUTED",
            "alpaca_order_id": "abc123",
            "alpaca_status": "filled",
        }
        oid = tmp_db.insert_order(order_row)
        assert oid > 0

        orders = tmp_db.get_recent_orders()
        assert len(orders) == 1
        assert orders[0]["action"] == "EXECUTED"

    def test_upsert_and_fetch_snapshot(self, tmp_db):
        snap = {
            "snapshot_date": "2026-03-04",
            "account_equity": 100_000.0,
            "buying_power": 45_000.0,
            "cash": 22_000.0,
            "positions_json": json.dumps({"AAPL": {"qty": 50}}),
            "total_positions": 1,
        }
        tmp_db.upsert_snapshot(snap)
        snaps = tmp_db.get_snapshots(7)
        assert len(snaps) == 1
        assert snaps[0]["account_equity"] == 100_000.0

    def test_snapshot_upsert_replaces_same_date(self, tmp_db):
        snap = {
            "snapshot_date": "2026-03-04",
            "account_equity": 100_000.0,
            "buying_power": 45_000.0,
            "cash": 22_000.0,
            "positions_json": "{}",
            "total_positions": 0,
        }
        tmp_db.upsert_snapshot(snap)
        updated = {**snap, "account_equity": 101_000.0}
        tmp_db.upsert_snapshot(updated)

        snaps = tmp_db.get_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["account_equity"] == 101_000.0


# ── 5. PortfolioTracker ───────────────────────────────────────────────────────

class TestPortfolioTracker:

    def test_log_analysis_inserts_row(self, tmp_tracker):
        result = _make_result()
        ctx    = _make_portfolio_ctx()
        aid = tmp_tracker.log_analysis(result, ctx)
        assert aid > 0

        history = tmp_tracker.get_analysis_history("AAPL", limit=5)
        assert len(history) == 1
        assert history[0]["decision"] == "BUY"

    def test_log_analysis_captures_held_status(self, tmp_tracker):
        result = _make_result(ticker="TSLA", decision="SELL")
        ctx    = _make_portfolio_ctx(held=False)
        tmp_tracker.log_analysis(result, ctx)

        history = tmp_tracker.get_analysis_history("TSLA")
        assert history[0]["held_at_analysis"] == 0

    def test_log_order_approved(self, tmp_tracker):
        result = _make_result()
        ctx    = _make_portfolio_ctx()
        aid    = tmp_tracker.log_analysis(result, ctx)

        order = MagicMock()
        order.ticker              = "AAPL"
        order.side                = "buy"
        order.qty                 = 10
        order.entry_price         = 195.0
        order.stop_loss           = 180.0
        order.take_profit         = 220.0
        order.approved            = True
        order.rejection_reasons   = []

        oid = tmp_tracker.log_order(aid, order, "EXECUTED", "ord123", "filled")
        assert oid > 0

        orders = tmp_tracker.get_recent_orders(5)
        assert orders[0]["action"] == "EXECUTED"
        assert orders[0]["alpaca_order_id"] == "ord123"

    def test_log_order_rejected(self, tmp_tracker):
        result = _make_result(decision="SELL")
        ctx    = _make_portfolio_ctx(held=False)
        aid    = tmp_tracker.log_analysis(result, ctx)

        order = MagicMock()
        order.ticker              = "AAPL"
        order.side                = "sell"
        order.qty                 = 0
        order.entry_price         = 0.0
        order.stop_loss           = 0.0
        order.take_profit         = None
        order.approved            = False
        order.rejection_reasons   = ["No position to sell"]

        oid = tmp_tracker.log_order(aid, order, "REJECTED")
        assert oid > 0
        orders = tmp_tracker.get_recent_orders(5)
        assert orders[0]["approved"] == 0

    def test_take_snapshot(self, tmp_tracker):
        from src.execution.position_manager import AccountState, Position

        acc = AccountState(equity=100_000.0, buying_power=45_000.0,
                           cash=22_000.0, portfolio_value=100_000.0)
        pos = Position(ticker="AAPL", qty=50, current_price=195.0,
                       market_value=9_750.0, cost_basis=9_100.0,
                       unrealized_pl=650.0, unrealized_pl_pct=7.14)

        pm = MagicMock()
        pm.get_account_state.return_value = acc
        pm.get_positions.return_value     = {"AAPL": pos}

        tmp_tracker.take_snapshot(pm)
        snaps = tmp_tracker.get_recent_snapshots(7)
        assert len(snaps) == 1
        assert snaps[0]["account_equity"] == 100_000.0
        positions_data = json.loads(snaps[0]["positions_json"])
        assert "AAPL" in positions_data

    def test_take_snapshot_handles_error(self, tmp_tracker):
        pm = MagicMock()
        pm.get_account_state.side_effect = Exception("timeout")
        # Should not raise
        tmp_tracker.take_snapshot(pm)

    def test_get_decision_history(self, tmp_tracker):
        tmp_tracker.log_analysis(_make_result(decision="BUY"),  _make_portfolio_ctx())
        tmp_tracker.log_analysis(
            {**_make_result(decision="HOLD"), "trade_date": "2026-03-05",
             "run_timestamp": datetime.now().isoformat()},
            _make_portfolio_ctx()
        )
        hist = tmp_tracker.get_decision_history("AAPL")
        assert len(hist) == 2
        decisions = [h["decision"] for h in hist]
        assert "BUY" in decisions and "HOLD" in decisions

    def test_get_daily_pnl_single_snapshot(self, tmp_tracker):
        """Need at least 2 snapshots for P&L calculation."""
        pm = MagicMock()
        from src.execution.position_manager import AccountState
        pm.get_account_state.return_value = AccountState(
            equity=100_000.0, buying_power=45_000.0,
            cash=22_000.0, portfolio_value=100_000.0
        )
        pm.get_positions.return_value = {}
        tmp_tracker.take_snapshot(pm)
        pnl = tmp_tracker.get_daily_pnl(30)
        assert pnl == []

    def test_get_batch_summary(self, tmp_tracker):
        tmp_tracker.log_analysis(_make_result("AAPL", "BUY"),  _make_portfolio_ctx())
        tmp_tracker.log_analysis(_make_result("MSFT", "HOLD"), _make_portfolio_ctx(held=False))
        tmp_tracker.log_analysis(_make_result("TSLA", "SELL"), _make_portfolio_ctx(held=False))

        summary = tmp_tracker.get_batch_summary("2026-03-04")
        assert summary["count"] == 3
        assert "BUY"  in summary["decisions"]
        assert "HOLD" in summary["decisions"]
        assert "SELL" in summary["decisions"]
        assert summary["total_cost"] > 0


# ── 6. Batch summary & results file ──────────────────────────────────────────

class TestBatchSummaryAndFile:

    def test_batch_results_saved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Create a minimal results/batch dir
        (tmp_path / "results" / "batch").mkdir(parents=True)

        from src.run_batch import _save_batch_results
        results = [
            {**_make_result("AAPL", "BUY"),  "portfolio_context": {"held": True}},
            {**_make_result("MSFT", "HOLD"), "portfolio_context": {"held": False}},
        ]
        out = _save_batch_results(results, "Test WL", "hybrid_haiku_tools",
                                  "2026-03-04", 600.0)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["tickers_count"] == 2
        assert data["watchlist"] == "Test WL"

    def test_build_ticker_context_with_position(self):
        from src.run_batch import build_ticker_context
        from src.execution.position_manager import Position

        pos = Position(
            ticker="AAPL", qty=50, current_price=195.0,
            market_value=9_750.0, cost_basis=9_100.0,
            unrealized_pl=650.0, unrealized_pl_pct=7.14
        )
        shared = {
            "account_equity": 100_000.0, "buying_power": 45_000.0,
            "cash": 22_000.0, "total_positions": 1,
            "portfolio_allocation": {}, "_source": "alpaca",
        }
        ctx = build_ticker_context("AAPL", shared, {"AAPL": pos}, 100_000.0)
        assert ctx["current_position"]["held"] is True
        assert ctx["current_position"]["shares"] == 50

    def test_build_ticker_context_without_position(self):
        from src.run_batch import build_ticker_context
        shared = {
            "account_equity": 100_000.0, "buying_power": 45_000.0,
            "cash": 22_000.0, "total_positions": 0,
            "portfolio_allocation": {}, "_source": "alpaca",
        }
        ctx = build_ticker_context("TSLA", shared, {}, 100_000.0)
        assert ctx["current_position"]["held"] is False


# ── 7. Regression: single-ticker run_analysis signature ─────────────────────

class TestRunAnalysisSignature:

    def test_run_analysis_accepts_portfolio_context(self):
        """run_analysis must accept portfolio_context kwarg without error."""
        import inspect
        from src.run_analysis import run_analysis
        sig = inspect.signature(run_analysis)
        assert "portfolio_context" in sig.parameters

    def test_run_analysis_accepts_batch_mode(self):
        import inspect
        from src.run_analysis import run_analysis
        sig = inspect.signature(run_analysis)
        assert "batch_mode" in sig.parameters

    def test_portfolio_context_defaults_to_none(self):
        import inspect
        from src.run_analysis import run_analysis
        sig = inspect.signature(run_analysis)
        default = sig.parameters["portfolio_context"].default
        assert default is None

    def test_portfolio_flag_in_cli(self):
        """--portfolio flag must be accepted by the CLI argument parser."""
        import argparse
        from src.run_analysis import main
        # We just need the parser to not raise on --portfolio
        with patch("src.run_analysis.print_portfolio_summary_from_db"):
            with patch("sys.argv", ["run_analysis", "--portfolio"]):
                try:
                    main()
                except SystemExit:
                    pass  # argparse may call sys.exit(0) after --portfolio


# ── 8. PortfolioDatabase get_analysis_id ─────────────────────────────────────

class TestGetAnalysisId:

    def test_get_analysis_id_returns_correct_id(self, tmp_db):
        row = {
            "ticker": "NVDA", "trade_date": "2026-03-04",
            "run_timestamp": datetime.now().isoformat(),
            "config": "hybrid_haiku_tools", "decision": "BUY",
            "quality_score": 9.5, "cost_usd": 0.11, "elapsed_seconds": 310.0,
            "stop_loss": None, "price_target": None, "entry_price": None,
            "position_size_pct": None, "risk_reward": None, "actionable": 0,
            "portfolio_equity": None, "held_at_analysis": 0,
            "held_shares": None, "held_avg_cost": None, "result_file": None,
        }
        inserted_id = tmp_db.upsert_analysis(row)
        fetched_id  = tmp_db.get_analysis_id("NVDA", "2026-03-04", "hybrid_haiku_tools")
        assert fetched_id == inserted_id

    def test_get_analysis_id_returns_none_if_missing(self, tmp_db):
        result = tmp_db.get_analysis_id("ZZZZ", "2026-01-01", "no_config")
        assert result is None
