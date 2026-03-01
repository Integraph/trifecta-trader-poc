"""Tests for position management and order calculation."""

import pytest
from unittest.mock import MagicMock, patch
from src.execution.position_manager import (
    PositionManager, AccountState, Position, OrderCalculation,
    MAX_POSITION_PCT, MIN_QUALITY_SCORE, MAX_PORTFOLIO_RISK_PCT,
)
from src.execution.trade_params import TradeParams


def _mock_account(portfolio_value=100000, buying_power=50000, cash=50000):
    """Create a mock Alpaca account."""
    mock = MagicMock()
    mock.buying_power = str(buying_power)
    mock.portfolio_value = str(portfolio_value)
    mock.cash = str(cash)
    mock.equity = str(portfolio_value)
    return mock


def _mock_position(symbol="AAPL", qty=10, current_price=270.0):
    """Create a mock Alpaca position."""
    mock = MagicMock()
    mock.symbol = symbol
    mock.qty = str(qty)
    mock.current_price = str(current_price)
    mock.market_value = str(qty * current_price)
    mock.cost_basis = str(qty * 265.0)
    mock.unrealized_pl = str(qty * 5.0)
    mock.unrealized_plpc = str(0.0189)
    return mock


class TestPositionManager:

    def _make_pm(self, portfolio_value=100000, buying_power=50000, positions=None):
        """Create a PositionManager with mocked Alpaca client."""
        mock_client = MagicMock()
        mock_client.get_account.return_value = _mock_account(portfolio_value, buying_power)

        if positions:
            mock_client.get_all_positions.return_value = positions
            mock_client.get_open_position.side_effect = lambda t: next(
                (p for p in positions if p.symbol == t), None
            )
        else:
            mock_client.get_all_positions.return_value = []
            mock_client.get_open_position.side_effect = Exception("No position")

        return PositionManager(mock_client)

    def test_buy_order_calculation(self):
        """Standard BUY order should calculate quantity from position size."""
        pm = self._make_pm(portfolio_value=100000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, price_target=295.0,
            position_pct=5.0, risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        assert order.side == "buy"
        assert order.qty > 0
        assert order.approved
        assert order.stop_loss == 258.0
        assert order.take_profit == 295.0
        # 5% of $100K = $5000; $5000 / $270 = 18 shares
        assert order.qty == 18

    def test_sell_order_with_position(self):
        """SELL should sell existing shares."""
        positions = [_mock_position("AAPL", qty=20, current_price=270.0)]
        pm = self._make_pm(portfolio_value=100000, positions=positions)
        params = TradeParams(
            ticker="AAPL", decision="SELL", quality_score=9.5,
            entry_price=270.0, stop_loss=280.0, position_pct=5.0,
            risk_pct=3.7, reward_pct=10.0, risk_reward_ratio=2.7,
        )
        order = pm.calculate_order(params)

        assert order.side == "sell"
        assert order.qty <= 20  # Can't sell more than we have
        assert order.approved

    def test_sell_without_position_rejected(self):
        """SELL with no position should be rejected."""
        pm = self._make_pm()
        params = TradeParams(
            ticker="AAPL", decision="SELL", quality_score=9.5,
            entry_price=270.0, stop_loss=280.0,
        )
        order = pm.calculate_order(params)

        assert not order.approved
        assert any("No position" in r for r in order.rejection_reasons)

    def test_hold_not_actionable(self):
        """HOLD should produce no order."""
        pm = self._make_pm()
        params = TradeParams(ticker="AAPL", decision="HOLD", quality_score=10.0)
        order = pm.calculate_order(params)

        assert not order.approved
        assert order.qty == 0

    def test_low_quality_rejected(self):
        """Low quality score should reject the order."""
        pm = self._make_pm()
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=6.0,
            entry_price=270.0, stop_loss=258.0,
        )
        order = pm.calculate_order(params)

        assert not order.approved
        assert any("Quality score" in r for r in order.rejection_reasons)

    def test_no_stop_loss_rejected(self):
        """Missing stop-loss should reject the order."""
        pm = self._make_pm()
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=None,
        )
        order = pm.calculate_order(params)

        assert not order.approved
        assert any("stop-loss" in r.lower() for r in order.rejection_reasons)

    def test_position_size_capped(self):
        """Position should be capped at MAX_POSITION_PCT."""
        pm = self._make_pm(portfolio_value=100000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0,
            position_pct=25.0,  # Exceeds MAX_POSITION_PCT (15%)
            risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        # Should cap at 15%, not use 25%
        max_value = 100000 * (MAX_POSITION_PCT / 100)
        max_qty = int(max_value / 270.0)
        assert order.qty <= max_qty

    def test_insufficient_buying_power(self):
        """Should reduce qty if buying power is insufficient."""
        pm = self._make_pm(portfolio_value=100000, buying_power=1000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, position_pct=5.0,
            risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        # $1000 buying power / $270 = 3 shares max
        assert order.qty <= 3

    def test_existing_position_reduces_buy(self):
        """Existing position should reduce buy quantity."""
        positions = [_mock_position("AAPL", qty=10, current_price=270.0)]
        pm = self._make_pm(portfolio_value=100000, positions=positions)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, position_pct=5.0,
            risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        # Target: 5% of $100K = $5000. Already have 10 * $270 = $2700.
        # Remaining: $2300 / $270 = 8 shares
        assert order.qty == 8

    def test_get_account_state(self):
        """get_account_state should return correct AccountState."""
        pm = self._make_pm(portfolio_value=100000, buying_power=50000)
        state = pm.get_account_state()
        assert state.portfolio_value == 100000
        assert state.buying_power == 50000

    def test_order_calculation_to_dict(self):
        """to_dict should return serializable dict."""
        pm = self._make_pm(portfolio_value=100000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, price_target=295.0,
            position_pct=5.0, risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)
        d = order.to_dict()

        assert "ticker" in d
        assert "side" in d
        assert "approved" in d
        assert d["ticker"] == "AAPL"
