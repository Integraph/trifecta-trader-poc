"""Tests for the trade executor."""

import pytest
from unittest.mock import MagicMock, patch
from src.execution.trade_params import TradeParams
from src.execution.position_manager import OrderCalculation


class TestTradeExecutor:

    def test_rejected_order_not_submitted(self, tmp_path):
        """Rejected orders should not be submitted to Alpaca."""
        with patch("src.execution.executor.TradingClient"):
            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=0,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=0, position_pct_of_portfolio=0,
            risk_per_share=12.0, total_risk=0, portfolio_risk_pct=0,
            approved=False, rejection_reasons=["Quality too low"],
        )
        params = TradeParams(ticker="AAPL", decision="BUY", quality_score=6.0)

        result = executor.execute(order, params)

        assert result["action"] == "REJECTED"
        assert "Quality too low" in result["rejection_reasons"]
        # Verify no order was submitted
        executor._client.submit_order.assert_not_called()

    def test_approved_bracket_order_submitted(self, tmp_path):
        """Approved order with target should submit bracket order."""
        with patch("src.execution.executor.TradingClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_order = MagicMock()
            mock_order.id = "test-order-123"
            mock_order.status = "new"
            mock_instance.submit_order.return_value = mock_order

            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=18,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=4860.0, position_pct_of_portfolio=4.86,
            risk_per_share=12.0, total_risk=216.0, portfolio_risk_pct=0.216,
            approved=True, rejection_reasons=[],
        )
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            stop_loss=258.0, price_target=295.0,
        )

        result = executor.execute(order, params)

        assert result["action"] == "EXECUTED"
        assert result["alpaca_order_id"] == "test-order-123"
        executor._client.submit_order.assert_called_once()

    def test_paper_only_hardcoded(self, tmp_path):
        """Verify paper=True is hardcoded."""
        with patch("src.execution.executor.TradingClient") as MockClient:
            from src.execution.executor import TradeExecutor
            TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )
            # Verify paper=True was passed
            MockClient.assert_called_once_with(
                api_key="test", secret_key="test", paper=True
            )

    def test_audit_file_created(self, tmp_path):
        """Every execution should create an audit file."""
        with patch("src.execution.executor.TradingClient"):
            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=0,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=0, position_pct_of_portfolio=0,
            risk_per_share=12.0, total_risk=0, portfolio_risk_pct=0,
            approved=False, rejection_reasons=["Test rejection"],
        )
        params = TradeParams(ticker="AAPL", decision="BUY", quality_score=6.0)

        executor.execute(order, params)

        audit_files = list((tmp_path / "audit").glob("*.json"))
        assert len(audit_files) == 1

    def test_failed_order_logs_error(self, tmp_path):
        """API failures should be captured in audit entry."""
        with patch("src.execution.executor.TradingClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.submit_order.side_effect = Exception("API connection failed")

            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=18,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=4860.0, position_pct_of_portfolio=4.86,
            risk_per_share=12.0, total_risk=216.0, portfolio_risk_pct=0.216,
            approved=True, rejection_reasons=[],
        )
        params = TradeParams(ticker="AAPL", decision="BUY", quality_score=9.5)

        result = executor.execute(order, params)

        assert result["action"] == "FAILED"
        assert "API connection failed" in result["error"]
        # Audit file should still be created
        audit_files = list((tmp_path / "audit").glob("*.json"))
        assert len(audit_files) == 1

    def test_audit_entry_contains_key_fields(self, tmp_path):
        """Audit entry should contain all required fields."""
        with patch("src.execution.executor.TradingClient"):
            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="TSLA", side="sell", qty=5,
            entry_price=300.0, stop_loss=315.0, take_profit=260.0,
            position_value=1500.0, position_pct_of_portfolio=1.5,
            risk_per_share=15.0, total_risk=75.0, portfolio_risk_pct=0.075,
            approved=False, rejection_reasons=["Test"],
        )
        params = TradeParams(
            ticker="TSLA", decision="SELL", quality_score=9.5,
            risk_reward_ratio=2.0,
        )

        result = executor.execute(order, params)

        required_fields = [
            "timestamp", "ticker", "decision", "quality_score", "side",
            "qty", "entry_price", "stop_loss", "approved",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
