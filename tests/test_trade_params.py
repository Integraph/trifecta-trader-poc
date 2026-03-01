"""Tests for trade parameter extraction from pipeline output."""

import pytest
from src.execution.trade_params import extract_trade_params, TradeParams


class TestExtractTradeParams:
    """Test parameter extraction from various output formats."""

    def test_standard_format(self):
        """Standard format with clear stop-loss, target, sizing."""
        text = """
        Entry: $270.15. Stop-loss: $258.00 (-4.5% risk). Target: $295.00 (+9.2%).
        Risk/Reward: 2.04:1. Position size: 5% of portfolio.
        FINAL TRANSACTION PROPOSAL: BUY
        """
        params = extract_trade_params("AAPL", "BUY", 9.5, text, current_price=270.15)

        assert params.stop_loss == 258.00
        assert params.price_target == 295.00
        assert params.position_pct == 5.0
        assert params.is_actionable
        assert params.has_bracket_params

    def test_hybrid_output_format(self):
        """Format seen in actual hybrid_qwen_enhanced output."""
        text = """
        ## Risk Management
        - Stop-loss: $258 (-4.4%)
        - Price target: $295 (+9.3%)
        - Position sizing: 5% portfolio weight
        FINAL TRANSACTION PROPOSAL: HOLD
        """
        params = extract_trade_params("AAPL", "HOLD", 10.0, text)

        assert params.stop_loss == 258.0
        assert params.price_target == 295.0
        assert params.position_pct == 5.0
        assert not params.is_actionable  # HOLD is not actionable

    def test_all_cloud_sell_format(self):
        """Format from the all-cloud SELL output."""
        text = """
        Sell 60-70% of position at $272-275.
        Hard stop at $254 (February low).
        First target: $254. Second target: $230.
        Position size: Maximum 2-3% of portfolio.
        FINAL TRANSACTION PROPOSAL: SELL
        """
        params = extract_trade_params("AAPL", "SELL", 10.0, text, current_price=272.0)

        assert params.stop_loss == 254.0
        assert params.is_actionable

    def test_hold_not_actionable(self):
        """HOLD decisions should not be actionable."""
        params = extract_trade_params("AAPL", "HOLD", 10.0, "HOLD recommendation")
        assert not params.is_actionable

    def test_low_quality_not_actionable(self):
        """Low quality score should prevent execution."""
        text = "Stop-loss: $258. Target: $295. FINAL TRANSACTION PROPOSAL: BUY"
        params = extract_trade_params("AAPL", "BUY", 6.0, text)
        assert not params.is_actionable  # Quality < 8.0

    def test_missing_stop_loss(self):
        """Without stop-loss, BUY should not be actionable."""
        text = "Target: $295. Position: 5%. FINAL TRANSACTION PROPOSAL: BUY"
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert not params.is_actionable  # No stop-loss

    def test_risk_metrics_calculated(self):
        """Risk/reward should be calculated from entry, stop, target."""
        text = "Stop-loss: $258. Target: $295. FINAL TRANSACTION PROPOSAL: BUY"
        params = extract_trade_params("AAPL", "BUY", 9.5, text, current_price=270.0)

        assert params.risk_pct is not None
        assert params.reward_pct is not None
        assert params.risk_reward_ratio is not None
        assert params.risk_pct == pytest.approx(4.44, abs=0.1)
        assert params.reward_pct == pytest.approx(9.26, abs=0.1)

    def test_trailing_stop_format(self):
        """Trailing stop-loss phrasing."""
        text = "Set trailing stop-loss at 12% below. Trailing stop at $240.50."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        # Should extract the dollar value, not the percentage
        assert params.stop_loss == 240.50

    def test_empty_text(self):
        """Empty text should return params with no extracted values."""
        params = extract_trade_params("AAPL", "BUY", 9.5, "")
        assert params.stop_loss is None
        assert params.price_target is None
        assert params.position_pct is None
        assert not params.is_actionable

    def test_confidence_extraction(self):
        """Should extract confidence level."""
        text = "Confidence Level: HIGH (8.5/10). Stop-loss: $258."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.confidence == "high"

    def test_has_bracket_params_both_present(self):
        """has_bracket_params true when stop and target are present."""
        text = "Stop-loss: $258. Target: $295."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.has_bracket_params

    def test_has_bracket_params_missing_target(self):
        """has_bracket_params false when target missing."""
        text = "Stop-loss: $258."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert not params.has_bracket_params

    def test_raw_text_excerpt_captured(self):
        """raw_text_excerpt should be last 500 chars."""
        long_text = "A" * 600 + "Stop-loss: $258."
        params = extract_trade_params("AAPL", "BUY", 9.5, long_text)
        assert len(params.raw_text_excerpt) == 500
        assert "Stop-loss" in params.raw_text_excerpt

    def test_allocation_pattern(self):
        """Should extract position from allocation pattern."""
        text = "Allocation: 4% of portfolio. Stop-loss: $260."
        params = extract_trade_params("AAPL", "BUY", 9.0, text)
        assert params.position_pct == 4.0

    def test_take_profit_pattern(self):
        """take-profit pattern should be recognized as target."""
        text = "Stop-loss: $258. Take-profit: $295."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.price_target == 295.0
