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


class TestStructuredBlockExtraction:
    """Test extraction from the ## EXECUTION PARAMETERS block."""

    def test_full_structured_block(self):
        """Full structured block should extract all values."""
        text = """
        ... analysis text ...

        ## EXECUTION PARAMETERS
        - Decision: BUY
        - Entry Price: $270.15
        - Stop-Loss: $258.00 (4.5% below entry)
        - Price Target: $295.00 (9.2% above entry)
        - Risk/Reward Ratio: 2.1:1
        - Position Size: 5% of portfolio
        - Confidence: HIGH
        - Timeframe: 2-4 weeks
        """
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.stop_loss == 258.0
        assert params.price_target == 295.0
        assert params.entry_price == 270.15
        assert params.position_pct == 5.0
        assert params.confidence == "high"
        assert params.is_actionable

    def test_structured_block_overrides_body_text(self):
        """Structured block values should take priority over body text."""
        text = """
        The stop is around $250 based on support.
        Target could be $310 at resistance.

        ## EXECUTION PARAMETERS
        - Decision: BUY
        - Entry Price: $270.00
        - Stop-Loss: $258.00 (4.4% below entry)
        - Price Target: $295.00 (9.3% above entry)
        - Risk/Reward Ratio: 2.1:1
        - Position Size: 4% of portfolio
        - Confidence: MEDIUM
        - Timeframe: 1-3 months
        """
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.stop_loss == 258.0  # Not $250
        assert params.price_target == 295.0  # Not $310
        assert params.position_pct == 4.0

    def test_partial_structured_block_falls_through(self):
        """If structured block is missing some values, fall through to regex."""
        text = """
        Target at $295 based on fibonacci.

        ## EXECUTION PARAMETERS
        - Decision: BUY
        - Entry Price: $270.00
        - Stop-Loss: $258.00 (4.4% below entry)
        - Position Size: 5% of portfolio
        - Confidence: HIGH
        """
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.stop_loss == 258.0  # From structured block
        assert params.price_target == 295.0  # Fell through to regex
        assert params.position_pct == 5.0  # From structured block

    def test_no_structured_block_uses_regex(self):
        """Without structured block, existing regex should still work."""
        text = "Stop-loss: $258. Target: $295. Position size: 5% of portfolio."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.stop_loss == 258.0
        assert params.price_target == 295.0

    def test_structured_block_with_commas_in_numbers(self):
        """Numbers with commas (e.g., $1,250.00) should parse correctly."""
        text = """
        ## EXECUTION PARAMETERS
        - Decision: BUY
        - Entry Price: $1,250.00
        - Stop-Loss: $1,200.00 (4% below entry)
        - Price Target: $1,350.00 (8% above entry)
        - Risk/Reward Ratio: 2.0:1
        - Position Size: 3% of portfolio
        - Confidence: MEDIUM
        """
        params = extract_trade_params("GOOG", "BUY", 9.0, text)
        assert params.entry_price == 1250.0
        assert params.stop_loss == 1200.0
        assert params.price_target == 1350.0

    def test_risk_reward_from_structured_block(self):
        """R/R ratio should be extracted from structured block."""
        text = """
        ## EXECUTION PARAMETERS
        - Decision: BUY
        - Entry Price: $270.00
        - Stop-Loss: $258.00 (4.4% below entry)
        - Price Target: $295.00 (9.3% above entry)
        - Risk/Reward Ratio: 2.1:1
        - Position Size: 5% of portfolio
        - Confidence: HIGH
        """
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.risk_reward_ratio == 2.1
