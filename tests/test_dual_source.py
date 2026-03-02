"""Tests for dual-source parameter extraction (Risk Judge + Trader fallback)."""

import pytest
from src.execution.trade_params import extract_trade_params_dual


class TestDualSourceExtraction:

    def test_risk_judge_has_all_params(self):
        """When Risk Judge has everything, Trader is not needed."""
        judge_text = """
        ## EXECUTION PARAMETERS
        - Decision: SELL
        - Entry Price: $264.18
        - Stop-Loss: $256.50
        - Price Target: $248.00
        - Risk/Reward Ratio: 2.1:1
        - Position Size: 3% of portfolio
        - Confidence: HIGH
        """
        trader_text = "Stop-loss: $250. Target: $240."
        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4, judge_text, trader_text, current_price=264.18
        )
        assert params.stop_loss == 256.50  # From Judge, not Trader
        assert params.price_target == 248.0

    def test_trader_fallback_for_stop_loss(self):
        """When Risk Judge lacks stop-loss, use Trader's value."""
        judge_text = "I recommend SELL. Target around $248. Position 3% of portfolio."
        trader_text = """
        3. **Execution Parameters:**
        - Decision: SELL
        - Entry Price: $264.18
        - Stop-Loss: $256.50 (3% below)
        - Price Target: $248.00
        - Position Size: 2% of portfolio
        - Confidence: HIGH
        """
        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4, judge_text, trader_text, current_price=264.18
        )
        assert params.stop_loss == 256.50  # From Trader fallback
        assert params.price_target == 248.0  # From Judge regex
        assert params.is_actionable  # Now actionable!

    def test_no_trader_text_still_works(self):
        """When no Trader text provided, behaves like single-source."""
        judge_text = "Stop-loss: $256.50. Target: $248. Position: 3%."
        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4, judge_text, "", current_price=264.18
        )
        assert params.stop_loss == 256.50
        assert params.price_target == 248.0

    def test_combined_sources_make_actionable(self):
        """Risk Judge + Trader together should produce actionable params."""
        judge_text = "Final recommendation: SELL AAPL. Target: $248. Reduce to 3% position."
        trader_text = """
        FINAL TRANSACTION PROPOSAL: **SELL**
        **Execution Parameters:**
        - Entry Price: $264.18
        - Stop-Loss: $256.50
        - Price Target: $248.00
        - Risk/Reward Ratio: 2.1:1
        - Position Size: 3% of portfolio
        """
        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4, judge_text, trader_text, current_price=264.18
        )
        assert params.stop_loss == 256.50
        assert params.price_target == 248.0
        assert params.is_actionable

    def test_risk_metrics_recalculated_after_merge(self):
        """Risk metrics should be recalculated using merged values."""
        judge_text = "Sell recommendation. Target $248."
        trader_text = "Stop-loss: $256.50. Entry: $264.18."
        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4, judge_text, trader_text, current_price=264.18
        )
        assert params.risk_pct is not None
        assert params.reward_pct is not None
        assert params.risk_reward_ratio is not None

    def test_trader_does_not_override_judge_stop_loss(self):
        """Judge's stop-loss must not be replaced by Trader's even if both present."""
        judge_text = """
        ## EXECUTION PARAMETERS
        - Stop-Loss: $260.00
        - Price Target: $248.00
        - Position Size: 3% of portfolio
        """
        trader_text = "Stop-loss: $256.50."
        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4, judge_text, trader_text, current_price=264.18
        )
        assert params.stop_loss == 260.00  # Judge wins
