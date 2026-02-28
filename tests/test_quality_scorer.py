"""Tests for the quality scoring system."""

from src.quality_scorer import score_pipeline_output, compare_scores


class TestQualityScorer:
    """Test quality scoring logic."""

    def test_high_quality_output(self):
        """A detailed output with risk params should score high."""
        text = """
        ## Analysis Summary

        AAPL is trading at $270 with a P/E of 33x. Revenue grew 8% YoY
        with services reaching $100B. Market cap is $4.1T.

        The technical setup shows RSI at 62, MACD bullish crossover,
        and price above the 50-day SMA at $265.

        ### Risk Management
        - Stop-loss: $258 (-4.4%)
        - Price target: $295 (+9.3%)
        - Position sizing: 5% of portfolio

        ## FINAL TRANSACTION PROPOSAL: **HOLD**

        With active risk management and trim target at $280+.
        """
        score = score_pipeline_output(
            config_name="test",
            ticker="AAPL",
            trade_date="2026-02-27",
            final_trade_decision=text,
            extracted_decision="HOLD",
        )

        assert score.decision == "HOLD"
        assert score.decision_consistent
        assert score.reasoning_depth >= 2
        assert score.data_grounding >= 3
        assert score.risk_awareness >= 7
        assert score.has_stop_loss
        assert score.has_price_target
        assert score.has_position_sizing
        assert score.composite_score >= 5.0

    def test_low_quality_output(self):
        """A vague output should score low."""
        text = "I think we should buy it."
        score = score_pipeline_output(
            config_name="test_low",
            ticker="AAPL",
            trade_date="2026-02-27",
            final_trade_decision=text,
            extracted_decision="BUY",
        )

        assert score.reasoning_depth <= 2
        assert score.data_grounding <= 1
        assert score.risk_awareness == 0
        assert score.composite_score < 4.0

    def test_inconsistent_decision(self):
        """When extracted decision doesn't match, consistency is false."""
        text = "FINAL TRANSACTION PROPOSAL: **HOLD**"
        score = score_pipeline_output(
            config_name="test_inconsistent",
            ticker="AAPL",
            trade_date="2026-02-27",
            final_trade_decision=text,
            extracted_decision="SELL",
        )

        assert not score.decision_consistent
        assert score.composite_score < 5.0

    def test_comparison_report(self):
        """Compare scores generates a formatted report."""
        scores = [
            score_pipeline_output("all_cloud", "AAPL", "2026-02-27",
                "FINAL TRANSACTION PROPOSAL: **HOLD**\nStop-loss: $258\nTarget: $295",
                "HOLD"),
            score_pipeline_output("hybrid_qwen", "AAPL", "2026-02-27",
                "FINAL TRANSACTION PROPOSAL: **HOLD**",
                "HOLD"),
        ]
        report = compare_scores(scores)
        assert "HYBRID LLM QUALITY COMPARISON" in report
        assert "all_cloud" in report
        assert "hybrid_qwen" in report
