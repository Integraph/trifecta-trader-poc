"""Tests for improved signal processing."""

from src.signal_processing import extract_decision


class TestExtractDecision:
    """Test the decision extraction logic."""

    def test_final_transaction_proposal_hold(self):
        text = """
        ## WHY THIS IS SUPERIOR TO FULL SELL
        The analyst's SELL recommendation is intellectually rigorous but operationally suboptimal.

        ## FINAL TRANSACTION PROPOSAL: **HOLD**

        With mandatory risk management:
        - Stop-loss: $258
        """
        assert extract_decision(text) == "HOLD"

    def test_final_transaction_proposal_buy(self):
        text = "FINAL TRANSACTION PROPOSAL: **BUY** AAPL stocks"
        assert extract_decision(text) == "BUY"

    def test_final_transaction_proposal_sell(self):
        text = "FINAL TRANSACTION PROPOSAL: SELL"
        assert extract_decision(text) == "SELL"

    def test_ignores_negation_not_recommending_sell(self):
        """The word SELL in 'NOT Recommending SELL' should not be picked up."""
        text = """
        ### Why I'm NOT Recommending Immediate Full SELL:
        The market conditions don't warrant panic.

        ### Why I'm NOT Recommending BUY:
        Valuation is stretched.

        ## FINAL TRANSACTION PROPOSAL: **HOLD**
        """
        assert extract_decision(text) == "HOLD"

    def test_multiple_proposals_takes_last(self):
        """When the output loops with multiple proposals, take the last one."""
        text = """
        FINAL TRANSACTION PROPOSAL: **BUY**
        ...repeated content...
        FINAL TRANSACTION PROPOSAL: **BUY**
        ...more content...
        FINAL TRANSACTION PROPOSAL: **HOLD**
        """
        assert extract_decision(text) == "HOLD"

    def test_recommendation_pattern(self):
        text = "## MY RECOMMENDATION: HOLD WITH DISCIPLINED RISK MANAGEMENT"
        assert extract_decision(text) == "HOLD"

    def test_no_markdown_bold(self):
        text = "FINAL TRANSACTION PROPOSAL: HOLD"
        assert extract_decision(text) == "HOLD"

    def test_empty_input(self):
        assert extract_decision("") == "UNKNOWN"
        assert extract_decision(None) == "UNKNOWN"

    def test_no_decision_found(self):
        text = "This is just some analysis text with no clear decision."
        assert extract_decision(text) == "UNKNOWN"

    def test_sell_in_reasoning_hold_in_proposal(self):
        """The actual bug we observed: SELL appears in reasoning but HOLD is the decision."""
        text = """
        The analyst recommends SELL based on technical weakness.
        However, considering the strong fundamentals, we disagree.

        ### Why I'm NOT Recommending Immediate Full SELL:
        - Strong balance sheet
        - Brand moat

        ### Why I'm NOT Recommending BUY:
        - Elevated valuation
        - Momentum concerns

        ## MY RECOMMENDATION: HOLD WITH DISCIPLINED RISK MANAGEMENT

        ## FINAL TRANSACTION PROPOSAL: **HOLD**

        With mandatory risk management:
        - Stop-loss: $258
        - Trim target: $280+
        """
        assert extract_decision(text) == "HOLD"

    def test_case_insensitive(self):
        text = "Final Transaction Proposal: hold"
        assert extract_decision(text) == "HOLD"

    def test_standalone_decision_fallback(self):
        """When no PROPOSAL line exists, use the last standalone decision word."""
        text = """
        After careful analysis, we believe the right course of action is to HOLD.
        """
        assert extract_decision(text) == "HOLD"


class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_buy_with_extra_text(self):
        text = "FINAL TRANSACTION PROPOSAL: **BUY** with a strategic focus on long-term growth"
        assert extract_decision(text) == "BUY"

    def test_hold_with_conditions(self):
        text = "FINAL TRANSACTION PROPOSAL: **HOLD** - active position management required"
        assert extract_decision(text) == "HOLD"

    def test_hash_prefix_on_proposal(self):
        text = "## FINAL TRANSACTION PROPOSAL: **SELL**"
        assert extract_decision(text) == "SELL"

    def test_conviction_level_does_not_interfere(self):
        text = """
        ## CONVICTION LEVEL: 7/10 ON HOLD
        Why HOLD over SELL:
        - Reasons here
        Why HOLD over BUY:
        - Reasons here
        ## FINAL TRANSACTION PROPOSAL: **HOLD**
        """
        assert extract_decision(text) == "HOLD"
