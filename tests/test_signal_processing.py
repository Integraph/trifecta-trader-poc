"""Tests for improved signal processing."""

from src.signal_processing import extract_decision, deduplicate_repeated_blocks


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

    # --- TradingAgents v0.3.0 Portfolio Manager 5-level rating (TRI-66) ---

    def test_pm_rating_overweight_maps_to_buy(self):
        """v0.3.0 PM free-text render: '**Rating**: Overweight' -> BUY."""
        text = "**Rating**: Overweight\n\n**Executive Summary**: Initiate/add to AAPL..."
        assert extract_decision(text) == "BUY"

    def test_pm_recommendation_underweight_maps_to_sell(self):
        """v0.3.0 PM structured render: '**Recommendation**: Underweight' -> SELL."""
        assert extract_decision("**Recommendation**: Underweight") == "SELL"

    def test_pm_rating_all_five_levels(self):
        expected = {
            "Buy": "BUY", "Overweight": "BUY", "Hold": "HOLD",
            "Underweight": "SELL", "Sell": "SELL",
        }
        for rating, decision in expected.items():
            assert extract_decision(f"**Rating**: {rating}") == decision

    def test_pm_rating_ignores_prose_mention(self):
        """A rating word in reasoning prose (no header) must not be picked up."""
        text = "We weighed an Overweight thesis but the Underweight risks won out."
        assert extract_decision(text) == "UNKNOWN"

    def test_pm_action_label_local_qwen(self):
        """Local qwen labels the PM decision '**Action**: <rating>'."""
        assert extract_decision("**Action**: Underweight") == "SELL"
        assert extract_decision("**Action**: Overweight") == "BUY"

    def test_pm_final_transaction_proposal_5level(self):
        """Local qwen also writes '**Final Transaction Proposal: <5-level rating>**'."""
        assert extract_decision("**Final Transaction Proposal: Underweight**") == "SELL"
        assert extract_decision("Final Transaction Proposal: Overweight") == "BUY"

    def test_pm_ignores_underweighted_prose(self):
        """'underweighted position' in prose must not map to SELL."""
        assert extract_decision("an underweighted position might be prudent") == "UNKNOWN"

    # --- Codex QA P1 adversarial set (TRI-66): label-shaped prose must NOT
    # --- override the real decision header. Exact repro cases from the review.

    def test_pm_header_not_overridden_by_prose_rating(self):
        text = "**Rating**: Hold\n\nRationale: prior note wrote Rating: Underweight."
        assert extract_decision(text) == "HOLD"

    def test_pm_header_not_overridden_by_prose_action(self):
        text = "**Recommendation**: Buy\n\nRationale: prior playbook said Action: Sell."
        assert extract_decision(text) == "BUY"

    def test_pm_header_not_overridden_by_prose_proposal(self):
        text = (
            "**Action**: Hold\n\n"
            "Rationale: stale memo labeled Final Transaction Proposal: Sell."
        )
        assert extract_decision(text) == "HOLD"

    def test_pm_label_in_prose_only_is_unknown(self):
        text = (
            "The analyst wrote Rating: Underweight in the bearish argument; "
            "we have not decided yet."
        )
        assert extract_decision(text) == "UNKNOWN"

    def test_pm_midline_bold_label_is_not_a_header(self):
        """Even a bold label mid-sentence is not a decision line."""
        assert extract_decision("We keep **Rating**: Overweight in place.") == "UNKNOWN"

    def test_pm_looping_output_last_decision_line_wins(self):
        text = "**Rating**: Buy\n...model loops...\n**Rating**: Hold"
        assert extract_decision(text) == "HOLD"

    def test_pm_markdown_heading_and_list_decorations(self):
        assert extract_decision("### Recommendation: Overweight") == "BUY"
        assert extract_decision("- **Action**: Underweight") == "SELL"

    def test_pm_qualified_label_investment_recommendation(self):
        """qwen also emits '**Investment Recommendation: <rating>**' (varies per run)."""
        assert extract_decision("**Investment Recommendation: Underweight**") == "SELL"
        assert extract_decision("**Final Recommendation**: Hold") == "HOLD"

    def test_pm_unlisted_qualifier_does_not_match(self):
        """Random words before the label are not decision headers.

        Uses 'Underweight' (5-level-only) so the legacy method-3 standalone
        BUY/HOLD/SELL fallback can't mask the method-0 behavior under test.
        """
        text = "Yesterday Recommendation: Underweight was the desk chatter."
        assert extract_decision(text) == "UNKNOWN"

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


class TestDeduplicateRepeatedBlocks:
    """Test the deduplication utility for repeated pipeline output."""

    def test_deduplicate_removes_repeated_blocks(self):
        """Repeated identical blocks should be reduced to one occurrence."""
        block = "A" * 250
        text = f"{block}\nContinue\n{block}\nContinue\n{block}"
        result = deduplicate_repeated_blocks(text)
        assert result.count(block) == 1

    def test_deduplicate_preserves_unique_content(self):
        """Unique content should pass through unchanged."""
        text = "First unique block.\n\nSecond unique block.\n\nThird unique block."
        result = deduplicate_repeated_blocks(text)
        assert result == text

    def test_short_text_returned_unchanged(self):
        """Texts shorter than 2x min_block_length are returned as-is."""
        text = "Short text."
        result = deduplicate_repeated_blocks(text)
        assert result == text

    def test_empty_text_returned_unchanged(self):
        """Empty string should be returned unchanged."""
        assert deduplicate_repeated_blocks("") == ""

    def test_none_text_returned_unchanged(self):
        """None should be returned as-is (guard against None input)."""
        assert deduplicate_repeated_blocks(None) is None
