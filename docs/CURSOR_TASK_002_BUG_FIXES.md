# Cursor Task 002: Fix Signal Processing Bug and Looping Issue

## Objective
Fix two bugs identified during pipeline testing:
1. **Signal Processing Bug**: The `process_signal` function extracts the wrong decision from the pipeline output (e.g., returns "SELL" when the trader clearly recommended "HOLD")
2. **Looping Bug**: The final trade decision repeats 5 times identically before the pipeline completes

## Context
The pipeline runs end-to-end successfully (Task 001 verified this), but the final output has two problems:
- The trader agent produces a detailed analysis concluding with `FINAL TRANSACTION PROPOSAL: **HOLD**` including specific risk parameters (stop-loss at $258, trim target at $280+, re-entry zone at $235-245). However, `process_signal()` extracts "SELL" because it picks up the word "SELL" from context like "Why I'm NOT Recommending SELL" in the body of the report.
- The final trade decision output repeats 5 times identically, suggesting a loop in the risk debate or portfolio manager stage of the LangGraph state machine.

## Important Rules
- **DO NOT modify files in `vendor/TradingAgents/` directly** — we need to keep the upstream framework clean
- Instead, create wrapper/override files in `src/` that patch or extend the upstream behavior
- If a fix absolutely requires modifying vendor code, document it clearly in the report so we can submit it upstream or manage it as a patch

---

## Bug 1: Signal Processing Fix

### Investigation Steps

1. Read the signal processing code:
```bash
cat vendor/TradingAgents/tradingagents/graph/signal_processing.py
```

2. Understand how `process_signal()` works — it takes the full `final_trade_decision` text and extracts a simple BUY/HOLD/SELL signal from it.

3. The bug is that the function likely does a simple string search for "BUY", "SELL", or "HOLD" and picks up these words from the reasoning context rather than from the actual decision line.

### Expected Fix

Create `src/signal_processing.py` that implements a more robust signal extractor:

1. **Primary extraction**: Look for the pattern `FINAL TRANSACTION PROPOSAL:` followed by BUY, HOLD, or SELL. This is the authoritative decision line that every agent output uses.

2. **Secondary extraction**: If no "FINAL TRANSACTION PROPOSAL" line is found, look for the LAST occurrence of BUY/HOLD/SELL in the text (the final statement is most likely the actual decision, not earlier mentions in the reasoning).

3. **Fallback**: If neither method works, return "UNKNOWN" rather than guessing.

The extractor should:
- Be case-insensitive
- Handle markdown bold formatting (`**HOLD**`, `**BUY**`, `**SELL**`)
- Ignore occurrences inside negation phrases like "NOT Recommending SELL", "not a SELL", "Why I'm NOT Recommending BUY"
- Return one of: "BUY", "HOLD", "SELL", or "UNKNOWN"

### Implementation

Create `src/signal_processing.py`:

```python
"""
Improved signal processing for extracting trade decisions.

Fixes the upstream bug where process_signal() picks up BUY/SELL/HOLD
from reasoning context rather than the actual decision line.
"""

import re
from typing import Optional


def extract_decision(full_signal: str) -> str:
    """Extract the trade decision from the full signal text.

    Uses a priority-based extraction:
    1. Look for 'FINAL TRANSACTION PROPOSAL: <DECISION>'
    2. Look for the last standalone BUY/HOLD/SELL not in a negation context
    3. Return 'UNKNOWN' if no clear decision found

    Args:
        full_signal: The complete text output from the trading pipeline

    Returns:
        One of: 'BUY', 'HOLD', 'SELL', or 'UNKNOWN'
    """
    if not full_signal or not isinstance(full_signal, str):
        return "UNKNOWN"

    # Method 1: Look for FINAL TRANSACTION PROPOSAL line
    # Handles formats like:
    #   FINAL TRANSACTION PROPOSAL: **HOLD**
    #   FINAL TRANSACTION PROPOSAL: HOLD
    #   FINAL TRANSACTION PROPOSAL: **BUY** AAPL
    #   ## FINAL TRANSACTION PROPOSAL: **SELL**
    proposal_pattern = r'FINAL\s+TRANSACTION\s+PROPOSAL[:\s]*\*{0,2}(BUY|HOLD|SELL)\*{0,2}'
    proposals = re.findall(proposal_pattern, full_signal, re.IGNORECASE)

    if proposals:
        # Take the LAST proposal found (in case of multiple, the final one is authoritative)
        return proposals[-1].upper()

    # Method 2: Look for "MY RECOMMENDATION: <DECISION>" pattern
    recommendation_pattern = r'MY\s+RECOMMENDATION[:\s]*\*{0,2}(BUY|HOLD|SELL)\*{0,2}'
    recommendations = re.findall(recommendation_pattern, full_signal, re.IGNORECASE)

    if recommendations:
        return recommendations[-1].upper()

    # Method 3: Look for standalone decision words, excluding negation contexts
    # Remove negation phrases first
    cleaned = full_signal
    negation_patterns = [
        r"(?:NOT|n't|not)\s+(?:recommending|recommend|suggesting|suggest)\s+(?:a\s+)?(?:full\s+)?(BUY|HOLD|SELL)",
        r"(?:NOT|n't|not)\s+(?:a\s+)?(BUY|HOLD|SELL)",
        r"Why\s+(?:I'm\s+)?NOT\s+(?:Recommending\s+)?(BUY|HOLD|SELL)",
        r"(?:rather\s+than|instead\s+of|over)\s+(?:a\s+)?(?:full\s+)?(BUY|HOLD|SELL)",
    ]
    for pattern in negation_patterns:
        cleaned = re.sub(pattern, "[NEGATED]", cleaned, flags=re.IGNORECASE)

    # Find remaining standalone BUY/HOLD/SELL
    standalone_pattern = r'\b(BUY|HOLD|SELL)\b'
    decisions = re.findall(standalone_pattern, cleaned, re.IGNORECASE)

    if decisions:
        # Take the last occurrence as the most likely final decision
        return decisions[-1].upper()

    return "UNKNOWN"
```

### Update run_analysis.py

Modify `src/run_analysis.py` to use our improved signal processor instead of the upstream one. After the line where `ta.propagate()` is called, override the decision:

```python
from src.signal_processing import extract_decision

# ... after ta.propagate() returns ...
# Override the upstream signal processing with our improved version
final_trade_text = final_state.get("final_trade_decision", "")
decision = extract_decision(final_trade_text)
```

### Tests

Create `tests/test_signal_processing.py`:

```python
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
```

---

## Bug 2: Looping Fix Investigation

### Investigation Steps

1. Read the graph setup to understand the state machine:
```bash
cat vendor/TradingAgents/tradingagents/graph/setup.py
```

2. Read the conditional logic to understand loop/exit conditions:
```bash
cat vendor/TradingAgents/tradingagents/graph/conditional_logic.py
```

3. Read the propagation logic:
```bash
cat vendor/TradingAgents/tradingagents/graph/propagation.py
```

4. Look at how `max_debate_rounds` and `max_risk_discuss_rounds` control the loops:
```bash
grep -r "max_debate_rounds\|max_risk_discuss\|max_recur" vendor/TradingAgents/tradingagents/ --include="*.py"
```

### What to Look For

The looping manifests as the final HOLD/BUY/SELL recommendation being repeated 5 times identically. This suggests:
- The risk debate or portfolio manager node re-enters its own loop
- OR the conditional logic for "should we continue debating" doesn't properly detect that a decision has been reached
- OR the recursion limit counter isn't being checked correctly

### Expected Fix

Based on investigation, create a fix in `src/` that either:
- Wraps the graph setup to correctly configure loop termination
- Overrides the conditional logic to detect repeated identical outputs and break
- Adjusts the config parameters to prevent the loop

**If the fix requires modifying vendor code**, document exactly which file and lines need to change, and note this in the report so we can manage it as a tracked patch.

### Test for the Looping Fix

Add to `tests/test_pipeline.py`:

```python
"""Pipeline integration tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))


def test_decision_not_repeated(tmp_path):
    """Verify the final decision appears only once (no looping)."""
    # This test reads the most recent pipeline output from results/
    # and checks that the FINAL TRANSACTION PROPOSAL appears at most 2 times
    # (once in the trader output, once in the risk manager output)
    results_dir = Path(__file__).parent.parent / "results" / "AAPL"

    if not results_dir.exists():
        import pytest
        pytest.skip("No pipeline results available - run the pipeline first")

    # Find the most recent result
    result_files = sorted(results_dir.glob("*.json"))
    assert len(result_files) > 0, "No result files found"

    # Note: This is a placeholder test. The actual validation would need
    # access to the full pipeline output log, not just the JSON summary.
    # The real fix is in the graph conditional logic.
```

---

## Verification

### Run All Tests
```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
pytest tests/ -v
```

**Expected:** All tests pass, including the new signal processing tests.

### Run Pipeline and Verify Signal Extraction
```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
python -m src.run_analysis --ticker AAPL --date 2026-02-27 --provider anthropic --no-debug
```

**Expected:** The final decision in the JSON output should match the FINAL TRANSACTION PROPOSAL in the pipeline's reasoning, NOT a stray BUY/SELL/HOLD from the body text.

**IMPORTANT:** Running the full pipeline costs money (Anthropic API calls) and takes ~20 minutes. Only run it if the unit tests all pass first. If you want to skip the full pipeline run, note that in the report.

### Commit
```bash
git add .
git commit -m "Fix signal processing extraction and investigate looping issue"
git push
```

---

## Verification Checklist

- [ ] `src/signal_processing.py` created with robust `extract_decision()` function
- [ ] `src/run_analysis.py` updated to use improved signal processor
- [ ] `tests/test_signal_processing.py` created with 15+ test cases
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Looping issue investigated — root cause documented
- [ ] If vendor code needed modification, documented which files/lines changed
- [ ] Changes committed and pushed

---

## Report

After completing all steps, create `docs/TASK_002_REPORT.md` containing:
1. Which steps succeeded and which had issues
2. The output of `pytest tests/ -v` (all tests)
3. Root cause analysis of the looping bug
4. What fix was applied for the looping bug (or what was attempted)
5. Whether the vendor code was modified (and if so, exactly which files/lines)
6. Any modifications to the task instructions
7. The output of `git log --oneline` showing commits
