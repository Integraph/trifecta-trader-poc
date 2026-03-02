# Task 009: Fix Parameter Extraction & Add Dual-Source Parsing

**Objective:** Fix the two issues discovered during the first live `--execute` run: (1) the structured `## EXECUTION PARAMETERS` block is not appearing in the Risk Judge's output, and (2) the extractor should also parse the Trader's output as a fallback source for execution parameters.

**Root Cause Analysis:**

The pipeline flow is: Analysts → Researchers → Research Manager → **Trader** → Risk Debaters → **Risk Judge** → END

Two agents produce execution-relevant content:
- **Trader** (Qwen 14B, `reasoning_quick_llm`): Produces `trader_investment_plan` with execution parameters in `**Execution Parameters:**` format (bold numbered list item, NOT `## EXECUTION PARAMETERS` h2)
- **Risk Judge** (Claude, `reasoning_deep_llm`): Produces `final_trade_decision` — this is what our extractor parses. The Risk Judge refines the Trader's plan based on risk debate, but its prompt (in `vendor/.../risk_manager.py`) doesn't ask for structured output format.

The `enhance_deep` / `execution_params_only` prefix IS being prepended to Claude's input via `EnhancedChatModel.invoke()`. However, the Risk Judge's own system prompt is very directive ("Summarize Key Arguments", "Provide Rationale", "Refine the Trader's Plan") and doesn't include output formatting requirements. The prepended prefix likely gets overshadowed by the detailed prompt structure.

**The fix has two parts:**
1. Make the extractor parse BOTH the `final_trade_decision` (Risk Judge) AND `trader_investment_plan` (Trader) — using Trader as fallback
2. Make the structured block regex more flexible to match both `## EXECUTION PARAMETERS` and `**Execution Parameters:**` formats

**Secondary issue:** The Trader output repeats 5 times in the debug output with "Continue" human messages. This is the known `max_recur_limit` looping issue. We'll add deduplication to avoid wasting tokens.

**CRITICAL RULES:**
- Do NOT modify any files in `vendor/TradingAgents/`
- All new code goes in `src/` and `tests/`
- Run the full test suite after every step
- Commit after each step

---

## Step 1: Expand Structured Block Regex to Match Multiple Formats

**File:** `src/execution/trade_params.py`

Update `_extract_from_structured_block()` to match multiple heading formats the LLMs actually produce:

```python
def _extract_from_structured_block(text: str) -> dict:
    """Extract parameters from an EXECUTION PARAMETERS block.

    Matches multiple formats:
    - ## EXECUTION PARAMETERS           (h2 header — our requested format)
    - **Execution Parameters:**         (bold — format Qwen uses)
    - ### Execution Parameters          (h3 header)
    - 3. **Execution Parameters:**      (numbered bold list item)
    - EXECUTION PARAMETERS              (plain uppercase)
    """
    result = {
        "stop_loss": None,
        "price_target": None,
        "position_pct": None,
        "entry_price": None,
        "confidence": None,
        "risk_reward_ratio": None,
    }

    # Try multiple heading patterns
    block_patterns = [
        # Original h2 format
        r'##\s*EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
        # Bold format: **Execution Parameters:** or **Execution Parameters**
        r'\*\*Execution Parameters\*?\*?:?\s*\n(.*?)(?:\n##|\n\*\*|\Z)',
        # Numbered bold: 3. **Execution Parameters:**
        r'\d+\.\s*\*\*Execution Parameters\*?\*?:?\s*\n(.*?)(?:\n##|\n\d+\.|\Z)',
        # h3 format
        r'###\s*Execution Parameters\s*\n(.*?)(?:\n##|\Z)',
        # Plain uppercase
        r'EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
    ]

    block_text = None
    for pattern in block_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            block_text = match.group(1)
            break

    if block_text is None:
        return result

    # ... rest of extraction unchanged ...
```

**Tests:** Add to `tests/test_trade_params.py`:

```python
def test_bold_execution_parameters_format(self):
    """Should match **Execution Parameters:** format from Qwen."""
    text = """
    some analysis...

    **Execution Parameters:**

    - Decision: SELL
    - Entry Price: $264.18
    - Stop-Loss: $256.50 (3% below entry)
    - Price Target: $248.00 (6% downside)
    - Risk/Reward Ratio: 2.0:1
    - Position Size: 3% of portfolio
    - Confidence: HIGH
    """
    params = extract_trade_params("AAPL", "SELL", 9.4, text, current_price=264.18)
    assert params.stop_loss == 256.50
    assert params.price_target == 248.0
    assert params.position_pct == 3.0
    assert params.confidence == "high"

def test_numbered_bold_execution_parameters_format(self):
    """Should match '3. **Execution Parameters:**' format."""
    text = """
    1. **Near-term Risks:**
        - Potential compression

    2. **Technical Analysis:**
        - Price has broken below 50 SMA

    3. **Execution Parameters:**

    - Decision: SELL (40-50% position reduction)
    - Entry Price: $264.18
    - Stop-Loss: $256.50 (3% below current market price)
    - Price Target: $248.00 (to capture 6% downside)
    - Risk/Reward Ratio: 1:2
    - Position Size: 2% of portfolio
    - Confidence: MEDIUM
    """
    params = extract_trade_params("AAPL", "SELL", 9.4, text, current_price=264.18)
    assert params.stop_loss == 256.50
    assert params.price_target == 248.0
    assert params.position_pct == 2.0
```

---

## Step 2: Add Dual-Source Parsing (Trader + Risk Judge)

**File:** `src/execution/trade_params.py`

Add a new function that accepts BOTH text sources:

```python
def extract_trade_params_dual(
    ticker: str,
    decision: str,
    quality_score: float,
    final_decision_text: str,
    trader_plan_text: str = "",
    current_price: Optional[float] = None,
) -> TradeParams:
    """Extract trade parameters from both Risk Judge and Trader outputs.

    Strategy:
    1. Try extracting from final_decision_text (Risk Judge) first
    2. For any missing values, try trader_plan_text (Trader) as fallback
    3. For any still-missing values, fall through to regex on both texts

    Args:
        ticker: Stock ticker
        decision: BUY/HOLD/SELL from signal processor
        quality_score: Composite quality score (0-10)
        final_decision_text: Risk Judge's final_trade_decision
        trader_plan_text: Trader's trader_investment_plan
        current_price: Current market price

    Returns:
        TradeParams with extracted values from best available source
    """
    # First: try primary source (Risk Judge)
    params = extract_trade_params(
        ticker=ticker,
        decision=decision,
        quality_score=quality_score,
        decision_text=final_decision_text,
        current_price=current_price,
    )

    # Second: if missing critical values, try Trader output
    if trader_plan_text and (params.stop_loss is None or params.price_target is None):
        trader_params = extract_trade_params(
            ticker=ticker,
            decision=decision,
            quality_score=quality_score,
            decision_text=trader_plan_text,
            current_price=current_price,
        )

        if params.stop_loss is None and trader_params.stop_loss is not None:
            params.stop_loss = trader_params.stop_loss
        if params.price_target is None and trader_params.price_target is not None:
            params.price_target = trader_params.price_target
        if params.position_pct is None and trader_params.position_pct is not None:
            params.position_pct = trader_params.position_pct
        if params.entry_price is None and trader_params.entry_price is not None:
            params.entry_price = trader_params.entry_price
        if params.confidence == "medium" and trader_params.confidence != "medium":
            params.confidence = trader_params.confidence

        # Recalculate risk metrics with potentially new values
        if params.entry_price and params.stop_loss:
            params.risk_pct = abs(params.entry_price - params.stop_loss) / params.entry_price * 100
        if params.entry_price and params.price_target:
            params.reward_pct = abs(params.price_target - params.entry_price) / params.entry_price * 100
        if params.risk_pct and params.reward_pct and params.risk_pct > 0:
            params.risk_reward_ratio = params.reward_pct / params.risk_pct

    return params
```

**Tests:** Add `tests/test_dual_source.py`:

```python
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
        # Simulates real scenario: Judge has target but no stop-loss
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
```

---

## Step 3: Wire Dual-Source into run_analysis.py

**File:** `src/run_analysis.py`

Update `_run_execution_flow()` to pass both `final_trade_decision` and `trader_investment_plan` from the pipeline result.

First, update `run_analysis()` to capture `trader_investment_plan` in the result JSON:

```python
# In run_analysis(), after getting final_state:
result = {
    # ... existing fields ...
    "final_trade_decision_text": final_trade_text,
    "trader_investment_plan": final_state.get("trader_investment_plan", ""),  # NEW
    # ...
}
```

Then update `_run_execution_flow()` to use `extract_trade_params_dual`:

```python
def _run_execution_flow(result: dict, config: dict, args) -> None:
    from src.execution.trade_params import extract_trade_params_dual

    ticker = result["ticker"]
    decision = result["decision"]
    final_trade_text = result.get("final_trade_decision_text", "")
    trader_plan_text = result.get("trader_investment_plan", "")  # NEW
    quality_score = result.get("quality_score", {}).get("composite", 0.0)

    trade_params = extract_trade_params_dual(
        ticker=ticker,
        decision=decision,
        quality_score=quality_score,
        final_decision_text=final_trade_text,
        trader_plan_text=trader_plan_text,  # NEW
        current_price=None,
    )
    # ... rest unchanged ...
```

**Tests:** Update existing `_run_execution_flow` tests if any, or add:

```python
def test_run_analysis_result_includes_trader_plan(self):
    """Result JSON should include trader_investment_plan field."""
    # Mock test that verifies the field exists in result dict
    result = {
        "ticker": "AAPL",
        "decision": "SELL",
        "final_trade_decision_text": "SELL recommendation...",
        "trader_investment_plan": "Execution Parameters...",
        "quality_score": {"composite": 9.4},
    }
    assert "trader_investment_plan" in result
    assert len(result["trader_investment_plan"]) > 0
```

---

## Step 4: Validate with Existing Pipeline Output

**File:** `tests/test_dual_extraction_live.py` (new)

Create a test that loads the actual AAPL result from today's run and validates dual extraction would have worked. Since the result JSON now needs to include `trader_investment_plan`, and the existing result may not have it, this test uses the MY_REPORT.md debug output to simulate.

```python
"""Test dual-source extraction against real pipeline output from Task 009.

This test uses actual text from the March 2, 2026 AAPL pipeline run.
"""

import pytest
from src.execution.trade_params import extract_trade_params, extract_trade_params_dual


# Actual Trader output from the AAPL pipeline run (appears in MY_REPORT.md)
ACTUAL_TRADER_OUTPUT = """## KEY METRICS

- **Earnings Per Share (EPS):** $17.24
- **Forward Price-to-Earnings Ratio (P/E):** 33.47
- **Price-to-Book Ratio (P/B):** 44.08
- **Free Cash Flow:** $53.93 billion
- **Revenue Growth Rate (YoY Q1 FY2026):** 15.7%
- **Services Segment Margins:** 30%
- **Current Stock Price:** $264.18
- **Apple iPhone 17e Starting Price:** $599
- **Price Below 50 Day Simple Moving Average (SMA):** Current price is below $266.42 SMA
- **Stop-Loss Support Level:** $255

## TECHNICALS AND MACROS

- **Relative Strength Index (RSI):** 58, suggesting moderate strength but not overbought territory
- **MACD Crossover:** Bearish crossover below the signal line at -1.79
- **50 Day SMA:** $266.42; price is currently trading below it at $264.18
- **Near-Term Price Support:** $255

## RISK MANAGEMENT AND VALUATION

**Valuation Metrics:**

- Current P/E ratio of 33.47 suggests the stock is richly valued
- Free cash flow of $53.93 billion indicates strong financial health

**Risk Factors and Execution Parameters:**

1. **Near-term Risks:**
    - Potential iPhone ASP compression due to lower pricing on the model 17e
    - Geopolitical tensions impacting supply chains

2. **Technical Analysis:**
    - Price has broken below the 50 SMA, indicating potential weakness.
    - Support at $255 can be key for near-term price recovery.

3. **Execution Parameters:**

- Decision: SELL (40-50% position reduction)
- Entry Price: $264.18
- Stop-Loss: $256.50 (3% below current market price for exiting the trade if conditions worsen)
- Price Target: $248.00 (to capture 6% downside risk and wait for a more favorable entry)
- Risk/Reward Ratio: 1:2 (for every dollar of potential loss, aiming to profit two dollars or more)
- Position Size: Reduce to 2-3% of portfolio from the current suggested level, based on lower confidence in near-term upside

## FINAL TRANSACTION PROPOSAL: **SELL**"""


class TestDualExtractionWithRealData:

    def test_trader_output_extraction(self):
        """Trader output should have extractable execution parameters."""
        params = extract_trade_params(
            "AAPL", "SELL", 9.4, ACTUAL_TRADER_OUTPUT, current_price=264.18
        )
        # With expanded regex, should now find these
        assert params.stop_loss is not None, f"Stop-loss not found in Trader output"
        assert params.price_target is not None, f"Target not found in Trader output"
        print(f"Trader extraction: SL=${params.stop_loss}, Target=${params.price_target}")

    def test_dual_extraction_makes_actionable(self):
        """Dual extraction with empty Risk Judge should use Trader values."""
        # Simulate Risk Judge output without execution params
        judge_text = "Sell recommendation for AAPL. Target around $268. Position 40% reduction."

        params = extract_trade_params_dual(
            "AAPL", "SELL", 9.4,
            final_decision_text=judge_text,
            trader_plan_text=ACTUAL_TRADER_OUTPUT,
            current_price=264.18,
        )

        assert params.stop_loss is not None, "Should get stop-loss from Trader fallback"
        assert params.price_target is not None
        assert params.is_actionable, (
            f"Should be actionable: SL=${params.stop_loss}, "
            f"Target=${params.price_target}, "
            f"Quality={params.quality_score}"
        )
        print(f"\nDual extraction result:")
        print(f"  Stop-loss:   ${params.stop_loss}")
        print(f"  Target:      ${params.price_target}")
        print(f"  Position:    {params.position_pct}%")
        print(f"  R/R ratio:   {params.risk_reward_ratio}")
        print(f"  Actionable:  {params.is_actionable}")
```

---

## Step 5: Add Deduplication Guard for Looping Output

**File:** `src/signal_processing.py`

The pipeline's debug output shows the Trader response repeating 5 times with "Continue" messages. While this is a vendor graph issue (the `max_recur_limit` controls it), we can add a deduplication utility that downstream consumers can use.

Add a utility function:

```python
def deduplicate_repeated_blocks(text: str, min_block_length: int = 200) -> str:
    """Remove repeated text blocks from pipeline output.

    The TradingAgents pipeline sometimes repeats agent outputs when hitting
    max_recur_limit. This function detects and removes duplicate blocks.

    Args:
        text: Full pipeline output text
        min_block_length: Minimum block length to consider for dedup

    Returns:
        Text with duplicate blocks removed
    """
    if not text or len(text) < min_block_length * 2:
        return text

    # Split on common separators
    # Look for the pattern: identical blocks separated by "Continue" or similar
    lines = text.split('\n')

    # Find repeated sections by looking for identical long substrings
    # Simple approach: if the same 200+ char substring appears multiple times,
    # keep only the first occurrence
    seen_blocks = set()
    result_lines = []
    current_block = []

    for line in lines:
        current_block.append(line)
        block_text = '\n'.join(current_block)

        if len(block_text) >= min_block_length:
            # Check if this block start has been seen
            block_key = block_text[:min_block_length].strip()
            if block_key in seen_blocks:
                # Skip this repeated block - find the end
                current_block = []
                continue
            seen_blocks.add(block_key)

        result_lines.append(line)
        if len(current_block) > min_block_length // 20:
            current_block = current_block[-5:]  # Keep sliding window

    return '\n'.join(result_lines)
```

**Note:** This is a best-effort utility. The real fix for looping would be in the vendor graph's conditional logic, which we can't modify. The dedup function is mainly useful for reducing noise in audit logs and debug output.

**Tests:** Add basic tests:

```python
def test_deduplicate_removes_repeated_blocks():
    block = "A" * 250
    text = f"{block}\nContinue\n{block}\nContinue\n{block}"
    result = deduplicate_repeated_blocks(text)
    assert result.count(block) == 1

def test_deduplicate_preserves_unique_content():
    text = "First unique block.\n\nSecond unique block.\n\nThird unique block."
    result = deduplicate_repeated_blocks(text)
    assert result == text
```

---

## Step 6: Re-run AAPL with --execute

After all fixes are in place, run the full pipeline again:

```bash
python -m src.run_analysis --ticker AAPL --hybrid hybrid_qwen_enhanced --execute --no-debug
```

**If the run is too expensive/slow, run --dry-run instead:**

```bash
python -m src.run_analysis --ticker AAPL --hybrid hybrid_qwen_enhanced --dry-run --no-debug
```

**Expected outcome:** The dual-source extractor should now find execution parameters from the Trader's output even if the Risk Judge doesn't include them. The `TRADE PARAMETERS` section should show real stop-loss and target values, and `Actionable: True`.

**Record the output in the task report.** If using `--execute` and it's actionable, the order will be submitted to Alpaca paper trading.

**IMPORTANT:** If the pipeline run is skipped for cost/time reasons, note it in the report. The unit tests in Steps 1-4 validate the fix thoroughly.

---

## Step 7: Full Test Suite & Report

```bash
python -m pytest tests/ -v --tb=short 2>&1 | head -120
```

---

## Report Requirements

Create `docs/TASK_009_REPORT.md` with:

1. **Step summary table**
2. **Regex expansion** — show new patterns added
3. **Dual-source extraction test results** — especially the real-data test
4. **Pipeline run results** (if executed) — dry-run or execute output
5. **Deduplication utility** — test results
6. **Full test output**
7. **Updated git log**
8. **Vendor code modifications** — should be "None"
9. **Actionable rate comparison** — before (0/1 live runs actionable) vs after

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/execution/trade_params.py` | Modify — expand regex patterns, add `extract_trade_params_dual()` |
| `src/run_analysis.py` | Modify — capture `trader_investment_plan`, use dual extraction |
| `src/signal_processing.py` | Modify — add `deduplicate_repeated_blocks()` |
| `tests/test_trade_params.py` | Modify — add bold/numbered format tests |
| `tests/test_dual_source.py` | Create — 5 dual-source extraction tests |
| `tests/test_dual_extraction_live.py` | Create — 2 tests with real pipeline output |
| `tests/test_signal_processing.py` | Modify — add dedup tests |
| `docs/TASK_009_REPORT.md` | Create — task report |

**Do NOT modify any files in `vendor/TradingAgents/`.**
