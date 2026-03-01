# Task 008: Structured Execution Output & First Live Paper Trade

**Objective:** Add a structured `## EXECUTION PARAMETERS` section to the enhanced LLM prompt so the trade parameter extractor reliably captures stop-loss, target, and position sizing. Then validate with a multi-ticker dry run, and prepare the `--execute` flow for the first real Alpaca paper trade.

**Context:** Task 007 built the execution layer, but the dry-run revealed that only 1 of 5 existing results had an extractable stop-loss. The pipeline text uses varied phrasing, making regex extraction fragile. The fix is to require a deterministic structured output section in every pipeline run, which the extractor can parse with near-perfect reliability.

**CRITICAL RULES:**
- Do NOT modify any files in `vendor/TradingAgents/`
- All new code goes in `src/` and `tests/`
- Run the full test suite after every step — all tests must pass (except the 2 pre-existing mistral failures)
- Commit after each step with a descriptive message

---

## Step 1: Add Structured Execution Parameters to Enhanced Prompt

**File:** `src/enhanced_llm.py`

Update `FINANCIAL_ANALYSIS_PREFIX` to append a structured output requirement at the end. The Trader agent (which produces the final decision) must output a machine-readable section.

Add this to the END of `FINANCIAL_ANALYSIS_PREFIX` (keep all existing content):

```
IMPORTANT: At the END of your final analysis, you MUST include this exact section with real values:

## EXECUTION PARAMETERS
- Decision: [BUY/HOLD/SELL]
- Entry Price: $[current price]
- Stop-Loss: $[exact price] ([X]% below entry)
- Price Target: $[exact price] ([X]% above entry)
- Risk/Reward Ratio: [X.X]:1
- Position Size: [X]% of portfolio
- Confidence: [HIGH/MEDIUM/LOW]
- Timeframe: [e.g., 2-4 weeks, 1-3 months]

This section is REQUIRED. Use the data from your analysis to fill in real values. Never omit the stop-loss or price target.
```

Also add the same block to `STRUCTURED_OUTPUT_PREFIX` and `FEW_SHOT_PREFIX` for consistency across all styles.

**Tests:** Add tests to `tests/test_enhanced_llm.py`:
- `test_execution_parameters_in_financial_prefix` — verify the prefix contains "## EXECUTION PARAMETERS"
- `test_execution_parameters_in_structured_prefix` — same for structured style
- `test_execution_parameters_in_few_shot_prefix` — same for few_shot style

---

## Step 2: Upgrade the Trade Parameter Extractor

**File:** `src/execution/trade_params.py`

Add a NEW extraction strategy that runs FIRST — before the existing regex patterns. It looks for the structured `## EXECUTION PARAMETERS` block and extracts values from it.

```python
def _extract_from_structured_block(text: str) -> dict:
    """Extract parameters from the ## EXECUTION PARAMETERS block.

    Returns dict with keys: stop_loss, price_target, position_pct,
    entry_price, confidence, risk_reward_ratio
    Any key may be None if not found.
    """
```

Pattern to match the block:
```python
# Find the EXECUTION PARAMETERS section
block_match = re.search(
    r'##\s*EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
    text,
    re.DOTALL | re.IGNORECASE,
)
```

Within the block, extract each line:
```python
# - Stop-Loss: $258.00 (4.5% below entry)
stop_match = re.search(r'Stop-Loss:\s*\$?([\d,]+(?:\.\d+)?)', block_text)
# - Price Target: $295.00 (9.2% above entry)
target_match = re.search(r'Price Target:\s*\$?([\d,]+(?:\.\d+)?)', block_text)
# - Entry Price: $270.15
entry_match = re.search(r'Entry Price:\s*\$?([\d,]+(?:\.\d+)?)', block_text)
# - Position Size: 5% of portfolio
size_match = re.search(r'Position Size:\s*([\d.]+)%', block_text)
# - Risk/Reward Ratio: 2.1:1
rr_match = re.search(r'Risk/Reward Ratio:\s*([\d.]+)', block_text)
# - Confidence: HIGH
conf_match = re.search(r'Confidence:\s*(\w+)', block_text)
```

Update `extract_trade_params()` to call `_extract_from_structured_block()` FIRST. If it returns values, use them. Fall through to existing regex patterns for any values not found in the structured block.

**Tests:** Add to `tests/test_trade_params.py`:

```python
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
        # Should use structured block values, not body text
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
```

---

## Step 3: Isolated Structured Output Validation

**File:** `tests/test_structured_output.py` (new)

This test validates that the enhanced Qwen model actually produces a `## EXECUTION PARAMETERS` block when given the updated prompt. It requires a running Ollama instance with `qwen2.5:14b`.

```python
"""Test that enhanced local models produce structured EXECUTION PARAMETERS blocks.

Run with: pytest tests/test_structured_output.py -v --run-comparison
"""

import pytest
import re

pytestmark = pytest.mark.skipif(
    "not config.getoption('--run-comparison')",
    reason="Requires --run-comparison flag and running Ollama",
)


def _model_available(model_name: str) -> bool:
    """Check if model is available in Ollama."""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        return model_name in result.stdout
    except Exception:
        return False


# Sample analysis data for the model to work with
SAMPLE_ANALYSIS_DATA = """
AAPL Current Analysis:
- Current Price: $272.15
- 50-day SMA: $265.20
- 200-day SMA: $252.80
- RSI(14): 58.3
- P/E Ratio: 28.5x
- Forward P/E: 26.2x
- EPS (TTM): $9.55
- Revenue Growth: 6.2% YoY
- Free Cash Flow: $108.4B
- Debt/Equity: 1.87
- 52-week range: $230.15 - $285.40
- Average Volume: 48.2M
- MACD: 1.35 (signal: 0.89)
- Analyst consensus: 32 buy, 8 hold, 2 sell

Based on this data, provide a complete trading analysis for AAPL with a BUY recommendation.
"""


class TestStructuredOutput:

    @pytest.mark.skipif(
        not _model_available("qwen2.5:14b"),
        reason="qwen2.5:14b not available in Ollama",
    )
    def test_qwen14b_produces_execution_params_block(self):
        """Qwen 14B with enhanced prompt should produce ## EXECUTION PARAMETERS."""
        from langchain_openai import ChatOpenAI
        from src.enhanced_llm import create_enhanced_llm

        base_llm = ChatOpenAI(
            model="qwen2.5:14b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.1,
        )
        enhanced = create_enhanced_llm(base_llm, style="financial_analysis")

        response = enhanced.invoke(SAMPLE_ANALYSIS_DATA)
        text = response.content if hasattr(response, 'content') else str(response)

        # Must contain the structured block
        assert "EXECUTION PARAMETERS" in text, (
            f"Model did not produce EXECUTION PARAMETERS block. Output:\n{text[:500]}"
        )

        # Extract the block and verify key fields
        block_match = re.search(
            r'(?:##\s*)?EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
            text,
            re.DOTALL | re.IGNORECASE,
        )
        assert block_match, "Could not parse EXECUTION PARAMETERS block"
        block = block_match.group(1)

        # Check required fields are present
        assert re.search(r'Stop-Loss:\s*\$[\d,]+', block), f"No stop-loss in block:\n{block}"
        assert re.search(r'Price Target:\s*\$[\d,]+', block), f"No target in block:\n{block}"
        assert re.search(r'Position Size:\s*[\d.]+%', block), f"No position size in block:\n{block}"

        print(f"\n{'='*60}")
        print("Structured output from qwen2.5:14b:")
        print(f"{'='*60}")
        print(text[-800:])  # Print last 800 chars to show the block

    @pytest.mark.skipif(
        not _model_available("qwen2.5:14b"),
        reason="qwen2.5:14b not available in Ollama",
    )
    def test_extracted_params_from_live_model(self):
        """Full integration: enhanced model -> extractor -> TradeParams."""
        from langchain_openai import ChatOpenAI
        from src.enhanced_llm import create_enhanced_llm
        from src.execution.trade_params import extract_trade_params

        base_llm = ChatOpenAI(
            model="qwen2.5:14b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.1,
        )
        enhanced = create_enhanced_llm(base_llm, style="financial_analysis")

        response = enhanced.invoke(SAMPLE_ANALYSIS_DATA)
        text = response.content if hasattr(response, 'content') else str(response)

        params = extract_trade_params(
            ticker="AAPL",
            decision="BUY",
            quality_score=9.5,
            decision_text=text,
            current_price=272.15,
        )

        print(f"\nExtracted params:")
        print(f"  Stop-loss:   ${params.stop_loss}")
        print(f"  Target:      ${params.price_target}")
        print(f"  Position %:  {params.position_pct}%")
        print(f"  Entry:       ${params.entry_price}")
        print(f"  R/R ratio:   {params.risk_reward_ratio}")
        print(f"  Confidence:  {params.confidence}")
        print(f"  Actionable:  {params.is_actionable}")
        print(f"  Bracket:     {params.has_bracket_params}")

        # With structured output, these should all be present
        assert params.stop_loss is not None, "Stop-loss not extracted"
        assert params.price_target is not None, "Price target not extracted"
        assert params.position_pct is not None, "Position size not extracted"
        assert params.is_actionable, "Should be actionable with all params present"
        assert params.has_bracket_params, "Should have bracket params"
```

Run with: `pytest tests/test_structured_output.py -v --run-comparison`

---

## Step 4: Multi-Ticker Pipeline Dry Run

Run the full pipeline with `--dry-run` for 3 tickers to verify end-to-end extraction with the new structured output. This generates fresh results with the updated enhanced prompt.

```bash
# Run one at a time (each takes ~20-30 minutes)
python -m src.run_analysis --ticker AAPL --hybrid hybrid_qwen_enhanced --dry-run --no-debug
python -m src.run_analysis --ticker TSLA --hybrid hybrid_qwen_enhanced --dry-run --no-debug
python -m src.run_analysis --ticker JPM --hybrid hybrid_qwen_enhanced --dry-run --no-debug
```

**Important:** If these runs are too expensive/slow for this task, SKIP them and note in the report that they were skipped. The isolated test in Step 3 validates the structured output works. The full pipeline runs can be done manually later.

**Expected outcome:** All 3 results should now show `Actionable: True` with stop-loss, target, and position size extracted. The dry-run output should display complete `TRADE PARAMETERS` with no `N/A` values.

**Record the dry-run output for each ticker in the task report.**

---

## Step 5: Alpaca Paper Account Setup Validation

**File:** `tests/test_alpaca_connection.py` (new)

Create a test that validates the Alpaca paper trading connection works. This test only runs when env vars are set.

```python
"""Test Alpaca paper trading connection.

Run with: pytest tests/test_alpaca_connection.py -v
Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars.
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("APCA_API_KEY_ID"),
    reason="Alpaca API credentials not set",
)


class TestAlpacaConnection:

    def test_paper_account_accessible(self):
        """Should connect to Alpaca paper trading and get account info."""
        from alpaca.trading.client import TradingClient

        client = TradingClient(
            api_key=os.environ["APCA_API_KEY_ID"],
            secret_key=os.environ["APCA_API_SECRET_KEY"],
            paper=True,
        )
        account = client.get_account()

        assert account is not None
        assert float(account.portfolio_value) > 0
        print(f"\nAlpaca Paper Account:")
        print(f"  Portfolio value: ${float(account.portfolio_value):,.2f}")
        print(f"  Buying power:    ${float(account.buying_power):,.2f}")
        print(f"  Cash:            ${float(account.cash):,.2f}")
        print(f"  Status:          {account.status}")

    def test_position_manager_integration(self):
        """PositionManager should work with real Alpaca client."""
        from alpaca.trading.client import TradingClient
        from src.execution.position_manager import PositionManager

        client = TradingClient(
            api_key=os.environ["APCA_API_KEY_ID"],
            secret_key=os.environ["APCA_API_SECRET_KEY"],
            paper=True,
        )
        pm = PositionManager(client)
        state = pm.get_account_state()

        assert state.portfolio_value > 0
        assert state.buying_power >= 0

        positions = pm.get_positions()
        print(f"\n  Open positions: {len(positions)}")
        for ticker, pos in positions.items():
            print(f"    {ticker}: {pos.qty} shares @ ${pos.current_price:.2f}")

    def test_executor_initialization(self):
        """TradeExecutor should initialize with paper=True."""
        from src.execution.executor import TradeExecutor

        executor = TradeExecutor()
        # Verify the client was created successfully
        assert executor.client is not None

        # Verify we can get account (proves connection works)
        account = executor.client.get_account()
        assert account is not None
        print(f"\n  Executor connected to paper account")
        print(f"  Account status: {account.status}")
```

**Note to Cursor:** If APCA_API_KEY_ID is not set in `.env`, these tests will be auto-skipped. That's fine — just report what happened. The user will set up credentials separately and run these tests manually.

---

## Step 6: Full Test Suite & Report

Run the complete test suite:
```bash
pytest tests/ -v --tb=short 2>&1 | head -100
```

**Expected results:**
- All existing tests still pass (except 2 pre-existing mistral failures)
- New enhanced prompt tests pass (Step 1)
- New structured block extraction tests pass (Step 2)
- Structured output tests pass if Ollama is running (Step 3)
- Alpaca connection tests skip if no credentials (Step 5)

---

## Report Requirements

Create `docs/TASK_008_REPORT.md` with:

1. **Step summary table** (step, status, notes)
2. **Enhanced prompt diff** — show what was added to each prefix
3. **Structured block extraction results** — show test outputs
4. **Isolated model test results** (Step 3) — paste the actual model output showing the EXECUTION PARAMETERS block
5. **Multi-ticker dry-run results** (Step 4) — if run, paste dry-run output; if skipped, note why
6. **Alpaca connection test results** (Step 5) — pass or skip
7. **Full test output** with counts
8. **Updated git log**
9. **Vendor code modifications** — should be "None"
10. **Stop-loss extraction rate** — compare before (1/5 = 20%) vs after

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/enhanced_llm.py` | Modify — add EXECUTION PARAMETERS block to all 3 prefixes |
| `src/execution/trade_params.py` | Modify — add `_extract_from_structured_block()`, call it first |
| `tests/test_enhanced_llm.py` | Modify — add 3 tests for EXECUTION PARAMETERS in prefixes |
| `tests/test_trade_params.py` | Modify — add `TestStructuredBlockExtraction` class with 5 tests |
| `tests/test_structured_output.py` | Create — 2 isolated model tests (--run-comparison) |
| `tests/test_alpaca_connection.py` | Create — 3 connection tests (skip if no creds) |
| `docs/TASK_008_REPORT.md` | Create — task report |

**Do NOT modify any files in `vendor/TradingAgents/`.**
