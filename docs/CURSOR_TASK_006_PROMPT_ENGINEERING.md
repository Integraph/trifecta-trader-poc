# Cursor Task 006: Prompt Engineering for Local Model Quality + Multi-Ticker Validation

## Objective
Close the data grounding gap between local models and Claude by engineering better prompts, then validate the improved hybrid pipeline across multiple tickers.

## Context
- Task 005 found that the quality gap is NOT about model size — qwen2.5:32b performed the same as 14B
- The gap is in **data grounding**: Claude cites 28 specific financial metrics, Qwen 14B cites only 7-8
- The vendor agent prompts never explicitly require data citations — they say "detailed analysis" but don't say "cite exact numbers"
- Claude does this naturally; smaller models need explicit instructions
- All vendor agent prompts are **hardcoded** in create_* factory functions with **no custom prompt parameter**
- Our `HybridGraphSetup` already creates each agent manually, giving us a clean injection point

### Key Finding from Vendor Code Analysis

The vendor prompts for reasoning agents (bull/bear researchers, debaters, trader) say things like:
- "Build an evidence-based case" (bull_researcher)
- "Present well-reasoned argument" (bear_researcher)
- "Provide specific recommendation" (trader)

But they NEVER say:
- "Cite exact dollar values, percentages, and ratios from the data"
- "Reference at least N specific data points"
- "Include stop-loss price, price target, and position size with exact numbers"

This is why Claude (which naturally cites data) scores 28 data points while Qwen 14B (which needs explicit instructions) scores only 7-8.

## Important Rules
- **DO NOT modify files in `vendor/TradingAgents/` directly**
- All new code goes in `src/` and `tests/`
- Phase 1 uses ONLY isolated reasoning tests (cheap, fast)
- Phase 2 modifies the hybrid graph infrastructure
- Phase 3 runs full pipeline (costs money — only 3 tickers)

---

## Phase 1: Prompt Engineering in Isolated Tests

### Step 1: Create Enhanced Prompt Variants

### Goal
Test whether explicit data citation instructions close the quality gap for qwen2.5:14b.

### Implementation

Create `tests/test_prompt_engineering.py`:

```python
"""
Prompt engineering experiments for local model quality improvement.

Tests whether adding explicit data citation requirements to prompts
closes the quality gap between local models and Claude.

Run with:
    pytest tests/test_prompt_engineering.py -v -s --run-comparison
"""

import pytest
import os
import re
import time

from dotenv import load_dotenv
load_dotenv()


def _ollama_available():
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


skip_unless_comparison = pytest.mark.skipif(
    "not config.getoption('--run-comparison')",
    reason="Requires --run-comparison flag",
)


# ============================================================
# MARKET DATA (same as test_reasoning_comparison.py)
# ============================================================

SAMPLE_MARKET_DATA = """
=== MARKET DATA FOR AAPL (2026-02-27) ===
Current Price: $270.15
52-Week High: $295.80
52-Week Low: $218.42
P/E Ratio: 33.2x
Forward P/E: 28.5x
Revenue (TTM): $412B
Revenue Growth: 8.2% YoY
Net Income (TTM): $105B
EPS: $6.85
Free Cash Flow: $110B
Debt/Equity: 1.45
RSI (14-day): 62.3
MACD: Bullish crossover 3 days ago
50-day SMA: $265.20
200-day SMA: $252.80
Volume: 48M (vs 55M avg)
Insider Activity: Tim Cook sold $25M in shares (planned sale)
Recent News: Vision Pro 2 announcement next month, EU DMA compliance costs $2B
"""


# ============================================================
# PROMPT VARIANTS — testing different instruction styles
# ============================================================

# --- VARIANT A: Original (baseline, no citation requirement) ---

BULL_ORIGINAL = f"""You are a bull-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BULL case for buying AAPL. Include:
1. Specific data points from the market data above
2. Growth catalysts
3. Valuation justification
4. Risk mitigation factors

Keep your response under 400 words. Be specific with numbers."""


# --- VARIANT B: Explicit citation count requirement ---

BULL_VARIANT_B = f"""You are a bull-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BULL case for buying AAPL.

CRITICAL REQUIREMENTS:
- You MUST cite at least 10 specific data points from the market data above, using their exact values (dollar amounts, percentages, ratios)
- Every claim must be backed by a specific number from the data
- Include growth catalysts with quantified impact estimates
- Include valuation justification using P/E, Forward P/E, and EPS
- Include risk mitigation using stop-loss levels and price targets with exact dollar values

Keep your response under 400 words. Prioritize data density over narrative."""


# --- VARIANT C: Structured output format ---

BULL_VARIANT_C = f"""You are a bull-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BULL case for buying AAPL.

Structure your response EXACTLY as follows:

KEY METRICS (cite at least 8 specific values from the data):
[List the most relevant data points with exact values]

GROWTH CATALYSTS:
[2-3 catalysts, each with a quantified impact estimate]

VALUATION CASE:
[Why the current P/E of 33.2x and Forward P/E of 28.5x are justified, using specific comparisons]

RISK/REWARD:
[Entry price, stop-loss with exact dollar value, price target with exact dollar value, risk/reward ratio]

Keep your response under 400 words."""


# --- VARIANT D: Few-shot example style ---

BULL_VARIANT_D = f"""You are a bull-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BULL case for buying AAPL.

Here is an example of the LEVEL OF SPECIFICITY expected (for a different stock):

"MSFT trades at $420 with a Forward P/E of 32.1x, justified by 15.3% revenue growth and $72B FCF.
The RSI at 58 and price 4.2% above the 50-day SMA ($403.20) suggest momentum without overextension.
Entry at $420, stop-loss at $395 (-5.9%), target $465 (+10.7%), risk/reward 1.8:1.
Position size: 4% of portfolio with scale-in at $405 (50-day SMA test)."

Now produce a similarly data-dense analysis for AAPL using the market data provided above.
Cite at least 10 specific numbers. Include stop-loss, price target, and position size with exact values.

Keep your response under 400 words."""


# --- Trader prompt variants ---

TRADER_ORIGINAL = f"""You are a portfolio trader making a final trading decision.

{SAMPLE_MARKET_DATA}

Bull Argument: Strong services growth at 15% YoY, Vision Pro 2 catalyst, massive $110B free cash flow supports buybacks and dividends.

Bear Argument: Elevated 33x P/E, insider selling, EU regulatory costs of $2B, volume below average suggesting institutional caution.

Based on the bull and bear arguments and the market data, make your final trading decision.
Your response MUST end with a line in this exact format:
FINAL TRANSACTION PROPOSAL: <BUY|HOLD|SELL>

Include specific risk management parameters (stop-loss, price target, position size)."""


TRADER_VARIANT_B = f"""You are a portfolio trader making a final trading decision.

{SAMPLE_MARKET_DATA}

Bull Argument: Strong services growth at 15% YoY, Vision Pro 2 catalyst, massive $110B free cash flow supports buybacks and dividends.

Bear Argument: Elevated 33x P/E, insider selling, EU regulatory costs of $2B, volume below average suggesting institutional caution.

Based on the bull and bear arguments and the market data, make your final trading decision.

CRITICAL REQUIREMENTS FOR YOUR RESPONSE:
1. Cite at least 12 specific data points from the market data (exact dollar values, percentages, ratios)
2. Calculate a specific risk/reward ratio using exact entry price, stop-loss, and target
3. Specify exact position sizing as a percentage of portfolio
4. Reference technical levels (RSI, MACD, SMAs) with their exact values
5. End with: FINAL TRANSACTION PROPOSAL: <BUY|HOLD|SELL>

Keep your response under 500 words. Every sentence should contain at least one specific number."""


TRADER_VARIANT_C = f"""You are a portfolio trader making a final trading decision.

{SAMPLE_MARKET_DATA}

Bull Argument: Strong services growth at 15% YoY, Vision Pro 2 catalyst, massive $110B free cash flow supports buybacks and dividends.

Bear Argument: Elevated 33x P/E, insider selling, EU regulatory costs of $2B, volume below average suggesting institutional caution.

Analyze both arguments against the market data and produce a trading decision.

Your response MUST follow this EXACT structure:

QUANTITATIVE SUMMARY:
- Current Price: [from data]
- Valuation: P/E [from data], Forward P/E [from data], EPS [from data]
- Technical: RSI [from data], 50-day SMA [from data], 200-day SMA [from data]
- Momentum: MACD [from data], Volume vs Average [from data]

RISK/REWARD CALCULATION:
- Entry: $[exact price]
- Stop-Loss: $[exact price] ([percentage]% risk)
- Target: $[exact price] ([percentage]% upside)
- Risk/Reward Ratio: [calculated ratio]

POSITION SIZING:
- Allocation: [percentage]% of portfolio
- Scaling strategy: [describe]

DECISION RATIONALE:
[2-3 sentences citing specific data points]

FINAL TRANSACTION PROPOSAL: <BUY|HOLD|SELL>"""


TRADER_VARIANT_D = f"""You are a portfolio trader making a final trading decision.

{SAMPLE_MARKET_DATA}

Bull Argument: Strong services growth at 15% YoY, Vision Pro 2 catalyst, massive $110B free cash flow supports buybacks and dividends.

Bear Argument: Elevated 33x P/E, insider selling, EU regulatory costs of $2B, volume below average suggesting institutional caution.

Here is an example of the LEVEL OF SPECIFICITY expected (for a different stock):

"MSFT at $420: Forward P/E 32.1x on $72B FCF and 15.3% revenue growth. RSI 58 shows momentum
without overextension. MACD bullish, price 4.2% above 50-day SMA ($403.20) and 12.1% above
200-day SMA ($374.50). Volume at 28M vs 32M avg suggests slight distribution.
Entry: $420. Stop-loss: $395 (-5.9% risk). Target: $465 (+10.7%). R/R: 1.81:1.
Position: 4% portfolio, scale-in 2% now, 2% at $405 retest.
FINAL TRANSACTION PROPOSAL: BUY"

Now produce a similarly data-dense decision for AAPL. Cite at least 12 specific numbers.

FINAL TRANSACTION PROPOSAL: <BUY|HOLD|SELL>"""


# ============================================================
# TEST INFRASTRUCTURE
# ============================================================

PROMPT_VARIANTS = {
    "bull": {
        "original": BULL_ORIGINAL,
        "variant_b": BULL_VARIANT_B,
        "variant_c": BULL_VARIANT_C,
        "variant_d": BULL_VARIANT_D,
    },
    "trader": {
        "original": TRADER_ORIGINAL,
        "variant_b": TRADER_VARIANT_B,
        "variant_c": TRADER_VARIANT_C,
        "variant_d": TRADER_VARIANT_D,
    },
}


def _invoke_model(provider: str, model: str, prompt: str):
    """Invoke a model and return (response_text, elapsed_seconds)."""
    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    start = time.time()
    result = llm.invoke(prompt)
    elapsed = time.time() - start

    return result.content, elapsed


def _score_response(response: str) -> dict:
    """Score a response for data grounding quality."""
    numbers = re.findall(r'\$[\d,.]+|\d+\.?\d*%|\d+\.\d+x', response)
    has_stop_loss = bool(re.search(r'stop.?loss', response, re.IGNORECASE))
    has_price_target = bool(re.search(r'(?:price\s+)?target|upside', response, re.IGNORECASE))
    has_position_sizing = bool(re.search(r'position\s+siz|allocation|portfolio', response, re.IGNORECASE))
    has_risk_reward = bool(re.search(r'risk.?reward|r/?r\s*(?:ratio)?:?\s*\d', response, re.IGNORECASE))

    # Check for specific AAPL data references
    aapl_values = ["270.15", "295.80", "218.42", "33.2", "28.5", "412", "8.2",
                   "105", "6.85", "110", "1.45", "62.3", "265.20", "252.80", "48"]
    data_hits = sum(1 for v in aapl_values if v in response)

    return {
        "word_count": len(response.split()),
        "char_count": len(response),
        "numbers_cited": len(numbers),
        "aapl_data_hits": data_hits,
        "has_stop_loss": has_stop_loss,
        "has_price_target": has_price_target,
        "has_position_sizing": has_position_sizing,
        "has_risk_reward": has_risk_reward,
        "risk_param_count": sum([has_stop_loss, has_price_target, has_position_sizing, has_risk_reward]),
    }


# ============================================================
# TESTS
# ============================================================

@pytest.fixture(scope="module")
def prompt_experiment_results(request):
    """Run all prompt variants against qwen2.5:14b and Claude."""
    if not request.config.getoption("--run-comparison"):
        pytest.skip("Requires --run-comparison flag")

    models = []

    if _ollama_available():
        models.append({"name": "qwen2.5:14b", "provider": "ollama", "model": "qwen2.5:14b"})

    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append({"name": "claude-sonnet-4.5", "provider": "anthropic", "model": "claude-sonnet-4-5-20250929"})

    if not models:
        pytest.skip("No models available")

    results = {}

    for model_cfg in models:
        model_name = model_cfg["name"]
        results[model_name] = {}

        for prompt_type in ["bull", "trader"]:
            results[model_name][prompt_type] = {}
            for variant_name, prompt in PROMPT_VARIANTS[prompt_type].items():
                try:
                    response, elapsed = _invoke_model(
                        model_cfg["provider"], model_cfg["model"], prompt
                    )
                    score = _score_response(response)
                    results[model_name][prompt_type][variant_name] = {
                        "response": response,
                        "elapsed": elapsed,
                        "score": score,
                    }
                    print(
                        f"  {model_name} / {prompt_type} / {variant_name}: "
                        f"{elapsed:.1f}s, {score['numbers_cited']} numbers, "
                        f"{score['aapl_data_hits']} AAPL refs"
                    )
                except Exception as e:
                    results[model_name][prompt_type][variant_name] = {
                        "response": f"ERROR: {e}",
                        "elapsed": 0,
                        "score": {},
                    }
                    print(f"  {model_name} / {prompt_type} / {variant_name}: ERROR: {e}")

    return results


@skip_unless_comparison
def test_prompt_variant_comparison(prompt_experiment_results):
    """Compare all prompt variants across models."""
    print("\n" + "=" * 100)
    print("PROMPT ENGINEERING EXPERIMENT — DATA GROUNDING COMPARISON")
    print("=" * 100)

    for prompt_type in ["bull", "trader"]:
        print(f"\n{'='*50} {prompt_type.upper()} PROMPT {'='*50}")
        header = (
            f"{'Model':<20} {'Variant':<12} {'Words':<7} {'Numbers':<9} "
            f"{'AAPL Refs':<10} {'StopLoss':<9} {'Target':<8} {'Sizing':<8} "
            f"{'R/R':<5} {'Time':<7}"
        )
        print(header)
        print("-" * 100)

        for model_name, model_results in prompt_experiment_results.items():
            if prompt_type not in model_results:
                continue
            for variant_name, data in model_results[prompt_type].items():
                if "ERROR" in str(data.get("response", "")):
                    print(f"{model_name:<20} {variant_name:<12} ERROR")
                    continue
                s = data["score"]
                print(
                    f"{model_name:<20} {variant_name:<12} "
                    f"{s.get('word_count', 0):<7} "
                    f"{s.get('numbers_cited', 0):<9} "
                    f"{s.get('aapl_data_hits', 0):<10} "
                    f"{'Yes' if s.get('has_stop_loss') else 'No':<9} "
                    f"{'Yes' if s.get('has_price_target') else 'No':<8} "
                    f"{'Yes' if s.get('has_position_sizing') else 'No':<8} "
                    f"{'Yes' if s.get('has_risk_reward') else 'No':<5} "
                    f"{data.get('elapsed', 0):<7.1f}"
                )

        print("-" * 100)


@skip_unless_comparison
def test_best_variant_vs_claude(prompt_experiment_results):
    """Identify which variant closes the gap with Claude most."""
    print("\n" + "=" * 100)
    print("GAP ANALYSIS: Best Local Variant vs Claude Baseline")
    print("=" * 100)

    for prompt_type in ["bull", "trader"]:
        # Get Claude baseline (original prompt)
        claude_data = (
            prompt_experiment_results
            .get("claude-sonnet-4.5", {})
            .get(prompt_type, {})
            .get("original", {})
        )
        claude_score = claude_data.get("score", {})
        claude_numbers = claude_score.get("numbers_cited", 0)
        claude_aapl = claude_score.get("aapl_data_hits", 0)

        if not claude_score:
            print(f"\n{prompt_type}: No Claude baseline available (API key not set?)")
            continue

        print(f"\n--- {prompt_type.upper()} ---")
        print(f"Claude baseline (original prompt): {claude_numbers} numbers, {claude_aapl} AAPL refs")

        # Find best Qwen variant
        qwen_results = (
            prompt_experiment_results
            .get("qwen2.5:14b", {})
            .get(prompt_type, {})
        )

        best_variant = None
        best_numbers = 0
        for variant_name, data in qwen_results.items():
            s = data.get("score", {})
            nums = s.get("numbers_cited", 0)
            if nums > best_numbers:
                best_numbers = nums
                best_variant = variant_name

        if best_variant:
            best_score = qwen_results[best_variant]["score"]
            print(f"Best Qwen variant: {best_variant}")
            print(f"  Numbers cited: {best_score.get('numbers_cited', 0)} (Claude: {claude_numbers})")
            print(f"  AAPL data refs: {best_score.get('aapl_data_hits', 0)} (Claude: {claude_aapl})")
            print(f"  Risk params: {best_score.get('risk_param_count', 0)}/4")

            gap_pct = (best_numbers / claude_numbers * 100) if claude_numbers > 0 else 0
            print(f"  Gap closure: {gap_pct:.0f}% of Claude's data density")

            # Print the best variant's last 500 chars for manual inspection
            print(f"\n  --- Best variant response (last 500 chars) ---")
            response = qwen_results[best_variant]["response"]
            print(f"  {response[-500:]}")


@skip_unless_comparison
def test_trader_decision_across_variants(prompt_experiment_results):
    """Verify all trader prompt variants produce parseable decisions."""
    from src.signal_processing import extract_decision

    print("\n" + "=" * 80)
    print("TRADER DECISION EXTRACTION ACROSS VARIANTS")
    print("=" * 80)

    all_pass = True
    for model_name, model_results in prompt_experiment_results.items():
        trader_results = model_results.get("trader", {})
        for variant_name, data in trader_results.items():
            response = data.get("response", "")
            if response.startswith("ERROR"):
                continue
            decision = extract_decision(response)
            status = "PASS" if decision != "UNKNOWN" else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {model_name:<20} {variant_name:<12} -> {decision:<6} [{status}]")

    assert all_pass, "Some variants produced unparseable decisions"
```

### Run

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

pytest tests/test_prompt_engineering.py -v -s --run-comparison 2>&1 | tee results/prompt_engineering_results.txt
```

### What to Record
- Which variant produces the most data citations for qwen2.5:14b
- Whether any variant matches or approaches Claude's 28 data points
- Whether structured output format (variant C) or few-shot (variant D) works better
- Whether all variants still produce parseable decisions
- Timing differences between variants

### Quality Gate
If the best variant gets qwen2.5:14b to cite ≥ 18 data points (64% of Claude's 28), proceed to Phase 2. If no variant exceeds 12 data points, try additional variants before proceeding.

---

## Phase 2: Integrate Best Prompts into Pipeline

### Step 2: Create Enhanced LLM Wrapper

### Goal
Wrap the local LLM with a system-level instruction prefix that improves data grounding, without modifying any vendor agent code.

### Why a Wrapper LLM?
The vendor agents call `llm.invoke(prompt)` with hardcoded prompts. We can't change those prompts, but we CAN wrap the LLM so that every invocation receives additional instructions. This is the cleanest approach — zero vendor code changes.

### Implementation

Create `src/enhanced_llm.py`:

```python
"""
Enhanced LLM wrapper that prepends data citation instructions to local model calls.

The vendor agent prompts don't explicitly require data citations — they rely on
the model's natural tendency to cite data. Claude does this automatically; smaller
models need explicit instructions. This wrapper bridges that gap.

Usage:
    from src.enhanced_llm import create_enhanced_llm

    base_llm = ChatOpenAI(model="qwen2.5:14b", ...)
    enhanced_llm = create_enhanced_llm(base_llm, style="financial_analysis")
"""

import logging
from typing import Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


# The enhancement prefix — determined by Phase 1 experiments.
# This should be updated to match whichever variant (B, C, or D) won.
# Starting with the best general-purpose enhancement:

FINANCIAL_ANALYSIS_PREFIX = """IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:
- You MUST cite at least 10 specific data points using their exact values (dollar amounts, percentages, ratios, multiples)
- Every analytical claim must reference a specific number from the data provided
- When discussing risk management, always include:
  * Exact stop-loss price with percentage risk from entry
  * Exact price target with percentage upside from entry
  * Risk/reward ratio (calculated from stop and target)
  * Position size as percentage of portfolio
- When discussing valuation, cite P/E, Forward P/E, EPS, and revenue figures with exact values
- When discussing technicals, cite RSI, MACD, SMA levels with exact values
- Prioritize data density: every sentence should contain at least one specific number
"""

# Alternative: structured output style (use if variant C wins)
STRUCTURED_OUTPUT_PREFIX = """IMPORTANT: Structure your analysis with specific data points.
Always include exact dollar values, percentages, and ratios from the data.
For trading decisions, always calculate and state:
- Entry price (exact), Stop-loss (exact price and % risk), Target (exact price and % upside)
- Risk/reward ratio, Position sizing (% of portfolio)
Reference at least 10 specific numbers from the provided data.
"""

# Alternative: few-shot style (use if variant D wins)
FEW_SHOT_PREFIX = """When analyzing financial data, be extremely specific with numbers.
For example, instead of "the stock is near its moving average", say
"price at $270.15 is 1.9% above the 50-day SMA of $265.20 and 6.8% above the 200-day SMA of $252.80."
Always calculate exact percentages, risk/reward ratios, and position sizes.
Cite at least 10 specific data points from the data provided.
"""

ENHANCEMENT_STYLES = {
    "financial_analysis": FINANCIAL_ANALYSIS_PREFIX,
    "structured": STRUCTURED_OUTPUT_PREFIX,
    "few_shot": FEW_SHOT_PREFIX,
}


class EnhancedChatModel:
    """Wraps a LangChain chat model to prepend enhancement instructions.

    This wrapper intercepts calls to invoke() and prepends a system-level
    instruction that improves data citation quality for local models.

    It handles both string and message-list invocations used by different
    vendor agents.
    """

    def __init__(self, base_llm: BaseChatModel, prefix: str):
        self._base_llm = base_llm
        self._prefix = prefix
        # Proxy all attributes to base LLM (needed for bind_tools, etc.)
        # Note: we do NOT enhance tool-calling agents — only reasoning agents

    def invoke(self, input: Any, **kwargs) -> Any:
        """Intercept invoke calls and prepend enhancement instructions."""
        if isinstance(input, str):
            # Plain string prompt (bull/bear researchers, debaters, risk manager)
            enhanced = f"{self._prefix}\n\n{input}"
            return self._base_llm.invoke(enhanced, **kwargs)
        elif isinstance(input, list):
            # Message list (trader uses this format)
            # Prepend as a system message
            enhanced_messages = [SystemMessage(content=self._prefix)] + [
                m if isinstance(m, BaseMessage) else
                (SystemMessage(content=m["content"]) if m.get("role") == "system"
                 else HumanMessage(content=m["content"]))
                for m in input
            ]
            return self._base_llm.invoke(enhanced_messages, **kwargs)
        else:
            # Unknown format — pass through unmodified
            logger.warning("EnhancedChatModel: unknown input type %s, passing through", type(input))
            return self._base_llm.invoke(input, **kwargs)

    def bind_tools(self, *args, **kwargs):
        """Pass through to base LLM — tool binding should NOT be enhanced."""
        return self._base_llm.bind_tools(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy all other attributes to the base LLM."""
        return getattr(self._base_llm, name)


def create_enhanced_llm(
    base_llm: BaseChatModel,
    style: str = "financial_analysis",
) -> EnhancedChatModel:
    """Create an enhanced LLM wrapper.

    Args:
        base_llm: The base LangChain chat model to wrap
        style: Enhancement style — one of 'financial_analysis', 'structured', 'few_shot'

    Returns:
        EnhancedChatModel that prepends instructions to every invocation
    """
    if style not in ENHANCEMENT_STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(ENHANCEMENT_STYLES.keys())}")

    prefix = ENHANCEMENT_STYLES[style]
    logger.info("Creating enhanced LLM wrapper with style '%s'", style)
    return EnhancedChatModel(base_llm, prefix)
```

### Step 3: Integrate Enhanced LLM into Hybrid Graph

Update `src/hybrid_llm.py` to add an `enhanced` flag:

```python
# Add to HybridLLMConfig:
class HybridLLMConfig:
    def __init__(
        self,
        ...
        enhance_local: bool = False,
        enhance_style: str = "financial_analysis",
    ):
        ...
        self.enhance_local = enhance_local
        self.enhance_style = enhance_style
```

Update `create_hybrid_llms()` in `src/hybrid_llm.py`:

```python
def create_hybrid_llms(hybrid_config: HybridLLMConfig) -> Dict[str, Any]:
    ...
    # After creating reasoning_quick_llm:
    if hybrid_config.enhance_local and hybrid_config.reasoning_quick_provider == "ollama":
        from src.enhanced_llm import create_enhanced_llm
        llms["reasoning_quick_llm"] = create_enhanced_llm(
            llms["reasoning_quick_llm"],
            style=hybrid_config.enhance_style,
        )
        logger.info("Enhanced local reasoning LLM with style '%s'", hybrid_config.enhance_style)

    # Same for reasoning_deep if it's local:
    if hybrid_config.enhance_local and hybrid_config.reasoning_deep_provider == "ollama":
        from src.enhanced_llm import create_enhanced_llm
        llms["reasoning_deep_llm"] = create_enhanced_llm(
            llms["reasoning_deep_llm"],
            style=hybrid_config.enhance_style,
        )
    ...
```

Add new enhanced hybrid configs to the CONFIGS dict:

```python
    "hybrid_qwen_enhanced": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=True,
        enhance_style="financial_analysis",  # UPDATE after Phase 1 results
    ),
```

Update `src/run_analysis.py` to include `hybrid_qwen_enhanced` in the `--hybrid` choices.

### Step 4: Tests for Enhanced LLM

Create `tests/test_enhanced_llm.py`:

```python
"""Tests for the enhanced LLM wrapper."""

import pytest
from unittest.mock import MagicMock, patch
from src.enhanced_llm import EnhancedChatModel, create_enhanced_llm, ENHANCEMENT_STYLES


class TestEnhancedChatModel:

    def test_string_prompt_gets_prefix(self):
        """String prompts should have the prefix prepended."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="test response")

        enhanced = EnhancedChatModel(mock_llm, "PREFIX INSTRUCTIONS")
        enhanced.invoke("Analyze this stock")

        # Verify the prompt was enhanced
        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args.startswith("PREFIX INSTRUCTIONS")
        assert "Analyze this stock" in call_args

    def test_list_prompt_gets_system_message(self):
        """Message list prompts should get a prepended system message."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="test response")

        enhanced = EnhancedChatModel(mock_llm, "PREFIX INSTRUCTIONS")
        enhanced.invoke([
            {"role": "system", "content": "You are a trader"},
            {"role": "user", "content": "What should I do?"},
        ])

        # Verify system message was prepended
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 3  # prefix + system + user
        assert "PREFIX INSTRUCTIONS" in call_args[0].content

    def test_bind_tools_passes_through(self):
        """bind_tools should NOT be enhanced — passes through to base LLM."""
        mock_llm = MagicMock()
        enhanced = EnhancedChatModel(mock_llm, "PREFIX")
        enhanced.bind_tools(["tool1", "tool2"])
        mock_llm.bind_tools.assert_called_once_with(["tool1", "tool2"])

    def test_create_enhanced_llm_valid_style(self):
        """Factory function with valid style should work."""
        mock_llm = MagicMock()
        enhanced = create_enhanced_llm(mock_llm, style="financial_analysis")
        assert isinstance(enhanced, EnhancedChatModel)

    def test_create_enhanced_llm_invalid_style(self):
        """Factory function with invalid style should raise."""
        mock_llm = MagicMock()
        with pytest.raises(ValueError, match="Unknown style"):
            create_enhanced_llm(mock_llm, style="nonexistent")

    def test_all_styles_exist(self):
        """All documented styles should be in ENHANCEMENT_STYLES."""
        assert "financial_analysis" in ENHANCEMENT_STYLES
        assert "structured" in ENHANCEMENT_STYLES
        assert "few_shot" in ENHANCEMENT_STYLES

    def test_prefix_contains_data_requirements(self):
        """All prefixes should mention data citation requirements."""
        for style, prefix in ENHANCEMENT_STYLES.items():
            assert "10" in prefix or "data point" in prefix.lower(), (
                f"Style '{style}' doesn't mention data citation count"
            )
```

### Run

```bash
pytest tests/test_enhanced_llm.py -v
```

---

## Phase 3: Multi-Ticker Validation

### Step 5: Run Enhanced Hybrid Pipeline on Multiple Tickers

### Goal
Validate the enhanced hybrid pipeline across 3 different tickers to confirm the quality improvement is consistent.

### Tickers to Test
- **AAPL** — Large-cap tech, well-covered (comparison baseline exists)
- **TSLA** — High-volatility, sentiment-driven (tests bear/bull extremes)
- **JPM** — Financial sector, fundamentals-heavy (tests data grounding)

### Commands

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

# Run 1: AAPL with enhanced hybrid (compare against existing all_cloud and hybrid_qwen results)
python -m src.run_analysis --ticker AAPL --date 2026-02-27 --hybrid hybrid_qwen_enhanced --no-debug \
    2>&1 | tee results/AAPL/run_hybrid_qwen_enhanced.log

# Run 2: TSLA with enhanced hybrid
python -m src.run_analysis --ticker TSLA --date 2026-02-27 --hybrid hybrid_qwen_enhanced --no-debug \
    2>&1 | tee results/TSLA/run_hybrid_qwen_enhanced.log

# Run 3: JPM with enhanced hybrid
python -m src.run_analysis --ticker JPM --date 2026-02-27 --hybrid hybrid_qwen_enhanced --no-debug \
    2>&1 | tee results/JPM/run_hybrid_qwen_enhanced.log
```

### After Each Run

Compare results:

```bash
# AAPL: compare all three configs
python -m src.compare_runs \
    results/AAPL/analysis_2026-02-27_all_cloud.json \
    results/AAPL/analysis_2026-02-27_hybrid_qwen.json \
    results/AAPL/analysis_2026-02-27_hybrid_qwen_enhanced.json

# TSLA and JPM: just the enhanced result (no baseline for these yet)
python -m src.compare_runs results/TSLA/analysis_2026-02-27_hybrid_qwen_enhanced.json
python -m src.compare_runs results/JPM/analysis_2026-02-27_hybrid_qwen_enhanced.json
```

---

## Step 6: Final Verification

### Run All Unit Tests

```bash
pytest tests/ -v --ignore=tests/test_reasoning_comparison.py --ignore=tests/test_prompt_engineering.py \
    2>&1 | tee results/task_006_unit_tests.txt
```

### Commit

```bash
git add .
git commit -m "Add prompt engineering for local model quality and multi-ticker validation"
git push
```

---

## Execution Order (IMPORTANT)

**This task has a natural dependency chain. Follow this order:**

1. **Phase 1, Step 1**: Run prompt engineering tests → identify best variant
2. **Phase 2, Step 2**: Update `FINANCIAL_ANALYSIS_PREFIX` in `enhanced_llm.py` based on Phase 1 winner
3. **Phase 2, Step 3**: Wire enhanced LLM into hybrid graph
4. **Phase 2, Step 4**: Run unit tests for enhanced LLM
5. **Phase 3, Step 5**: Run 3 full pipeline runs (this is the expensive part — only after everything else works)
6. **Step 6**: Final verification and commit

**If Phase 1 shows no variant significantly improves data grounding**, document the finding and skip Phase 3. No point running expensive pipeline tests with prompts that don't help.

---

## Verification Checklist

- [ ] Phase 1: At least 4 prompt variants tested against qwen2.5:14b
- [ ] Phase 1: Best variant identified with data grounding metrics
- [ ] Phase 1: All variants produce parseable trader decisions
- [ ] Phase 2: `src/enhanced_llm.py` created with LLM wrapper
- [ ] Phase 2: `src/hybrid_llm.py` updated with `enhance_local` flag and new config
- [ ] Phase 2: `src/run_analysis.py` updated with `hybrid_qwen_enhanced` choice
- [ ] Phase 2: `tests/test_enhanced_llm.py` created and passing
- [ ] Phase 3: 3 full pipeline runs completed (AAPL, TSLA, JPM)
- [ ] Phase 3: Comparison report generated for AAPL (3-way: all_cloud vs hybrid_qwen vs hybrid_qwen_enhanced)
- [ ] Phase 3: Quality scores recorded for all 3 tickers
- [ ] All unit tests pass
- [ ] No vendor code modified
- [ ] Changes committed and pushed

---

## Report

After completing all steps, create `docs/TASK_006_REPORT.md` containing:

1. **Phase 1 Results**: Prompt variant comparison table showing data grounding improvement
2. **Winning variant**: Which prompt style works best and why
3. **Phase 2**: Enhanced LLM wrapper implementation notes
4. **Phase 3 Results**: Multi-ticker comparison table:
   | Ticker | Config | Decision | Composite Score | Reasoning Depth | Data Grounding | Risk Awareness | Time |
5. **AAPL 3-way comparison**: all_cloud vs hybrid_qwen vs hybrid_qwen_enhanced
6. **Quality improvement**: Did prompt engineering close the gap? By how much?
7. **Cost analysis**: Estimated API cost per run for enhanced hybrid vs all-cloud
8. **Recommendation**: Is `hybrid_qwen_enhanced` ready to be the default config?
9. **Git log**: `git log --oneline`
10. **Next steps**: Ready for Phase B (Alpaca paper trading)?
