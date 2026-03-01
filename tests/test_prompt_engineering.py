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
