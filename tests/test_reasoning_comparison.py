"""
Side-by-side reasoning comparison: Claude vs local models.

This test sends the same prompt to multiple models and compares output quality.
It does NOT run the full pipeline -- it tests individual agent prompts in isolation.

NOTE: This test makes real API calls. Run with:
    pytest tests/test_reasoning_comparison.py -v -s --run-comparison
"""

import pytest
import os
import re


def _ollama_available():
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


skip_unless_comparison = pytest.mark.skipif(
    "not config.getoption('--run-comparison')",
    reason="Reasoning comparison tests require --run-comparison flag",
)


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

BULL_PROMPT = f"""You are a bull-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BULL case for buying AAPL. Include:
1. Specific data points from the market data above
2. Growth catalysts
3. Valuation justification
4. Risk mitigation factors

Keep your response under 400 words. Be specific with numbers."""

BEAR_PROMPT = f"""You are a bear-case investment researcher.

{SAMPLE_MARKET_DATA}

Make the strongest possible BEAR case for selling AAPL. Include:
1. Specific data points from the market data above
2. Risk factors and headwinds
3. Valuation concerns
4. Technical warning signs

Keep your response under 400 words. Be specific with numbers."""

TRADER_PROMPT = f"""You are a portfolio trader making a final trading decision.

{SAMPLE_MARKET_DATA}

Bull Argument: Strong services growth at 15% YoY, Vision Pro 2 catalyst, massive $110B free cash flow supports buybacks and dividends.

Bear Argument: Elevated 33x P/E, insider selling, EU regulatory costs of $2B, volume below average suggesting institutional caution.

Based on the bull and bear arguments and the market data, make your final trading decision.
Your response MUST end with a line in this exact format:
FINAL TRANSACTION PROPOSAL: <BUY|HOLD|SELL>

Include specific risk management parameters (stop-loss, price target, position size)."""


def _create_model_configs():
    """Define models to compare."""
    configs = []

    if _ollama_available():
        configs.append({
            "name": "qwen2.5:14b",
            "provider": "ollama",
            "model": "qwen2.5:14b",
        })
        configs.append({
            "name": "mistral-small:22b",
            "provider": "ollama",
            "model": "mistral-small:22b",
        })

    if os.environ.get("ANTHROPIC_API_KEY"):
        configs.append({
            "name": "claude-sonnet-4.5",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        })

    return configs


def _invoke_model(provider: str, model: str, prompt: str) -> str:
    """Invoke a model and return the response text."""
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

    result = llm.invoke(prompt)
    return result.content


@pytest.fixture(scope="module")
def comparison_results(request):
    """Run all comparisons and collect results."""
    if not request.config.getoption("--run-comparison"):
        pytest.skip("Requires --run-comparison flag")

    configs = _create_model_configs()
    if not configs:
        pytest.skip("No models available for comparison")

    results = {}

    for config in configs:
        name = config["name"]
        results[name] = {}

        for prompt_name, prompt in [
            ("bull", BULL_PROMPT),
            ("bear", BEAR_PROMPT),
            ("trader", TRADER_PROMPT),
        ]:
            try:
                response = _invoke_model(config["provider"], config["model"], prompt)
                results[name][prompt_name] = response
            except Exception as e:
                results[name][prompt_name] = f"ERROR: {e}"

    return results


@skip_unless_comparison
def test_bull_case_quality(comparison_results):
    """Compare bull case reasoning across models."""
    print("\n" + "=" * 80)
    print("BULL CASE COMPARISON")
    print("=" * 80)

    for model_name, responses in comparison_results.items():
        response = responses.get("bull", "NOT RUN")
        word_count = len(response.split()) if not response.startswith("ERROR") else 0
        has_numbers = bool(re.findall(r'\$[\d,.]+|\d+\.?\d*%', response))

        print(f"\n--- {model_name} ---")
        print(f"Words: {word_count}, Has numbers: {has_numbers}")
        print(f"First 300 chars: {response[:300]}")

        if not response.startswith("ERROR"):
            assert word_count > 50, f"{model_name} bull case too short"


@skip_unless_comparison
def test_trader_decision_format(comparison_results):
    """Check that each model produces a properly formatted decision."""
    print("\n" + "=" * 80)
    print("TRADER DECISION FORMAT")
    print("=" * 80)

    from src.signal_processing import extract_decision

    for model_name, responses in comparison_results.items():
        response = responses.get("trader", "NOT RUN")
        decision = extract_decision(response) if not response.startswith("ERROR") else "ERROR"

        print(f"\n--- {model_name} ---")
        print(f"Decision: {decision}")
        print(f"Last 200 chars: {response[-200:]}")

        if not response.startswith("ERROR"):
            assert decision != "UNKNOWN", (
                f"{model_name} did not produce a parseable decision. "
                f"Response tail: {response[-300:]}"
            )
