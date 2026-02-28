"""
Trifecta Trader POC - Main Analysis Runner

Usage:
    python -m src.run_analysis --ticker AAPL --date 2026-02-27
    python -m src.run_analysis --ticker AAPL --date 2026-02-27 --provider ollama
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import date, datetime

# Add vendor TradingAgents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def get_config(provider: str = "anthropic", deep_model: str = None, quick_model: str = None) -> dict:
    """Build configuration for the trading agents pipeline.

    Args:
        provider: LLM provider - 'anthropic', 'openai', 'google', 'ollama'
        deep_model: Model for complex reasoning tasks
        quick_model: Model for quick analysis tasks

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider

    # Set models based on provider
    model_defaults = {
        "anthropic": ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        "openai": ("gpt-5.2", "gpt-5-mini"),
        "google": ("gemini-2.0-flash", "gemini-2.0-flash"),
        "ollama": ("llama3.1:8b", "mistral:7b"),
    }

    defaults = model_defaults.get(provider, model_defaults["anthropic"])
    config["deep_think_llm"] = deep_model or defaults[0]
    config["quick_think_llm"] = quick_model or defaults[1]

    # Pipeline settings
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["max_recur_limit"] = 100

    # Data vendor settings (all yfinance for POC)
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    # Override results directory to our project
    config["results_dir"] = str(Path(__file__).parent.parent / "results")

    return config


def run_analysis(ticker: str, trade_date: str, provider: str = "anthropic",
                 deep_model: str = None, quick_model: str = None,
                 debug: bool = True) -> dict:
    """Run the full trading agents analysis pipeline.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        trade_date: Date for analysis (e.g., '2026-02-27')
        provider: LLM provider
        deep_model: Override for deep thinking model
        quick_model: Override for quick thinking model
        debug: Enable debug output

    Returns:
        Dictionary with analysis results and decision
    """
    config = get_config(provider, deep_model, quick_model)

    print(f"\n{'='*60}")
    print(f"Trifecta Trader POC - Analysis Run")
    print(f"{'='*60}")
    print(f"Ticker:    {ticker}")
    print(f"Date:      {trade_date}")
    print(f"Provider:  {provider}")
    print(f"Deep LLM:  {config['deep_think_llm']}")
    print(f"Quick LLM: {config['quick_think_llm']}")
    print(f"{'='*60}\n")

    # Initialize and run
    ta = TradingAgentsGraph(debug=debug, config=config)
    final_state, decision = ta.propagate(ticker, trade_date)

    # Save results
    results_dir = Path(config["results_dir"]) / ticker
    results_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ticker": ticker,
        "trade_date": trade_date,
        "provider": provider,
        "deep_model": config["deep_think_llm"],
        "quick_model": config["quick_think_llm"],
        "decision": decision,
        "run_timestamp": datetime.now().isoformat(),
    }

    result_file = results_dir / f"analysis_{trade_date}_{provider}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DECISION: {decision}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Trifecta Trader POC - Run trading analysis")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--date", type=str, default=str(date.today()), help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "openai", "google", "ollama"],
                        help="LLM provider")
    parser.add_argument("--deep-model", type=str, default=None, help="Override deep thinking model")
    parser.add_argument("--quick-model", type=str, default=None, help="Override quick thinking model")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")

    args = parser.parse_args()

    run_analysis(
        ticker=args.ticker,
        trade_date=args.date,
        provider=args.provider,
        deep_model=args.deep_model,
        quick_model=args.quick_model,
        debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
