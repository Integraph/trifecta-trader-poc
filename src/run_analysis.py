"""
Trifecta Trader POC - Main Analysis Runner

Usage:
    python -m src.run_analysis --ticker AAPL --date 2026-02-27
    python -m src.run_analysis --ticker AAPL --date 2026-02-27 --provider ollama
    python -m src.run_analysis --ticker AAPL --date 2026-02-27 --hybrid hybrid_qwen
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from datetime import date, datetime

# Add vendor TradingAgents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from src.signal_processing import extract_decision


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
                 debug: bool = True, hybrid: str = None) -> dict:
    """Run the full trading agents analysis pipeline.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        trade_date: Date for analysis (e.g., '2026-02-27')
        provider: LLM provider
        deep_model: Override for deep thinking model
        quick_model: Override for quick thinking model
        debug: Enable debug output
        hybrid: Hybrid LLM config name (e.g., 'hybrid_qwen')

    Returns:
        Dictionary with analysis results and decision
    """
    config = get_config(provider, deep_model, quick_model)

    print(f"\n{'='*60}")
    print(f"Trifecta Trader POC - Analysis Run")
    print(f"{'='*60}")
    print(f"Ticker:    {ticker}")
    print(f"Date:      {trade_date}")
    if hybrid:
        print(f"Mode:      HYBRID ({hybrid})")
    else:
        print(f"Provider:  {provider}")
    print(f"Deep LLM:  {config['deep_think_llm']}")
    print(f"Quick LLM: {config['quick_think_llm']}")
    print(f"{'='*60}\n")

    if hybrid:
        from src.hybrid_llm import CONFIGS
        from src.hybrid_graph import HybridTradingGraph
        hybrid_config = CONFIGS[hybrid]
        ta = HybridTradingGraph(
            hybrid_config=hybrid_config,
            debug=debug,
            config=config,
        )
        print(f"Hybrid routing: {hybrid_config.to_dict()}")
    else:
        ta = TradingAgentsGraph(debug=debug, config=config)

    start_time = time.time()
    final_state, upstream_decision = ta.propagate(ticker, trade_date)
    elapsed_seconds = time.time() - start_time

    # Override the upstream signal processing with our improved version
    final_trade_text = final_state.get("final_trade_decision", "")
    decision = extract_decision(final_trade_text)

    decision_corrected = decision != upstream_decision
    if decision_corrected:
        print(f"[signal_processing] Corrected decision: upstream='{upstream_decision}' -> ours='{decision}'")

    # Quality scoring
    from src.quality_scorer import score_pipeline_output
    config_label = hybrid if hybrid else provider
    score = score_pipeline_output(
        config_name=config_label,
        ticker=ticker,
        trade_date=trade_date,
        final_trade_decision=final_trade_text,
        extracted_decision=decision,
    )

    # Save results
    results_dir = Path(config["results_dir"]) / ticker
    results_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "ticker": ticker,
        "trade_date": trade_date,
        "provider": provider,
        "hybrid_config": hybrid,
        "deep_model": config["deep_think_llm"],
        "quick_model": config["quick_think_llm"],
        "decision": decision,
        "upstream_decision": upstream_decision,
        "decision_corrected": decision_corrected,
        "final_trade_decision_text": final_trade_text,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "run_timestamp": datetime.now().isoformat(),
        "quality_score": {
            "composite": score.composite_score,
            "reasoning_depth": score.reasoning_depth,
            "data_grounding": score.data_grounding,
            "risk_awareness": score.risk_awareness,
            "decision_consistent": score.decision_consistent,
            "has_stop_loss": score.has_stop_loss,
            "has_price_target": score.has_price_target,
            "has_position_sizing": score.has_position_sizing,
        },
    }

    result_file = results_dir / f"analysis_{trade_date}_{config_label}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DECISION: {decision}")
    print(f"\nQuality Score: {score.composite_score:.1f}/10")
    print(f"  Reasoning depth:     {score.reasoning_depth}/10")
    print(f"  Data grounding:      {score.data_grounding}/10")
    print(f"  Risk awareness:      {score.risk_awareness}/10")
    print(f"  Decision consistent: {'Yes' if score.decision_consistent else 'No'}")
    print(f"  Elapsed time:        {elapsed_seconds:.1f}s")
    print(f"\nResults saved to: {result_file}")
    print(f"{'='*60}\n")

    return result


def _run_execution_flow(result: dict, config: dict, args) -> None:
    """Run the trade execution flow after analysis completes."""
    from src.execution.trade_params import extract_trade_params

    ticker = result["ticker"]
    decision = result["decision"]
    final_trade_text = result.get("final_trade_decision_text", "")
    quality_score = result.get("quality_score", {}).get("composite", 0.0)

    trade_params = extract_trade_params(
        ticker=ticker,
        decision=decision,
        quality_score=quality_score,
        decision_text=final_trade_text,
        current_price=None,
    )

    print(f"\n{'='*60}")
    print(f"TRADE PARAMETERS")
    print(f"{'='*60}")
    print(f"  Decision:    {trade_params.decision}")
    print(f"  Stop-loss:   ${trade_params.stop_loss or 'N/A'}")
    print(f"  Target:      ${trade_params.price_target or 'N/A'}")
    print(f"  Position:    {trade_params.position_pct or 'N/A'}%")
    print(f"  R/R Ratio:   {trade_params.risk_reward_ratio or 'N/A'}")
    print(f"  Actionable:  {trade_params.is_actionable}")

    if args.dry_run:
        print(f"\n  [DRY RUN — no order submitted]")
    elif args.execute and trade_params.is_actionable:
        from src.execution.executor import TradeExecutor
        from src.execution.position_manager import PositionManager

        audit_dir = str(Path(config["results_dir"]) / "audit")
        executor = TradeExecutor(audit_dir=audit_dir)
        pm = PositionManager(executor.client)

        order = pm.calculate_order(trade_params)
        print(f"\n{'='*60}")
        print(f"ORDER CALCULATION")
        print(f"{'='*60}")
        print(f"  Side:        {order.side}")
        print(f"  Qty:         {order.qty}")
        print(f"  Value:       ${order.position_value:.0f}")
        print(f"  Portfolio %: {order.position_pct_of_portfolio:.1f}%")
        print(f"  Risk/trade:  ${order.total_risk:.0f} ({order.portfolio_risk_pct:.2f}% of portfolio)")
        print(f"  Approved:    {order.approved}")
        if not order.approved:
            print(f"  Rejections:  {order.rejection_reasons}")

        exec_result = executor.execute(order, trade_params)
        print(f"\n  Action: {exec_result['action']}")
        if exec_result.get("alpaca_order_id"):
            print(f"  Order ID: {exec_result['alpaca_order_id']}")
    elif args.execute and not trade_params.is_actionable:
        print(f"\n  [NOT ACTIONABLE — {trade_params.decision}, score={quality_score}]")


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
    parser.add_argument("--hybrid", type=str, default=None,
                        choices=["all_cloud", "hybrid_qwen", "hybrid_mistral",
                                 "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
                                 "hybrid_qwen32", "hybrid_aggressive_qwen32",
                                 "hybrid_qwen_enhanced"],
                        help="Use hybrid LLM routing config")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the trade on Alpaca paper trading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calculate order but don't submit (shows what would happen)")

    args = parser.parse_args()

    config = None
    # Capture config for execution flow
    _original_get_config = get_config

    result = run_analysis(
        ticker=args.ticker,
        trade_date=args.date,
        provider=args.provider,
        deep_model=args.deep_model,
        quick_model=args.quick_model,
        debug=not args.no_debug,
        hybrid=args.hybrid,
    )

    if args.execute or args.dry_run:
        config = get_config(args.provider, args.deep_model, args.quick_model)
        _run_execution_flow(result, config, args)


if __name__ == "__main__":
    main()
