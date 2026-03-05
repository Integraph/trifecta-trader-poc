"""
Trifecta Trader POC - Main Analysis Runner

Usage:
    python -m src.run_analysis --ticker AAPL --date 2026-02-27
    python -m src.run_analysis --ticker AAPL --date 2026-02-27 --provider ollama
    python -m src.run_analysis --ticker AAPL --date 2026-02-27 --hybrid hybrid_qwen
    python -m src.run_analysis --portfolio
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from datetime import date, datetime
from typing import Optional

# Add vendor TradingAgents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from src.signal_processing import extract_decision

# Safety constant imported so pre-filter can use it
from src.execution.position_manager import MAX_POSITION_PCT

MIN_BUYING_POWER = 1_000.0  # Warn when buying power drops below this


def get_config(provider: str = "anthropic", deep_model: str = None, quick_model: str = None) -> dict:
    """Build configuration for the trading agents pipeline."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider

    model_defaults = {
        "anthropic": ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        "openai":    ("gpt-5.2", "gpt-5-mini"),
        "google":    ("gemini-2.0-flash", "gemini-2.0-flash"),
        "ollama":    ("llama3.1:8b", "mistral:7b"),
    }

    defaults = model_defaults.get(provider, model_defaults["anthropic"])
    config["deep_think_llm"]  = deep_model or defaults[0]
    config["quick_think_llm"] = quick_model or defaults[1]

    config["max_debate_rounds"]      = 1
    config["max_risk_discuss_rounds"] = 1
    config["max_recur_limit"]        = 100

    config["data_vendors"] = {
        "core_stock_apis":       "yfinance",
        "technical_indicators":  "yfinance",
        "fundamental_data":      "yfinance",
        "news_data":             "yfinance",
    }

    config["results_dir"] = str(Path(__file__).parent.parent / "results")
    return config


# ── Portfolio helpers ──────────────────────────────────────────────────────────

def _make_trading_client():
    """Return an Alpaca TradingClient (paper=True, hardcoded)."""
    from alpaca.trading.client import TradingClient
    api_key    = os.environ.get("APCA_API_KEY_ID", "")
    secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


def build_portfolio_context(ticker: str) -> dict:
    """Query Alpaca and build a portfolio context dict for *ticker*.

    Returns a context dict on success; falls back to a minimal stub on error
    so the analysis pipeline is never blocked by portfolio connectivity issues.
    """
    try:
        from src.execution.position_manager import PositionManager
        client    = _make_trading_client()
        pm        = PositionManager(client)
        account   = pm.get_account_state()
        positions = pm.get_positions()
        current   = positions.get(ticker.upper())

        portfolio_allocation = {
            sym: {
                "pct":    round(pos.market_value / account.equity * 100, 1) if account.equity else 0,
                "shares": pos.qty,
            }
            for sym, pos in positions.items()
        }

        current_position = {"held": False}
        if current:
            pct = round(current.market_value / account.equity * 100, 1) if account.equity else 0
            current_position = {
                "held":           True,
                "shares":         current.qty,
                "avg_cost":       round(current.cost_basis / current.qty, 2) if current.qty else None,
                "unrealized_pnl": round(current.unrealized_pl, 2),
                "current_value":  round(current.market_value, 2),
                "portfolio_pct":  pct,
            }

        return {
            "account_equity":       round(account.equity, 2),
            "buying_power":         round(account.buying_power, 2),
            "cash":                 round(account.cash, 2),
            "total_positions":      len(positions),
            "current_position":     current_position,
            "portfolio_allocation": portfolio_allocation,
            "_source":              "alpaca",
        }

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not fetch portfolio context: %s", e)
        return {
            "account_equity":   None,
            "buying_power":     None,
            "cash":             None,
            "total_positions":  None,
            "current_position": {"held": False},
            "portfolio_allocation": {},
            "_source":          "unavailable",
            "_error":           str(e),
        }


def print_portfolio_context(ticker: str, ctx: dict) -> None:
    """Print a formatted portfolio context block before analysis."""
    if ctx.get("_source") == "unavailable":
        print(f"\n{'═'*60}")
        print("PORTFOLIO CONTEXT  (unavailable)")
        print(f"  Warning: {ctx.get('_error', 'unknown error')}")
        print(f"{'═'*60}")
        return

    pos = ctx.get("current_position", {})
    alloc = ctx.get("portfolio_allocation", {})
    alloc_str = ", ".join(
        f"{sym} {v['pct']:.0f}%" for sym, v in list(alloc.items())[:5]
    ) or "none"

    print(f"\n{'═'*60}")
    print("PORTFOLIO CONTEXT")
    print(f"{'═'*60}")
    print(f"  Account equity:   ${ctx['account_equity']:,.2f}")
    print(f"  Buying power:     ${ctx['buying_power']:,.2f}")
    print(f"  Cash:             ${ctx['cash']:,.2f}")
    print(f"  Positions:        {ctx['total_positions']} ({alloc_str})")
    print()

    if pos.get("held"):
        print(f"  {ticker}:            HELD  "
              f"{pos['shares']} shares @ avg ${pos['avg_cost']}  "
              f"P&L ${pos['unrealized_pnl']:+,.0f}  "
              f"({pos['portfolio_pct']:.1f}% of portfolio)")
    else:
        print(f"  {ticker}:            NOT HELD")
        print(f"  Recommendation context: BUY analysis only (no position to sell)")
    print(f"{'═'*60}")


def check_portfolio_warnings(ticker: str, ctx: dict, batch_mode: bool = False) -> list:
    """Evaluate pre-analysis portfolio warnings.

    These are informational only — they never block analysis.
    In batch mode warnings are always logged but never prompt the user.

    Returns list of warning strings.
    """
    warnings = []
    if ctx.get("_source") == "unavailable":
        return warnings

    pos = ctx.get("current_position", {})
    buying_power = ctx.get("buying_power", 0) or 0

    # Insufficient buying power for new positions
    if buying_power < MIN_BUYING_POWER:
        warnings.append(
            f"Insufficient buying power: ${buying_power:,.0f} "
            f"(minimum ${MIN_BUYING_POWER:,.0f} for new positions)"
        )

    # Already at max allocation for this ticker
    if pos.get("held") and pos.get("portfolio_pct", 0) >= MAX_POSITION_PCT:
        warnings.append(
            f"Already at max allocation for {ticker}: "
            f"{pos['portfolio_pct']:.1f}% (max {MAX_POSITION_PCT:.0f}%)"
        )

    # Warn about no position — SELL won't be actionable
    if not pos.get("held"):
        warnings.append(
            f"No position in {ticker}. SELL recommendations won't be actionable."
        )

    return warnings


def _print_warnings(warnings: list, ticker: str, batch_mode: bool = False) -> None:
    if not warnings:
        return
    print(f"\n  ⚠ Portfolio warnings for {ticker}:")
    for w in warnings:
        print(f"    • {w}")
    if not batch_mode:
        print()  # spacing before analysis begins


# ── Main analysis function ────────────────────────────────────────────────────

def run_analysis(ticker: str, trade_date: str, provider: str = "anthropic",
                 deep_model: str = None, quick_model: str = None,
                 debug: bool = True, hybrid: str = None,
                 use_cache: bool = True, cost_breakdown: bool = True,
                 portfolio_context: Optional[dict] = None,
                 batch_mode: bool = False) -> dict:
    """Run the full trading agents analysis pipeline.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        trade_date: Date for analysis (e.g., '2026-02-27')
        provider: LLM provider
        deep_model: Override for deep thinking model
        quick_model: Override for quick thinking model
        debug: Enable debug output
        hybrid: Hybrid LLM config name (e.g., 'hybrid_haiku_tools')
        use_cache: Enable analyst output caching (HybridTradingGraph only)
        cost_breakdown: Print cost breakdown after run
        portfolio_context: Pre-fetched portfolio context dict.
            If None, Alpaca is queried directly (single-ticker flow).
            Pass a pre-built dict from run_batch.py to avoid repeated API calls.
        batch_mode: Suppress interactive prompts (used by run_batch.py).

    Returns:
        Dictionary with analysis results and decision.
    """
    config = get_config(provider, deep_model, quick_model)

    # ── Portfolio context ──────────────────────────────────────────────────
    if portfolio_context is None:
        portfolio_context = build_portfolio_context(ticker)

    print_portfolio_context(ticker, portfolio_context)
    warnings = check_portfolio_warnings(ticker, portfolio_context, batch_mode)
    _print_warnings(warnings, ticker, batch_mode)

    # ── Analysis header ────────────────────────────────────────────────────
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
            use_cache=use_cache,
        )
        print(f"Hybrid routing: {hybrid_config.to_dict()}")
    else:
        ta = TradingAgentsGraph(debug=debug, config=config)

    start_time = time.time()
    final_state, upstream_decision = ta.propagate(ticker, trade_date)
    elapsed_seconds = time.time() - start_time

    # Collect cost breakdown (only available for HybridTradingGraph)
    cost_info = {}
    if hybrid and hasattr(ta, "cost_breakdown"):
        cost_info = ta.cost_breakdown()

    # Override upstream signal processing with our improved version
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

    trader_plan_text = final_state.get("trader_investment_plan", "")

    pos_ctx = portfolio_context.get("current_position", {})
    portfolio_summary = {
        "account_equity": portfolio_context.get("account_equity"),
        "buying_power":   portfolio_context.get("buying_power"),
        "held":           pos_ctx.get("held", False),
        "shares":         pos_ctx.get("shares", 0),
        "avg_cost":       pos_ctx.get("avg_cost"),
        "unrealized_pnl": pos_ctx.get("unrealized_pnl"),
        "portfolio_pct":  pos_ctx.get("portfolio_pct"),
    }

    result_file = results_dir / f"analysis_{trade_date}_{config_label}.json"

    result = {
        "ticker":                   ticker,
        "trade_date":               trade_date,
        "provider":                 provider,
        "hybrid_config":            hybrid,
        "deep_model":               config["deep_think_llm"],
        "quick_model":              config["quick_think_llm"],
        "decision":                 decision,
        "upstream_decision":        upstream_decision,
        "decision_corrected":       decision_corrected,
        "final_trade_decision_text": final_trade_text,
        "trader_investment_plan":   trader_plan_text,
        "elapsed_seconds":          round(elapsed_seconds, 1),
        "run_timestamp":            datetime.now().isoformat(),
        "quality_score": {
            "composite":          score.composite_score,
            "reasoning_depth":    score.reasoning_depth,
            "data_grounding":     score.data_grounding,
            "risk_awareness":     score.risk_awareness,
            "decision_consistent": score.decision_consistent,
            "has_stop_loss":      score.has_stop_loss,
            "has_price_target":   score.has_price_target,
            "has_position_sizing": score.has_position_sizing,
        },
        "cost_breakdown":    cost_info,
        "portfolio_context": portfolio_summary,
        "result_file":       str(result_file),
    }

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

    if cost_breakdown and cost_info:
        _print_cost_breakdown(cost_info, hybrid)

    print(f"\nResults saved to: {result_file}")
    print(f"{'='*60}\n")

    # ── Log to portfolio database ──────────────────────────────────────────
    try:
        from src.portfolio.tracker import PortfolioTracker
        tracker = PortfolioTracker()
        tracker.log_analysis(result, portfolio_context)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not log to portfolio DB: %s", e)

    return result


def _publish_signal(result: dict, trade_params=None) -> None:
    """Transform result into an AISignal and write it to Supabase.

    Called when --publish is active.  Errors are caught and logged — they
    never propagate to the caller.

    Args:
        result: Pipeline result dict from run_analysis().
        trade_params: TradeParams instance (may be None).
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        from src.integration.signal_adapter import transform_to_signal
        from src.integration.supabase_writer import SupabaseWriter

        signal = transform_to_signal(result, trade_params)
        writer = SupabaseWriter()
        writer.write_signal(signal)
    except Exception as e:
        _log.error("Signal publish failed: %s", e)


# ── Cost breakdown printer ────────────────────────────────────────────────────

def _print_cost_breakdown(cost_info: dict, hybrid_config: str = None) -> None:
    """Print a formatted cost breakdown table."""
    ALL_CLOUD_BASELINE_USD = 0.50

    by_model   = cost_info.get("by_model", {})
    total_usd  = cost_info.get("total_usd", 0.0)
    cache_hits = cost_info.get("cache_hits", 0)
    cache_misses = cost_info.get("cache_misses", 0)
    cache_total  = cache_hits + cache_misses

    savings_pct = 0.0
    if ALL_CLOUD_BASELINE_USD > 0 and total_usd < ALL_CLOUD_BASELINE_USD:
        savings_pct = (1 - total_usd / ALL_CLOUD_BASELINE_USD) * 100

    print(f"\n{'='*60}")
    print(f"COST BREAKDOWN")
    print(f"{'='*60}")
    for model_key, usage in sorted(by_model.items()):
        label = model_key
        if "haiku" in model_key:
            label = "Claude Haiku 4.5  (tool-calling agents)"
        elif "sonnet" in model_key:
            label = "Claude Sonnet 4.5 (reasoning judge)"
        elif "opus" in model_key:
            label = "Claude Opus 4.5   (deep reasoning)"
        elif "qwen" in model_key.lower() or "ollama" in model_key.lower():
            label = "Ollama/Qwen       (local — free)"
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        cost   = usage.get("cost_usd", 0.0)
        calls  = usage.get("calls", 0)
        print(f"  {label}")
        print(f"      {tokens:,} tokens ({calls} calls) → ${cost:.4f}")
    if cache_total > 0:
        print(f"  Cache hits:       {cache_hits}/{cache_total} analyst calls "
              f"({cost_info.get('cache_hit_rate_pct', 0):.0f}%)")
    print(f"  {'─'*50}")
    print(f"  Total:            ${total_usd:.4f}")
    if savings_pct > 0:
        print(f"  vs All-Cloud:     ~{savings_pct:.0f}% savings (baseline ${ALL_CLOUD_BASELINE_USD:.2f})")


# ── Execution flow ────────────────────────────────────────────────────────────

def _run_execution_flow(result: dict, config: dict, args,
                        tracker=None, analysis_id: int = None) -> dict:
    """Run the trade execution flow after analysis completes.

    Returns exec_result dict (may be empty for HOLD/dry-run).
    """
    from src.execution.trade_params import extract_trade_params_dual

    ticker           = result["ticker"]
    decision         = result["decision"]
    final_trade_text = result.get("final_trade_decision_text", "")
    trader_plan_text = result.get("trader_investment_plan", "")
    quality_score    = result.get("quality_score", {}).get("composite", 0.0)

    trade_params = extract_trade_params_dual(
        ticker=ticker,
        decision=decision,
        quality_score=quality_score,
        final_decision_text=final_trade_text,
        trader_plan_text=trader_plan_text,
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

    exec_result = {}

    if args.dry_run:
        print(f"\n  [DRY RUN — no order submitted]")

        if trade_params.is_actionable:
            from src.execution.executor import TradeExecutor
            from src.execution.position_manager import PositionManager
            audit_dir = str(Path(config["results_dir"]) / "audit")
            executor  = TradeExecutor(audit_dir=audit_dir)
            pm        = PositionManager(executor.client)
            order     = pm.calculate_order(trade_params)
            _print_order(order)
            exec_result = {"action": "DRY_RUN", "order": order.to_dict()}

            if tracker and analysis_id:
                try:
                    tracker.log_order(analysis_id, order, "DRY_RUN")
                except Exception:
                    pass

    elif args.execute and trade_params.is_actionable:
        from src.execution.executor import TradeExecutor
        from src.execution.position_manager import PositionManager

        audit_dir = str(Path(config["results_dir"]) / "audit")
        executor  = TradeExecutor(audit_dir=audit_dir)
        pm        = PositionManager(executor.client)

        order = pm.calculate_order(trade_params)
        _print_order(order)

        result_ex = executor.execute(order, trade_params)
        print(f"\n  Action: {result_ex['action']}")
        if result_ex.get("alpaca_order_id"):
            print(f"  Order ID: {result_ex['alpaca_order_id']}")

        action = result_ex.get("action", "EXECUTED")
        if tracker and analysis_id:
            try:
                tracker.log_order(
                    analysis_id, order, action,
                    alpaca_order_id=result_ex.get("alpaca_order_id"),
                    alpaca_status=result_ex.get("alpaca_status"),
                )
            except Exception:
                pass

        exec_result = result_ex

    elif args.execute and not trade_params.is_actionable:
        print(f"\n  [NOT ACTIONABLE — {trade_params.decision}, score={quality_score}]")

    return exec_result


def _print_order(order) -> None:
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


# ── Portfolio summary (--portfolio flag) ──────────────────────────────────────

def print_portfolio_summary_from_db(trade_date: str = None) -> None:
    """Print a full portfolio summary from Alpaca + DB history."""
    if trade_date is None:
        trade_date = str(date.today())

    print(f"\n{'═'*60}")
    print(f"PORTFOLIO SUMMARY ({trade_date})")
    print(f"{'═'*60}")

    # Live Alpaca data
    try:
        from src.execution.position_manager import PositionManager
        client    = _make_trading_client()
        pm        = PositionManager(client)
        account   = pm.get_account_state()
        positions = pm.get_positions()

        print(f"  Account equity:   ${account.equity:,.2f}")
        print(f"  Buying power:     ${account.buying_power:,.2f}")
        print(f"  Positions:        {len(positions)}")

        if positions:
            print(f"\n  HOLDINGS")
            print(f"  {'─'*50}")
            for sym, pos in sorted(positions.items()):
                avg_cost = pos.cost_basis / pos.qty if pos.qty else 0
                pct = pos.market_value / account.equity * 100 if account.equity else 0
                pnl_sign = "+" if pos.unrealized_pl >= 0 else ""
                print(f"  {sym:<6}  {pos.qty:>5} shares  "
                      f"${avg_cost:.2f} avg  "
                      f"{pnl_sign}${pos.unrealized_pl:,.0f}  "
                      f"{pnl_sign}{pos.unrealized_pl_pct:.1f}%  "
                      f"{pct:.1f}% of portfolio")
        else:
            print("\n  No open positions.")

    except Exception as e:
        print(f"  (Live account data unavailable: {e})")

    # Historical analyses from DB
    try:
        from src.portfolio.tracker import PortfolioTracker
        tracker = PortfolioTracker()
        recent  = tracker.get_batch_summary(trade_date)
        orders  = tracker.get_recent_orders(10)

        if recent["count"]:
            print(f"\n  RECENT ANALYSES ({trade_date})")
            print(f"  {'─'*50}")
            for row in recent["analyses"]:
                held_str = "yes" if row.get("held_at_analysis") else "no"
                print(f"  {row['trade_date']}  {row['ticker']:<6}  "
                      f"{row['decision']:<5}  "
                      f"{row['quality_score']:.1f}/10  "
                      f"(held: {held_str})")

        if orders:
            print(f"\n  ORDERS (recent {len(orders)})")
            print(f"  {'─'*50}")
            for o in orders:
                approved = "APPROVED" if o["approved"] else "REJECTED"
                print(f"  {o['timestamp'][:10]}  {o['ticker']:<6}  "
                      f"{o['side'].upper():<5}  {o['qty']:>5} shares  "
                      f"{approved} → {o['action']}")

    except Exception as e:
        print(f"\n  (Portfolio DB unavailable: {e})")

    print(f"{'═'*60}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trifecta Trader POC - Run trading analysis")
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol")
    parser.add_argument("--date", type=str, default=str(date.today()), help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "openai", "google", "ollama"],
                        help="LLM provider")
    parser.add_argument("--deep-model",  type=str, default=None)
    parser.add_argument("--quick-model", type=str, default=None)
    parser.add_argument("--no-debug",    action="store_true")
    parser.add_argument("--hybrid", type=str, default=None,
                        choices=["all_cloud", "hybrid_qwen", "hybrid_mistral",
                                 "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
                                 "hybrid_qwen32", "hybrid_aggressive_qwen32",
                                 "hybrid_qwen_enhanced",
                                 "hybrid_haiku_tools", "hybrid_haiku_aggressive",
                                 "hybrid_haiku_qwen35_27b", "hybrid_haiku_qwen35_35b",
                                 "hybrid_haiku_qwen35_9b"],
                        help="Use hybrid LLM routing config")
    parser.add_argument("--no-cache",          action="store_true")
    parser.add_argument("--no-cost-breakdown", action="store_true")
    parser.add_argument("--execute",   action="store_true",
                        help="Execute trade on Alpaca paper trading")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Calculate order but don't submit")
    parser.add_argument("--publish",   action="store_true",
                        help="Transform result to AISignal and write to Supabase platform")
    parser.add_argument("--portfolio", action="store_true",
                        help="Print portfolio summary from DB and exit (no --ticker required)")

    args = parser.parse_args()

    if args.portfolio:
        print_portfolio_summary_from_db(args.date)
        return

    # --ticker is required for all non-portfolio modes
    if not args.ticker:
        parser.error("--ticker is required unless --portfolio is specified")

    result = run_analysis(
        ticker=args.ticker,
        trade_date=args.date,
        provider=args.provider,
        deep_model=args.deep_model,
        quick_model=args.quick_model,
        debug=not args.no_debug,
        hybrid=args.hybrid,
        use_cache=not args.no_cache,
        cost_breakdown=not args.no_cost_breakdown,
    )

    # Extract trade params whenever publish / execute / dry-run is active
    trade_params = None
    if args.publish or args.execute or args.dry_run:
        try:
            from src.execution.trade_params import extract_trade_params_dual
            trade_params = extract_trade_params_dual(
                ticker=args.ticker,
                decision=result["decision"],
                quality_score=result.get("quality_score", {}).get("composite", 0.0),
                final_decision_text=result.get("final_trade_decision_text", ""),
                trader_plan_text=result.get("trader_investment_plan", ""),
                current_price=None,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Trade param extraction failed: %s", e)

    if args.execute or args.dry_run:
        config = get_config(args.provider, args.deep_model, args.quick_model)

        # Fetch analysis_id for tracker wiring (best-effort)
        tracker = None
        analysis_id = None
        try:
            from src.portfolio.tracker import PortfolioTracker
            tracker = PortfolioTracker()
            from src.portfolio.database import PortfolioDatabase
            db = PortfolioDatabase()
            config_label = args.hybrid if args.hybrid else args.provider
            analysis_id = db.get_analysis_id(args.ticker, args.date, config_label)
        except Exception:
            pass

        _run_execution_flow(result, config, args, tracker=tracker, analysis_id=analysis_id)

    # Publish signal to Supabase platform (additive — never blocks pipeline)
    if args.publish:
        _publish_signal(result, trade_params)


if __name__ == "__main__":
    main()
