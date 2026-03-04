"""
Trifecta Trader POC - Watchlist Batch Runner

Processes multiple tickers sequentially using a shared portfolio context
(queried once from Alpaca at the start of the batch run).

Usage:
    python -m src.run_batch --watchlist config/watchlists/default.yaml --hybrid hybrid_haiku_tools
    python -m src.run_batch --tickers AAPL,MSFT,NVDA --hybrid hybrid_haiku_tools --dry-run
    python -m src.run_batch --watchlist config/watchlists/default.yaml --execute
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Add vendor TradingAgents to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))

from dotenv import load_dotenv
load_dotenv()

from src.run_analysis import (
    run_analysis,
    build_portfolio_context,
    print_portfolio_context,
    check_portfolio_warnings,
    _print_warnings,
    get_config,
    _run_execution_flow,
)
from src.execution.position_manager import MAX_POSITION_PCT


# ── Watchlist loading ─────────────────────────────────────────────────────────

def load_watchlist(path: str) -> tuple[str, list[str]]:
    """Load tickers from a YAML watchlist file.

    Returns (watchlist_name, list_of_tickers).
    Raises FileNotFoundError or ValueError on invalid input.
    """
    import yaml

    wl_path = Path(path)
    if not wl_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {path}")

    with open(wl_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Watchlist file is empty: {path}")

    tickers = data.get("tickers", [])
    if not tickers:
        raise ValueError(f"Watchlist has no tickers: {path}")

    name = data.get("name", wl_path.stem)
    return name, [t.upper().strip() for t in tickers if t]


def tickers_from_string(ticker_str: str) -> list[str]:
    """Parse a comma-separated ticker string into a list."""
    return [t.upper().strip() for t in ticker_str.split(",") if t.strip()]


# ── Portfolio helpers ─────────────────────────────────────────────────────────

def build_ticker_context(ticker: str, shared_context: dict,
                         positions: dict, account_equity: float) -> dict:
    """Build per-ticker portfolio context from the shared batch context."""
    pos = positions.get(ticker.upper())
    current_position = {"held": False}
    if pos:
        pct = round(pos.market_value / account_equity * 100, 1) if account_equity else 0
        current_position = {
            "held":           True,
            "shares":         pos.qty,
            "avg_cost":       round(pos.cost_basis / pos.qty, 2) if pos.qty else None,
            "unrealized_pnl": round(pos.unrealized_pl, 2),
            "current_value":  round(pos.market_value, 2),
            "portfolio_pct":  pct,
        }

    return {
        **shared_context,
        "current_position": current_position,
    }


def _query_batch_portfolio() -> tuple[dict, dict, float]:
    """Query Alpaca once for the full batch run.

    Returns (shared_context_dict, positions_dict, account_equity).
    Falls back gracefully on failure.
    """
    try:
        import os
        from alpaca.trading.client import TradingClient
        from src.execution.position_manager import PositionManager

        api_key    = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        client     = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        pm         = PositionManager(client)
        account    = pm.get_account_state()
        positions  = pm.get_positions()

        allocation = {
            sym: {
                "pct":    round(pos.market_value / account.equity * 100, 1)
                          if account.equity else 0,
                "shares": pos.qty,
            }
            for sym, pos in positions.items()
        }

        shared_ctx = {
            "account_equity":       round(account.equity, 2),
            "buying_power":         round(account.buying_power, 2),
            "cash":                 round(account.cash, 2),
            "total_positions":      len(positions),
            "portfolio_allocation": allocation,
            "_source":              "alpaca",
        }
        return shared_ctx, positions, account.equity

    except Exception as e:
        print(f"  ⚠ Could not fetch portfolio from Alpaca: {e}")
        return (
            {"account_equity": None, "buying_power": None, "cash": None,
             "total_positions": None, "portfolio_allocation": {},
             "_source": "unavailable", "_error": str(e)},
            {},
            0.0,
        )


def _print_batch_portfolio_header(ctx: dict) -> None:
    """Print the portfolio overview at the start of a batch run."""
    if ctx.get("_source") == "unavailable":
        print(f"\n{'═'*60}")
        print("PORTFOLIO CONTEXT  (unavailable)")
        print(f"  {ctx.get('_error', '')}")
        print(f"{'═'*60}")
        return

    alloc = ctx.get("portfolio_allocation", {})
    alloc_str = ", ".join(
        f"{sym} {v['pct']:.0f}%" for sym, v in list(alloc.items())[:6]
    ) or "none"

    print(f"\n{'═'*60}")
    print("PORTFOLIO CONTEXT")
    print(f"{'═'*60}")
    print(f"  Account equity:   ${ctx['account_equity']:,.2f}")
    print(f"  Buying power:     ${ctx['buying_power']:,.2f}")
    print(f"  Cash:             ${ctx['cash']:,.2f}")
    print(f"  Positions:        {ctx['total_positions']} ({alloc_str})")
    print(f"{'═'*60}")


# ── Batch summary helpers ─────────────────────────────────────────────────────

def _print_batch_summary(results: list, watchlist_name: str,
                         hybrid: str, trade_date: str,
                         elapsed_total: float, execute: bool, dry_run: bool) -> None:
    """Print the consolidated batch summary table."""
    total_cost = sum(
        r.get("cost_breakdown", {}).get("total_usd", 0) for r in results
    )

    print(f"\n{'═'*60}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'═'*60}")
    print(f"  Watchlist:   {watchlist_name} ({len(results)} tickers analysed)")
    print(f"  Config:      {hybrid or 'default'}")
    print(f"  Date:        {trade_date}")
    print(f"  Total time:  {elapsed_total/60:.1f}m")
    print(f"  Total cost:  ${total_cost:.4f}")

    print(f"\n  RESULTS")
    print(f"  {'─'*50}")
    print(f"  {'Ticker':<8} {'Decision':<7} {'Quality':>7}  {'Cost':>8}  {'Target':>9}  Holdings")
    print(f"  {'─'*50}")
    for r in results:
        qs     = r.get("quality_score", {}).get("composite", 0)
        cost   = r.get("cost_breakdown", {}).get("total_usd", 0)
        pctx   = r.get("portfolio_context", {})
        held   = pctx.get("held", False)
        shares = pctx.get("shares", 0)

        # Price target from quality scorer (stored in result if available)
        target = r.get("trade_params", {}).get("price_target")
        target_str = f"${target:.0f}" if target else "—"
        held_str = f"HELD: {shares} shares" if held else "NOT HELD"

        cost_str = f"${cost:.3f}" if cost else "—"
        print(f"  {r['ticker']:<8} {r['decision']:<7} {qs:>5.1f}/10  "
              f"{cost_str:>8}  {target_str:>9}  ({held_str})")

    # Actionable signals
    buys  = [r["ticker"] for r in results if r["decision"] == "BUY"]
    sells = [r["ticker"] for r in results if r["decision"] == "SELL"]
    holds = [r["ticker"] for r in results if r["decision"] == "HOLD"]

    print(f"\n  ACTIONABLE SIGNALS")
    print(f"  {'─'*50}")
    if buys:
        print(f"  BUY:   {', '.join(buys)}")
    if sells:
        print(f"  SELL:  {', '.join(sells)}")
    if holds:
        print(f"  HOLD:  {', '.join(holds)}")
    print(f"{'═'*60}")


def _save_batch_results(results: list, watchlist_name: str,
                        hybrid: str, trade_date: str,
                        elapsed_total: float) -> Path:
    """Save consolidated batch results JSON to results/batch/."""
    batch_dir = Path("results") / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = watchlist_name.replace(" ", "_").lower()
    out  = batch_dir / f"batch_{ts}_{name}.json"

    total_cost = sum(
        r.get("cost_breakdown", {}).get("total_usd", 0) for r in results
    )

    payload = {
        "watchlist":     watchlist_name,
        "hybrid_config": hybrid,
        "trade_date":    trade_date,
        "run_timestamp": datetime.now().isoformat(),
        "tickers_count": len(results),
        "elapsed_seconds": round(elapsed_total, 1),
        "total_cost_usd": round(total_cost, 4),
        "results":       results,
    }

    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nBatch results saved to: {out}")
    return out


# ── Core batch runner ─────────────────────────────────────────────────────────

def run_batch(
    tickers:      list[str],
    watchlist_name: str,
    hybrid:       Optional[str],
    trade_date:   str,
    execute:      bool = False,
    dry_run:      bool = False,
    use_cache:    bool = True,
    cost_breakdown: bool = True,
    skip_held:    bool = False,
    priority_sort: bool = False,  # --priority-sort: Reserved for Scanner integration.
                                   # When the Market Scanner sends candidates, priority will be
                                   # defined by the Scanner's opportunity_score field. For now,
                                   # tickers are processed in watchlist order.
    provider:     str = "anthropic",
) -> list[dict]:
    """Process a list of tickers sequentially with shared portfolio context.

    Returns list of result dicts (one per analysed ticker).
    """
    # 1. Query portfolio ONCE for the entire batch
    shared_ctx, positions, account_equity = _query_batch_portfolio()
    _print_batch_portfolio_header(shared_ctx)

    # 2. Take a snapshot for daily P&L tracking
    try:
        if shared_ctx.get("_source") == "alpaca":
            import os
            from alpaca.trading.client import TradingClient
            from src.execution.position_manager import PositionManager
            from src.portfolio.tracker import PortfolioTracker

            api_key    = os.environ.get("APCA_API_KEY_ID", "")
            secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
            client     = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
            pm         = PositionManager(client)
            tracker    = PortfolioTracker()
            tracker.take_snapshot(pm)
    except Exception as e:
        print(f"  ⚠ Could not take portfolio snapshot: {e}")

    results       = []
    batch_start   = time.time()
    config        = get_config(provider)

    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] Analysing {ticker}...")

        # Build per-ticker context from the shared batch state
        ticker_ctx = build_ticker_context(ticker, shared_ctx, positions, account_equity)
        pos        = ticker_ctx.get("current_position", {})

        # --skip-held: skip tickers already at max allocation
        if skip_held and pos.get("held") and pos.get("portfolio_pct", 0) >= MAX_POSITION_PCT:
            print(f"  SKIPPED: {ticker} already at max allocation ({pos['portfolio_pct']:.1f}%)")
            continue

        # Portfolio warnings (batch mode = no interactive prompt)
        warnings = check_portfolio_warnings(ticker, ticker_ctx, batch_mode=True)
        _print_warnings(warnings, ticker, batch_mode=True)

        try:
            result = run_analysis(
                ticker=ticker,
                trade_date=trade_date,
                provider=provider,
                hybrid=hybrid,
                use_cache=use_cache,
                cost_breakdown=cost_breakdown,
                portfolio_context=ticker_ctx,
                batch_mode=True,
                debug=False,  # batch always suppresses agent debug output
            )
        except Exception as e:
            print(f"  ERROR analysing {ticker}: {e}")
            results.append({
                "ticker":   ticker,
                "decision": "ERROR",
                "error":    str(e),
                "portfolio_context": {"held": pos.get("held", False)},
            })
            continue

        # Execute / dry-run per-ticker if requested
        if execute or dry_run:
            try:
                # Fetch analysis_id for tracker wiring
                from src.portfolio.tracker import PortfolioTracker
                from src.portfolio.database import PortfolioDatabase
                trk = PortfolioTracker()
                db  = PortfolioDatabase()
                config_label = hybrid or provider
                aid = db.get_analysis_id(ticker, trade_date, config_label)

                class _FakeArgs:
                    execute  = False
                    dry_run  = False

                fa = _FakeArgs()
                fa.execute  = execute
                fa.dry_run  = dry_run
                _run_execution_flow(result, config, fa, tracker=trk, analysis_id=aid)
            except Exception as e:
                print(f"  ⚠ Execution flow failed for {ticker}: {e}")

        results.append(result)

        # Running cost total
        total_so_far = sum(
            r.get("cost_breakdown", {}).get("total_usd", 0) for r in results
        )
        print(f"  Cost so far: ${total_so_far:.4f}")

    elapsed_total = time.time() - batch_start

    # Print and save batch summary
    _print_batch_summary(results, watchlist_name, hybrid, trade_date,
                         elapsed_total, execute, dry_run)
    _save_batch_results(results, watchlist_name, hybrid, trade_date, elapsed_total)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trifecta Trader POC - Watchlist Batch Runner"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--watchlist", type=str,
                     help="Path to YAML watchlist file")
    src.add_argument("--tickers",   type=str,
                     help="Comma-separated ticker list (e.g. AAPL,MSFT,NVDA)")

    parser.add_argument("--hybrid", type=str, default="hybrid_haiku_tools",
                        choices=["all_cloud", "hybrid_qwen", "hybrid_mistral",
                                 "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
                                 "hybrid_qwen32", "hybrid_aggressive_qwen32",
                                 "hybrid_qwen_enhanced",
                                 "hybrid_haiku_tools", "hybrid_haiku_aggressive",
                                 "hybrid_haiku_qwen35_27b", "hybrid_haiku_qwen35_35b",
                                 "hybrid_haiku_qwen35_9b"],
                        help="Hybrid LLM config (default: hybrid_haiku_tools)")
    parser.add_argument("--date",     type=str, default=str(date.today()))
    parser.add_argument("--execute",  action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug output (passed through to run_analysis)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-cost-breakdown", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Reserved (always 1 — sequential processing only)")
    parser.add_argument("--skip-held",     action="store_true",
                        help="Skip tickers already at max portfolio allocation")
    parser.add_argument("--priority-sort", action="store_true",
                        help="Reserved for Scanner integration (no-op)")

    args = parser.parse_args()

    if args.watchlist:
        watchlist_name, tickers = load_watchlist(args.watchlist)
    else:
        tickers        = tickers_from_string(args.tickers)
        watchlist_name = "CLI tickers"

    if not tickers:
        print("No tickers to process.")
        return

    print(f"\nBatch run: {watchlist_name}  ({len(tickers)} tickers)")
    print(f"Config:    {args.hybrid}   Date: {args.date}")

    run_batch(
        tickers=tickers,
        watchlist_name=watchlist_name,
        hybrid=args.hybrid,
        trade_date=args.date,
        execute=args.execute,
        dry_run=args.dry_run,
        use_cache=not args.no_cache,
        cost_breakdown=not args.no_cost_breakdown,
        skip_held=args.skip_held,
        priority_sort=args.priority_sort,
    )


if __name__ == "__main__":
    main()
