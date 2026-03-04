"""
PortfolioTracker — business logic layer on top of PortfolioDatabase.

Logs analysis results, order attempts, and portfolio snapshots.
Provides summary queries used by the --portfolio CLI flag.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from src.portfolio.database import PortfolioDatabase

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """High-level interface for recording and querying portfolio history."""

    def __init__(self, db_path: str = "data/portfolio.db"):
        self._db = PortfolioDatabase(db_path)

    # ── write operations ──────────────────────────────────────────────────

    def log_analysis(self, result: dict, portfolio_context: dict) -> int:
        """Record an analysis result.  Returns the analysis row id."""
        qs = result.get("quality_score", {})
        cost_info = result.get("cost_breakdown", {})
        cost_usd = cost_info.get("total_usd", None)

        pos_ctx = portfolio_context.get("current_position", {})
        held = bool(pos_ctx.get("held", False))

        row = {
            "ticker":            result.get("ticker", ""),
            "trade_date":        result.get("trade_date", ""),
            "run_timestamp":     result.get("run_timestamp", datetime.now().isoformat()),
            "config":            result.get("hybrid_config") or result.get("provider", "unknown"),
            "decision":          result.get("decision", ""),
            "quality_score":     qs.get("composite", 0.0),
            "cost_usd":          cost_usd,
            "elapsed_seconds":   result.get("elapsed_seconds"),
            "stop_loss":         result.get("trade_params", {}).get("stop_loss"),
            "price_target":      result.get("trade_params", {}).get("price_target"),
            "entry_price":       result.get("trade_params", {}).get("entry_price"),
            "position_size_pct": result.get("trade_params", {}).get("position_pct"),
            "risk_reward":       result.get("trade_params", {}).get("risk_reward_ratio"),
            "actionable":        int(result.get("trade_params", {}).get("actionable", False)),
            "portfolio_equity":  portfolio_context.get("account_equity"),
            "held_at_analysis":  int(held),
            "held_shares":       pos_ctx.get("shares", 0) if held else None,
            "held_avg_cost":     pos_ctx.get("avg_cost") if held else None,
            "result_file":       result.get("result_file"),
        }

        analysis_id = self._db.upsert_analysis(row)
        logger.debug("Logged analysis id=%d ticker=%s decision=%s",
                     analysis_id, row["ticker"], row["decision"])
        return analysis_id

    def log_order(self, analysis_id: int, order_calc, action: str,
                  alpaca_order_id: str = None, alpaca_status: str = None) -> int:
        """Record an order attempt.  Returns order row id.

        Args:
            analysis_id: Foreign key to analyses table.
            order_calc: OrderCalculation dataclass from PositionManager.
            action: One of EXECUTED, REJECTED, DRY_RUN.
            alpaca_order_id: Alpaca order ID if submitted.
            alpaca_status: Alpaca order status string.
        """
        row = {
            "analysis_id":      analysis_id,
            "ticker":           order_calc.ticker,
            "timestamp":        datetime.now().isoformat(),
            "side":             order_calc.side,
            "qty":              order_calc.qty,
            "entry_price":      order_calc.entry_price,
            "stop_loss":        order_calc.stop_loss,
            "take_profit":      order_calc.take_profit,
            "approved":         int(order_calc.approved),
            "rejection_reasons": json.dumps(order_calc.rejection_reasons),
            "action":           action,
            "alpaca_order_id":  alpaca_order_id,
            "alpaca_status":    alpaca_status,
        }
        order_id = self._db.insert_order(row)
        logger.debug("Logged order id=%d ticker=%s action=%s",
                     order_id, order_calc.ticker, action)
        return order_id

    def take_snapshot(self, position_manager) -> None:
        """Capture today's portfolio state from Alpaca.

        Args:
            position_manager: PositionManager instance (already connected).
        """
        try:
            account = position_manager.get_account_state()
            positions = position_manager.get_positions()
            today = datetime.now().strftime("%Y-%m-%d")

            positions_data = {
                sym: {
                    "qty": pos.qty,
                    "market_value": pos.market_value,
                    "cost_basis": pos.cost_basis,
                    "unrealized_pl": pos.unrealized_pl,
                    "unrealized_pl_pct": pos.unrealized_pl_pct,
                }
                for sym, pos in positions.items()
            }

            self._db.upsert_snapshot({
                "snapshot_date":  today,
                "account_equity": account.equity,
                "buying_power":   account.buying_power,
                "cash":           account.cash,
                "positions_json": json.dumps(positions_data),
                "total_positions": len(positions),
            })
            logger.info("Portfolio snapshot saved for %s", today)
        except Exception as e:
            logger.warning("Failed to take portfolio snapshot: %s", e)

    # ── read operations ───────────────────────────────────────────────────

    def get_analysis_history(self, ticker: str, limit: int = 10) -> List[dict]:
        return self._db.get_recent_analyses(ticker, limit)

    def get_decision_history(self, ticker: str) -> List[dict]:
        return self._db.get_decision_history(ticker)

    def get_daily_pnl(self, days: int = 30) -> List[dict]:
        """Calculate equity change between consecutive daily snapshots."""
        snapshots = self._db.get_snapshots(days)
        if len(snapshots) < 2:
            return []

        pnl_rows = []
        for i in range(len(snapshots) - 1):
            today_snap = snapshots[i]
            prev_snap  = snapshots[i + 1]
            delta = today_snap["account_equity"] - prev_snap["account_equity"]
            pnl_pct = (delta / prev_snap["account_equity"] * 100
                       if prev_snap["account_equity"] else 0.0)
            pnl_rows.append({
                "date":           today_snap["snapshot_date"],
                "equity":         today_snap["account_equity"],
                "daily_pnl":      round(delta, 2),
                "daily_pnl_pct":  round(pnl_pct, 3),
            })
        return pnl_rows

    def get_batch_summary(self, trade_date: str) -> dict:
        """Summarise all analyses run on a given date."""
        rows = self._db.get_date_summary(trade_date)
        decisions = {}
        for r in rows:
            decisions.setdefault(r["decision"], []).append(r["ticker"])

        total_cost = sum(r.get("cost_usd") or 0 for r in rows)
        return {
            "date":       trade_date,
            "count":      len(rows),
            "decisions":  decisions,
            "total_cost": round(total_cost, 4),
            "analyses":   rows,
        }

    def get_recent_orders(self, limit: int = 20) -> List[dict]:
        return self._db.get_recent_orders(limit)

    def get_recent_snapshots(self, days: int = 7) -> List[dict]:
        return self._db.get_snapshots(days)
