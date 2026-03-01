"""
Trade executor for submitting orders to Alpaca paper trading.

SAFETY: This module ONLY connects to Alpaca paper trading.
The paper=True flag is hardcoded and cannot be overridden.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Submits orders to Alpaca paper trading with audit logging.

    Usage:
        executor = TradeExecutor(api_key="...", secret_key="...")
        result = executor.execute(order_calculation, trade_params)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        audit_dir: str = "audit",
    ):
        """Initialize executor with Alpaca paper trading client.

        Args:
            api_key: Alpaca API key (or set APCA_API_KEY_ID env var)
            secret_key: Alpaca secret key (or set APCA_API_SECRET_KEY env var)
            audit_dir: Directory for audit logs
        """
        import os

        self._api_key = api_key or os.environ.get("APCA_API_KEY_ID", "")
        self._secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")

        # SAFETY: paper=True is HARDCODED. This is NEVER configurable.
        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=True,  # ← HARDCODED. NEVER CHANGE THIS.
        )

        self._audit_dir = Path(audit_dir)
        self._audit_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        """Expose the trading client for PositionManager."""
        return self._client

    def execute(self, order_calc, trade_params) -> dict:
        """Execute an approved order on Alpaca paper trading.

        Args:
            order_calc: OrderCalculation from PositionManager
            trade_params: TradeParams from the parameter extractor

        Returns:
            dict with execution result and audit info
        """
        # Log the decision BEFORE execution
        audit_entry = self._create_audit_entry(order_calc, trade_params)

        if not order_calc.approved:
            audit_entry["action"] = "REJECTED"
            audit_entry["rejection_reasons"] = order_calc.rejection_reasons
            self._save_audit(audit_entry)
            logger.warning(
                "Order REJECTED for %s: %s",
                order_calc.ticker,
                order_calc.rejection_reasons,
            )
            return audit_entry

        # Submit order to Alpaca
        try:
            if order_calc.has_bracket_params and order_calc.take_profit:
                order = self._submit_bracket_order(order_calc)
            else:
                order = self._submit_market_order(order_calc)

            audit_entry["action"] = "EXECUTED"
            audit_entry["alpaca_order_id"] = str(order.id)
            audit_entry["alpaca_status"] = str(order.status)

            logger.info(
                "Order EXECUTED for %s: %s %d shares, order_id=%s",
                order_calc.ticker,
                order_calc.side,
                order_calc.qty,
                order.id,
            )

        except Exception as e:
            audit_entry["action"] = "FAILED"
            audit_entry["error"] = str(e)
            logger.error("Order FAILED for %s: %s", order_calc.ticker, e)

        self._save_audit(audit_entry)
        return audit_entry

    def _submit_bracket_order(self, order_calc):
        """Submit a bracket order (entry + stop-loss + take-profit)."""
        from alpaca.trading.requests import (
            MarketOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        side = OrderSide.BUY if order_calc.side == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=order_calc.ticker,
            qty=order_calc.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(order_calc.take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(order_calc.stop_loss, 2)),
        )

        logger.info(
            "Submitting BRACKET order: %s %d %s @ market, TP=$%.2f, SL=$%.2f",
            side.value, order_calc.qty, order_calc.ticker,
            order_calc.take_profit, order_calc.stop_loss,
        )

        return self._client.submit_order(request)

    def _submit_market_order(self, order_calc):
        """Submit a simple market order with a separate stop-loss."""
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        side = OrderSide.BUY if order_calc.side == "buy" else OrderSide.SELL

        # Use OTO (One-Triggers-Other) with stop-loss
        request = MarketOrderRequest(
            symbol=order_calc.ticker,
            qty=order_calc.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OTO,
            stop_loss=StopLossRequest(stop_price=round(order_calc.stop_loss, 2)),
        )

        logger.info(
            "Submitting OTO order: %s %d %s @ market, SL=$%.2f",
            side.value, order_calc.qty, order_calc.ticker, order_calc.stop_loss,
        )

        return self._client.submit_order(request)

    def _create_audit_entry(self, order_calc, trade_params) -> dict:
        """Create an audit log entry."""
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": order_calc.ticker,
            "decision": trade_params.decision,
            "quality_score": trade_params.quality_score,
            "confidence": trade_params.confidence,
            "side": order_calc.side,
            "qty": order_calc.qty,
            "entry_price": order_calc.entry_price,
            "stop_loss": order_calc.stop_loss,
            "take_profit": order_calc.take_profit,
            "position_value": order_calc.position_value,
            "position_pct": order_calc.position_pct_of_portfolio,
            "portfolio_risk_pct": order_calc.portfolio_risk_pct,
            "risk_reward_ratio": trade_params.risk_reward_ratio,
            "approved": order_calc.approved,
        }

    def _save_audit(self, audit_entry: dict):
        """Save audit entry to a JSON file."""
        ticker = audit_entry.get("ticker", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_{ts}.json"
        filepath = self._audit_dir / filename

        with open(filepath, "w") as f:
            json.dump(audit_entry, f, indent=2)

        logger.info("Audit log saved: %s", filepath)
