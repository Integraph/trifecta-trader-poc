"""
Position manager for tracking holdings and calculating order quantities.

Interfaces with the Alpaca API to get real-time account and position data,
and applies risk controls before order submission.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Safety constants — these are hardcoded, never configurable
MAX_POSITION_PCT = 15.0  # Never put more than 15% in a single stock
MIN_QUALITY_SCORE = 8.0  # Never trade on low-quality analysis
MAX_PORTFOLIO_RISK_PCT = 2.0  # Max portfolio risk per trade (position_size * stop_distance)
MIN_RISK_REWARD = 1.5  # Minimum risk/reward ratio


@dataclass
class AccountState:
    """Current account state from Alpaca."""
    buying_power: float
    portfolio_value: float
    cash: float
    equity: float


@dataclass
class Position:
    """A current stock position."""
    ticker: str
    qty: int
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class OrderCalculation:
    """Result of calculating an order based on trade params."""
    ticker: str
    side: str  # "buy" or "sell"
    qty: int
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]

    # Risk metrics
    position_value: float
    position_pct_of_portfolio: float
    risk_per_share: float
    total_risk: float
    portfolio_risk_pct: float

    # Validation
    approved: bool
    rejection_reasons: List[str]

    @property
    def has_bracket_params(self) -> bool:
        return self.take_profit is not None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_value": round(self.position_value, 2),
            "position_pct": round(self.position_pct_of_portfolio, 2),
            "risk_per_share": round(self.risk_per_share, 2),
            "total_risk": round(self.total_risk, 2),
            "portfolio_risk_pct": round(self.portfolio_risk_pct, 4),
            "approved": self.approved,
            "rejection_reasons": self.rejection_reasons,
        }


class PositionManager:
    """Manages positions and calculates orders with risk controls.

    Usage:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        pm = PositionManager(client)

        account = pm.get_account_state()
        positions = pm.get_positions()
        order = pm.calculate_order(trade_params)
    """

    def __init__(self, trading_client):
        """Initialize with an Alpaca TradingClient.

        Args:
            trading_client: An alpaca.trading.client.TradingClient instance
                           (must be paper=True)
        """
        self._client = trading_client

    def get_account_state(self) -> AccountState:
        """Get current account state from Alpaca."""
        account = self._client.get_account()
        return AccountState(
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            cash=float(account.cash),
            equity=float(account.equity),
        )

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions as a dict keyed by ticker."""
        positions = {}
        for p in self._client.get_all_positions():
            positions[p.symbol] = Position(
                ticker=p.symbol,
                qty=int(p.qty),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_pl_pct=float(p.unrealized_plpc) * 100,
            )
        return positions

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get a specific position, or None if not held."""
        try:
            p = self._client.get_open_position(ticker)
            return Position(
                ticker=p.symbol,
                qty=int(p.qty),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_pl_pct=float(p.unrealized_plpc) * 100,
            )
        except Exception:
            return None

    def calculate_order(self, trade_params) -> OrderCalculation:
        """Calculate an order with risk controls.

        Args:
            trade_params: TradeParams from the parameter extractor

        Returns:
            OrderCalculation with approved/rejected status and reasons
        """
        rejection_reasons = []
        account = self.get_account_state()
        current_position = self.get_position(trade_params.ticker)

        # --- Determine side ---
        if trade_params.decision == "BUY":
            side = "buy"
        elif trade_params.decision == "SELL":
            side = "sell"
        else:
            return OrderCalculation(
                ticker=trade_params.ticker, side="none", qty=0,
                entry_price=0, stop_loss=0, take_profit=None,
                position_value=0, position_pct_of_portfolio=0,
                risk_per_share=0, total_risk=0, portfolio_risk_pct=0,
                approved=False, rejection_reasons=["Decision is HOLD — no action"],
            )

        entry_price = trade_params.entry_price or 0
        stop_loss = trade_params.stop_loss or 0

        # --- Quality gate ---
        if trade_params.quality_score < MIN_QUALITY_SCORE:
            rejection_reasons.append(
                f"Quality score {trade_params.quality_score:.1f} < minimum {MIN_QUALITY_SCORE}"
            )

        # --- Stop-loss required ---
        if not trade_params.stop_loss:
            rejection_reasons.append("No stop-loss extracted from analysis")

        # --- Calculate position size ---
        target_pct = trade_params.position_pct or 5.0  # Default 5% if not extracted
        target_pct = min(target_pct, MAX_POSITION_PCT)  # Cap at maximum

        target_value = account.portfolio_value * (target_pct / 100.0)

        if entry_price > 0:
            qty = int(target_value / entry_price)
        else:
            qty = 0
            rejection_reasons.append("No entry price available")

        # --- Adjust for existing position ---
        if side == "buy" and current_position:
            existing_value = current_position.market_value
            remaining_value = target_value - existing_value
            if remaining_value <= 0:
                rejection_reasons.append(
                    f"Already hold {current_position.qty} shares "
                    f"(${existing_value:.0f}, {existing_value/account.portfolio_value*100:.1f}% of portfolio)"
                )
                qty = 0
            else:
                qty = int(remaining_value / entry_price)

        if side == "sell" and current_position:
            # Sell what we have, up to the calculated amount
            qty = min(qty, current_position.qty)
        elif side == "sell" and not current_position:
            rejection_reasons.append(f"No position in {trade_params.ticker} to sell")
            qty = 0

        # --- Check buying power ---
        if side == "buy" and entry_price * qty > account.buying_power:
            affordable_qty = int(account.buying_power / entry_price) if entry_price > 0 else 0
            if affordable_qty > 0:
                qty = affordable_qty
                logger.warning("Reduced qty to %d due to buying power ($%.0f)", qty, account.buying_power)
            else:
                rejection_reasons.append(
                    f"Insufficient buying power: ${account.buying_power:.0f}"
                )

        # --- Calculate risk metrics ---
        position_value = qty * entry_price if entry_price else 0
        position_pct = (position_value / account.portfolio_value * 100) if account.portfolio_value else 0

        risk_per_share = abs(entry_price - stop_loss) if entry_price and stop_loss else 0
        total_risk = risk_per_share * qty
        portfolio_risk_pct = (total_risk / account.portfolio_value * 100) if account.portfolio_value else 0

        # --- Risk/reward check ---
        if trade_params.risk_reward_ratio is not None and trade_params.risk_reward_ratio < MIN_RISK_REWARD:
            rejection_reasons.append(
                f"Risk/reward {trade_params.risk_reward_ratio:.2f} < minimum {MIN_RISK_REWARD}"
            )

        # --- Portfolio risk check ---
        if portfolio_risk_pct > MAX_PORTFOLIO_RISK_PCT:
            rejection_reasons.append(
                f"Portfolio risk {portfolio_risk_pct:.2f}% > maximum {MAX_PORTFOLIO_RISK_PCT}%"
            )

        # --- Position size check ---
        if position_pct > MAX_POSITION_PCT:
            rejection_reasons.append(
                f"Position {position_pct:.1f}% > maximum {MAX_POSITION_PCT}%"
            )

        # --- Zero quantity check ---
        if qty <= 0:
            if not rejection_reasons:
                rejection_reasons.append("Calculated quantity is 0")

        approved = len(rejection_reasons) == 0 and qty > 0

        return OrderCalculation(
            ticker=trade_params.ticker,
            side=side,
            qty=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=trade_params.price_target,
            position_value=position_value,
            position_pct_of_portfolio=position_pct,
            risk_per_share=risk_per_share,
            total_risk=total_risk,
            portfolio_risk_pct=portfolio_risk_pct,
            approved=approved,
            rejection_reasons=rejection_reasons,
        )
