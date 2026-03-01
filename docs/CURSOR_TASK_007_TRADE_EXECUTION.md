# Cursor Task 007: Position Management & Trade Execution Layer

## Objective
Build the execution layer that translates pipeline decisions (BUY/HOLD/SELL) into Alpaca paper trading orders with proper position management, risk controls, and an audit trail.

## Context
- Phase A (Tasks 001-006) validated the multi-agent analysis pipeline with hybrid LLM routing
- The pipeline produces a `decision` (BUY/HOLD/SELL), a `quality_score`, and rich `final_trade_decision_text` containing stop-loss levels, price targets, and position sizing recommendations
- We now need to bridge the gap between "the pipeline says BUY" and "an order gets placed on Alpaca"
- This is Phase B — paper trading integration
- **All trading is paper trading only** — no real money, no live API

### What the Pipeline Produces (example result JSON)

```json
{
  "ticker": "AAPL",
  "trade_date": "2026-02-27",
  "decision": "HOLD",
  "quality_score": {
    "composite": 10.0,
    "reasoning_depth": 10,
    "data_grounding": 10,
    "risk_awareness": 10,
    "decision_consistent": true,
    "has_stop_loss": true,
    "has_price_target": true,
    "has_position_sizing": true
  },
  "final_trade_decision_text": "... rich text with stop-loss at $258, target $295, position 5% of portfolio ..."
}
```

### What We Need to Build

1. **Trade parameter extractor** — parse stop-loss, price target, and position size from the decision text
2. **Position manager** — track what we own, calculate order quantities based on account size
3. **Trade executor** — submit orders to Alpaca with bracket orders (entry + stop-loss + take-profit)
4. **Safety controls** — minimum quality score threshold, maximum position size, confirmation logging
5. **Audit trail** — log every decision and action for review

## Important Rules
- **DO NOT modify files in `vendor/TradingAgents/` directly**
- **ALL trading is PAPER TRADING** — never connect to the live Alpaca API
- The `paper=True` flag must be hardcoded; it should never be configurable
- All new code goes in `src/execution/`, `src/audit/`, and `tests/`
- Every trade action must be logged before execution

---

## Step 1: Install Alpaca SDK

### Commands

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

# Install the official Alpaca Python SDK
pip install alpaca-py

# Add to pyproject.toml dependencies
```

Update `pyproject.toml` to include `alpaca-py` in the project dependencies.

---

## Step 2: Trade Parameter Extractor

### Goal
Parse the pipeline's `final_trade_decision_text` to extract actionable trade parameters: stop-loss price, price target, position size percentage.

### Implementation

Create `src/execution/trade_params.py`:

```python
"""
Extract actionable trade parameters from pipeline decision text.

The pipeline produces rich text analysis containing stop-loss levels,
price targets, and position sizing. This module parses those into
structured parameters for order submission.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeParams:
    """Structured trade parameters extracted from pipeline output."""
    ticker: str
    decision: str  # BUY, HOLD, SELL
    quality_score: float

    # Price levels (extracted from text)
    entry_price: Optional[float] = None  # Current/suggested entry
    stop_loss: Optional[float] = None
    price_target: Optional[float] = None

    # Position sizing
    position_pct: Optional[float] = None  # % of portfolio (e.g., 5.0 = 5%)
    shares: Optional[int] = None  # Calculated from position_pct and account value

    # Risk metrics (calculated)
    risk_pct: Optional[float] = None  # % risk from entry to stop
    reward_pct: Optional[float] = None  # % reward from entry to target
    risk_reward_ratio: Optional[float] = None

    # Metadata
    confidence: str = "medium"  # low, medium, high
    raw_text_excerpt: str = ""  # Last 500 chars for audit

    @property
    def is_actionable(self) -> bool:
        """Whether we have enough data to place an order."""
        if self.decision == "HOLD":
            return False
        return (
            self.decision in ("BUY", "SELL")
            and self.stop_loss is not None
            and self.quality_score >= 8.0
        )

    @property
    def has_bracket_params(self) -> bool:
        """Whether we have both stop-loss and target for a bracket order."""
        return self.stop_loss is not None and self.price_target is not None


def extract_trade_params(
    ticker: str,
    decision: str,
    quality_score: float,
    decision_text: str,
    current_price: Optional[float] = None,
) -> TradeParams:
    """Extract trade parameters from pipeline decision text.

    Args:
        ticker: Stock ticker
        decision: BUY/HOLD/SELL from signal processor
        quality_score: Composite quality score (0-10)
        decision_text: Full text of the final trade decision
        current_price: Current market price (optional, used as entry price fallback)

    Returns:
        TradeParams with extracted values
    """
    params = TradeParams(
        ticker=ticker,
        decision=decision,
        quality_score=quality_score,
        entry_price=current_price,
        raw_text_excerpt=decision_text[-500:] if decision_text else "",
    )

    if not decision_text:
        return params

    # --- Extract stop-loss ---
    # Patterns: "stop-loss: $258", "stop loss at $258", "stop at $258.00",
    # "trailing stop-loss at 12%", "hard stop at $254"
    stop_patterns = [
        r'stop[- ]?loss[:\s]+\$?([\d,.]+)',
        r'(?:hard|trailing)\s+stop[:\s]+(?:at\s+)?\$?([\d,.]+)',
        r'stop[:\s]+(?:at\s+)?\$?([\d,.]+)',
    ]
    for pattern in stop_patterns:
        match = re.search(pattern, decision_text, re.IGNORECASE)
        if match:
            try:
                params.stop_loss = float(match.group(1).replace(",", ""))
                break
            except ValueError:
                continue

    # --- Extract price target ---
    # Patterns: "price target: $295", "target $295", "target: $295.00",
    # "upside target of $300"
    target_patterns = [
        r'(?:price\s+)?target[:\s]+\$?([\d,.]+)',
        r'target\s+(?:of\s+|at\s+)?\$?([\d,.]+)',
        r'upside\s+(?:to\s+|target\s+)?\$?([\d,.]+)',
        r'take[- ]?profit[:\s]+\$?([\d,.]+)',
    ]
    for pattern in target_patterns:
        match = re.search(pattern, decision_text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                # Sanity check: target should be reasonable (not a percentage)
                if val > 10:  # Likely a dollar value, not a percentage
                    params.price_target = val
                    break
            except ValueError:
                continue

    # --- Extract position sizing ---
    # Patterns: "5% of portfolio", "position size: 4%", "allocate 3-4% of portfolio",
    # "position sizing: 5% portfolio weight"
    size_patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:portfolio|allocation|position)',
        r'position\s+siz(?:e|ing)[:\s]+(\d+(?:\.\d+)?)\s*%',
        r'allocat(?:e|ion)[:\s]+(\d+(?:\.\d+)?)\s*%',
        r'portfolio\s+weight[:\s]+(\d+(?:\.\d+)?)\s*%',
    ]
    for pattern in size_patterns:
        match = re.search(pattern, decision_text, re.IGNORECASE)
        if match:
            try:
                params.position_pct = float(match.group(1))
                break
            except ValueError:
                continue

    # --- Extract entry price (if not provided) ---
    if params.entry_price is None:
        entry_patterns = [
            r'entry[:\s]+(?:at\s+)?\$?([\d,.]+)',
            r'current\s+price[:\s]+\$?([\d,.]+)',
            r'trading\s+at\s+\$?([\d,.]+)',
        ]
        for pattern in entry_patterns:
            match = re.search(pattern, decision_text, re.IGNORECASE)
            if match:
                try:
                    params.entry_price = float(match.group(1).replace(",", ""))
                    break
                except ValueError:
                    continue

    # --- Calculate risk metrics ---
    if params.entry_price and params.stop_loss:
        params.risk_pct = abs(params.entry_price - params.stop_loss) / params.entry_price * 100

    if params.entry_price and params.price_target:
        params.reward_pct = abs(params.price_target - params.entry_price) / params.entry_price * 100

    if params.risk_pct and params.reward_pct and params.risk_pct > 0:
        params.risk_reward_ratio = params.reward_pct / params.risk_pct

    # --- Extract confidence ---
    confidence_match = re.search(
        r'(?:confidence|conviction)[:\s]+(\w+)',
        decision_text,
        re.IGNORECASE,
    )
    if confidence_match:
        conf = confidence_match.group(1).lower()
        if conf in ("high", "strong"):
            params.confidence = "high"
        elif conf in ("low", "weak", "uncertain"):
            params.confidence = "low"
        else:
            params.confidence = "medium"

    logger.info(
        "Extracted params for %s: decision=%s, stop=%.2f, target=%.2f, size=%.1f%%",
        ticker,
        decision,
        params.stop_loss or 0,
        params.price_target or 0,
        params.position_pct or 0,
    )

    return params
```

### Tests

Create `tests/test_trade_params.py`:

```python
"""Tests for trade parameter extraction from pipeline output."""

import pytest
from src.execution.trade_params import extract_trade_params, TradeParams


class TestExtractTradeParams:
    """Test parameter extraction from various output formats."""

    def test_standard_format(self):
        """Standard format with clear stop-loss, target, sizing."""
        text = """
        Entry: $270.15. Stop-loss: $258.00 (-4.5% risk). Target: $295.00 (+9.2%).
        Risk/Reward: 2.04:1. Position size: 5% of portfolio.
        FINAL TRANSACTION PROPOSAL: BUY
        """
        params = extract_trade_params("AAPL", "BUY", 9.5, text, current_price=270.15)

        assert params.stop_loss == 258.00
        assert params.price_target == 295.00
        assert params.position_pct == 5.0
        assert params.is_actionable
        assert params.has_bracket_params

    def test_hybrid_output_format(self):
        """Format seen in actual hybrid_qwen_enhanced output."""
        text = """
        ## Risk Management
        - Stop-loss: $258 (-4.4%)
        - Price target: $295 (+9.3%)
        - Position sizing: 5% portfolio weight
        FINAL TRANSACTION PROPOSAL: HOLD
        """
        params = extract_trade_params("AAPL", "HOLD", 10.0, text)

        assert params.stop_loss == 258.0
        assert params.price_target == 295.0
        assert params.position_pct == 5.0
        assert not params.is_actionable  # HOLD is not actionable

    def test_all_cloud_sell_format(self):
        """Format from the all-cloud SELL output."""
        text = """
        Sell 60-70% of position at $272-275.
        Hard stop at $254 (February low).
        First target: $254. Second target: $230.
        Position size: Maximum 2-3% of portfolio.
        FINAL TRANSACTION PROPOSAL: SELL
        """
        params = extract_trade_params("AAPL", "SELL", 10.0, text, current_price=272.0)

        assert params.stop_loss == 254.0
        assert params.is_actionable

    def test_hold_not_actionable(self):
        """HOLD decisions should not be actionable."""
        params = extract_trade_params("AAPL", "HOLD", 10.0, "HOLD recommendation")
        assert not params.is_actionable

    def test_low_quality_not_actionable(self):
        """Low quality score should prevent execution."""
        text = "Stop-loss: $258. Target: $295. FINAL TRANSACTION PROPOSAL: BUY"
        params = extract_trade_params("AAPL", "BUY", 6.0, text)
        assert not params.is_actionable  # Quality < 8.0

    def test_missing_stop_loss(self):
        """Without stop-loss, BUY should not be actionable."""
        text = "Target: $295. Position: 5%. FINAL TRANSACTION PROPOSAL: BUY"
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert not params.is_actionable  # No stop-loss

    def test_risk_metrics_calculated(self):
        """Risk/reward should be calculated from entry, stop, target."""
        text = "Stop-loss: $258. Target: $295. FINAL TRANSACTION PROPOSAL: BUY"
        params = extract_trade_params("AAPL", "BUY", 9.5, text, current_price=270.0)

        assert params.risk_pct is not None
        assert params.reward_pct is not None
        assert params.risk_reward_ratio is not None
        assert params.risk_pct == pytest.approx(4.44, abs=0.1)
        assert params.reward_pct == pytest.approx(9.26, abs=0.1)

    def test_trailing_stop_format(self):
        """Trailing stop-loss phrasing."""
        text = "Set trailing stop-loss at 12% below. Trailing stop at $240.50."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        # Should extract the dollar value, not the percentage
        assert params.stop_loss == 240.50

    def test_empty_text(self):
        """Empty text should return params with no extracted values."""
        params = extract_trade_params("AAPL", "BUY", 9.5, "")
        assert params.stop_loss is None
        assert params.price_target is None
        assert params.position_pct is None
        assert not params.is_actionable

    def test_confidence_extraction(self):
        """Should extract confidence level."""
        text = "Confidence Level: HIGH (8.5/10). Stop-loss: $258."
        params = extract_trade_params("AAPL", "BUY", 9.5, text)
        assert params.confidence == "high"
```

---

## Step 3: Position Manager

### Goal
Track current positions, calculate order quantities, and enforce risk limits.

### Implementation

Create `src/execution/position_manager.py`:

```python
"""
Position manager for tracking holdings and calculating order quantities.

Interfaces with the Alpaca API to get real-time account and position data,
and applies risk controls before order submission.
"""

import logging
from dataclasses import dataclass
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
        from src.execution.trade_params import TradeParams

        rejection_reasons = []
        account = self.get_account_state()
        current_position = self.get_position(trade_params.ticker)

        # --- Determine side and entry price ---
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
```

### Tests

Create `tests/test_position_manager.py`:

```python
"""Tests for position management and order calculation."""

import pytest
from unittest.mock import MagicMock, patch
from src.execution.position_manager import (
    PositionManager, AccountState, Position, OrderCalculation,
    MAX_POSITION_PCT, MIN_QUALITY_SCORE, MAX_PORTFOLIO_RISK_PCT,
)
from src.execution.trade_params import TradeParams


def _mock_account(portfolio_value=100000, buying_power=50000, cash=50000):
    """Create a mock Alpaca account."""
    mock = MagicMock()
    mock.buying_power = str(buying_power)
    mock.portfolio_value = str(portfolio_value)
    mock.cash = str(cash)
    mock.equity = str(portfolio_value)
    return mock


def _mock_position(symbol="AAPL", qty=10, current_price=270.0):
    """Create a mock Alpaca position."""
    mock = MagicMock()
    mock.symbol = symbol
    mock.qty = str(qty)
    mock.current_price = str(current_price)
    mock.market_value = str(qty * current_price)
    mock.cost_basis = str(qty * 265.0)
    mock.unrealized_pl = str(qty * 5.0)
    mock.unrealized_plpc = str(0.0189)
    return mock


class TestPositionManager:

    def _make_pm(self, portfolio_value=100000, buying_power=50000, positions=None):
        """Create a PositionManager with mocked Alpaca client."""
        mock_client = MagicMock()
        mock_client.get_account.return_value = _mock_account(portfolio_value, buying_power)

        if positions:
            mock_client.get_all_positions.return_value = positions
            mock_client.get_open_position.side_effect = lambda t: next(
                (p for p in positions if p.symbol == t), None
            )
        else:
            mock_client.get_all_positions.return_value = []
            mock_client.get_open_position.side_effect = Exception("No position")

        return PositionManager(mock_client)

    def test_buy_order_calculation(self):
        """Standard BUY order should calculate quantity from position size."""
        pm = self._make_pm(portfolio_value=100000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, price_target=295.0,
            position_pct=5.0, risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        assert order.side == "buy"
        assert order.qty > 0
        assert order.approved
        assert order.stop_loss == 258.0
        assert order.take_profit == 295.0
        # 5% of $100K = $5000; $5000 / $270 = 18 shares
        assert order.qty == 18

    def test_sell_order_with_position(self):
        """SELL should sell existing shares."""
        positions = [_mock_position("AAPL", qty=20, current_price=270.0)]
        pm = self._make_pm(portfolio_value=100000, positions=positions)
        params = TradeParams(
            ticker="AAPL", decision="SELL", quality_score=9.5,
            entry_price=270.0, stop_loss=280.0, position_pct=5.0,
            risk_pct=3.7, reward_pct=10.0, risk_reward_ratio=2.7,
        )
        order = pm.calculate_order(params)

        assert order.side == "sell"
        assert order.qty <= 20  # Can't sell more than we have
        assert order.approved

    def test_sell_without_position_rejected(self):
        """SELL with no position should be rejected."""
        pm = self._make_pm()
        params = TradeParams(
            ticker="AAPL", decision="SELL", quality_score=9.5,
            entry_price=270.0, stop_loss=280.0,
        )
        order = pm.calculate_order(params)

        assert not order.approved
        assert any("No position" in r for r in order.rejection_reasons)

    def test_hold_not_actionable(self):
        """HOLD should produce no order."""
        pm = self._make_pm()
        params = TradeParams(ticker="AAPL", decision="HOLD", quality_score=10.0)
        order = pm.calculate_order(params)

        assert not order.approved
        assert order.qty == 0

    def test_low_quality_rejected(self):
        """Low quality score should reject the order."""
        pm = self._make_pm()
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=6.0,
            entry_price=270.0, stop_loss=258.0,
        )
        order = pm.calculate_order(params)

        assert not order.approved
        assert any("Quality score" in r for r in order.rejection_reasons)

    def test_no_stop_loss_rejected(self):
        """Missing stop-loss should reject the order."""
        pm = self._make_pm()
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=None,
        )
        order = pm.calculate_order(params)

        assert not order.approved
        assert any("stop-loss" in r.lower() for r in order.rejection_reasons)

    def test_position_size_capped(self):
        """Position should be capped at MAX_POSITION_PCT."""
        pm = self._make_pm(portfolio_value=100000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0,
            position_pct=25.0,  # Exceeds MAX_POSITION_PCT (15%)
            risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        # Should cap at 15%, not use 25%
        max_value = 100000 * (MAX_POSITION_PCT / 100)
        max_qty = int(max_value / 270.0)
        assert order.qty <= max_qty

    def test_insufficient_buying_power(self):
        """Should reduce qty if buying power is insufficient."""
        pm = self._make_pm(portfolio_value=100000, buying_power=1000)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, position_pct=5.0,
            risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        # $1000 buying power / $270 = 3 shares max
        assert order.qty <= 3

    def test_existing_position_reduces_buy(self):
        """Existing position should reduce buy quantity."""
        positions = [_mock_position("AAPL", qty=10, current_price=270.0)]
        pm = self._make_pm(portfolio_value=100000, positions=positions)
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            entry_price=270.0, stop_loss=258.0, position_pct=5.0,
            risk_pct=4.44, reward_pct=9.26, risk_reward_ratio=2.08,
        )
        order = pm.calculate_order(params)

        # Target: 5% of $100K = $5000. Already have 10 * $270 = $2700.
        # Remaining: $2300 / $270 = 8 shares
        assert order.qty == 8
```

---

## Step 4: Trade Executor

### Goal
Submit orders to Alpaca's paper trading API with full audit logging.

### Implementation

Create `src/execution/executor.py`:

```python
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
        from alpaca.trading.client import TradingClient

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
        from src.execution.position_manager import OrderCalculation
        from src.execution.trade_params import TradeParams

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
```

### Tests

Create `tests/test_executor.py`:

```python
"""Tests for the trade executor."""

import pytest
from unittest.mock import MagicMock, patch
from src.execution.trade_params import TradeParams
from src.execution.position_manager import OrderCalculation


class TestTradeExecutor:

    def test_rejected_order_not_submitted(self, tmp_path):
        """Rejected orders should not be submitted to Alpaca."""
        with patch("src.execution.executor.TradingClient"):
            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=0,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=0, position_pct_of_portfolio=0,
            risk_per_share=12.0, total_risk=0, portfolio_risk_pct=0,
            approved=False, rejection_reasons=["Quality too low"],
        )
        params = TradeParams(ticker="AAPL", decision="BUY", quality_score=6.0)

        result = executor.execute(order, params)

        assert result["action"] == "REJECTED"
        assert "Quality too low" in result["rejection_reasons"]
        # Verify no order was submitted
        executor._client.submit_order.assert_not_called()

    def test_approved_bracket_order_submitted(self, tmp_path):
        """Approved order with target should submit bracket order."""
        with patch("src.execution.executor.TradingClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_order = MagicMock()
            mock_order.id = "test-order-123"
            mock_order.status = "new"
            mock_instance.submit_order.return_value = mock_order

            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=18,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=4860.0, position_pct_of_portfolio=4.86,
            risk_per_share=12.0, total_risk=216.0, portfolio_risk_pct=0.216,
            approved=True, rejection_reasons=[],
        )
        params = TradeParams(
            ticker="AAPL", decision="BUY", quality_score=9.5,
            stop_loss=258.0, price_target=295.0,
        )

        result = executor.execute(order, params)

        assert result["action"] == "EXECUTED"
        assert result["alpaca_order_id"] == "test-order-123"
        executor._client.submit_order.assert_called_once()

    def test_paper_only_hardcoded(self, tmp_path):
        """Verify paper=True is hardcoded."""
        with patch("src.execution.executor.TradingClient") as MockClient:
            from src.execution.executor import TradeExecutor
            TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )
            # Verify paper=True was passed
            MockClient.assert_called_once_with(
                api_key="test", secret_key="test", paper=True
            )

    def test_audit_file_created(self, tmp_path):
        """Every execution should create an audit file."""
        with patch("src.execution.executor.TradingClient"):
            from src.execution.executor import TradeExecutor
            executor = TradeExecutor(
                api_key="test", secret_key="test",
                audit_dir=str(tmp_path / "audit"),
            )

        order = OrderCalculation(
            ticker="AAPL", side="buy", qty=0,
            entry_price=270.0, stop_loss=258.0, take_profit=295.0,
            position_value=0, position_pct_of_portfolio=0,
            risk_per_share=12.0, total_risk=0, portfolio_risk_pct=0,
            approved=False, rejection_reasons=["Test rejection"],
        )
        params = TradeParams(ticker="AAPL", decision="BUY", quality_score=6.0)

        executor.execute(order, params)

        audit_files = list((tmp_path / "audit").glob("*.json"))
        assert len(audit_files) == 1
```

---

## Step 5: Integration — Wire Into run_analysis.py

### Goal
Add a `--execute` flag that runs the full flow: analyze → extract params → calculate order → execute (or dry-run).

### Implementation

Add to `src/run_analysis.py`:

```python
# New CLI arguments:
parser.add_argument("--execute", action="store_true",
                    help="Execute the trade on Alpaca paper trading")
parser.add_argument("--dry-run", action="store_true",
                    help="Calculate order but don't submit (shows what would happen)")
```

After the analysis completes and the result is saved, add:

```python
    if args.execute or args.dry_run:
        from src.execution.trade_params import extract_trade_params
        from src.execution.position_manager import PositionManager
        from src.execution.executor import TradeExecutor

        # Extract trade parameters from the analysis
        trade_params = extract_trade_params(
            ticker=ticker,
            decision=decision,
            quality_score=score.composite_score,
            decision_text=final_trade_text,
            current_price=None,  # Will be extracted from text or fetched
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
            executor = TradeExecutor(audit_dir=str(Path(config["results_dir"]) / "audit"))
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

            result = executor.execute(order, trade_params)
            print(f"\n  Action: {result['action']}")
            if result.get("alpaca_order_id"):
                print(f"  Order ID: {result['alpaca_order_id']}")
        elif args.execute and not trade_params.is_actionable:
            print(f"\n  [NOT ACTIONABLE — {trade_params.decision}, score={trade_params.quality_score}]")
```

---

## Step 6: Final Verification

### Run All Unit Tests

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
pytest tests/ -v --ignore=tests/test_reasoning_comparison.py --ignore=tests/test_prompt_engineering.py
```

### Run a Dry-Run (no Alpaca account needed)

```bash
# Use an existing result to test the execution flow without running the pipeline
python -c "
from src.execution.trade_params import extract_trade_params
import json

with open('results/AAPL/analysis_2026-02-27_all_cloud.json') as f:
    result = json.load(f)

params = extract_trade_params(
    ticker=result['ticker'],
    decision=result['decision'],
    quality_score=result['quality_score']['composite'],
    decision_text=result['final_trade_decision_text'],
)

print(f'Decision: {params.decision}')
print(f'Stop-loss: \${params.stop_loss}')
print(f'Target: \${params.price_target}')
print(f'Position: {params.position_pct}%')
print(f'R/R Ratio: {params.risk_reward_ratio}')
print(f'Actionable: {params.is_actionable}')
print(f'Bracket: {params.has_bracket_params}')
"
```

### Commit

```bash
git add .
git commit -m "Add trade execution layer with position management and Alpaca paper trading"
git push
```

---

## Verification Checklist

- [ ] `alpaca-py` installed and added to pyproject.toml
- [ ] `src/execution/trade_params.py` — parameter extraction from pipeline text
- [ ] `src/execution/position_manager.py` — position tracking and order calculation
- [ ] `src/execution/executor.py` — Alpaca paper trading order submission
- [ ] `tests/test_trade_params.py` — at least 10 tests, all passing
- [ ] `tests/test_position_manager.py` — at least 8 tests, all passing
- [ ] `tests/test_executor.py` — at least 4 tests, all passing
- [ ] `src/run_analysis.py` — `--execute` and `--dry-run` flags added
- [ ] Dry-run test using existing result JSON produces correct parameter extraction
- [ ] `paper=True` is hardcoded in executor (NEVER configurable)
- [ ] Audit directory created and audit files written for every execution attempt
- [ ] All existing tests still pass
- [ ] No vendor code modified

---

## Report

After completing all steps, create `docs/TASK_007_REPORT.md` containing:

1. Which steps succeeded and which had issues
2. **Parameter extraction results** — test the extractor against all 5 existing result JSONs (AAPL all_cloud, AAPL hybrid_qwen, AAPL hybrid_qwen_enhanced, TSLA, JPM) and show what it extracts from each
3. **Order calculation examples** — show a mock order calculation for each result JSON
4. **Safety validation** — confirm paper=True is hardcoded, confirm all risk controls work
5. **Test output** — `pytest tests/ -v` (excluding comparison tests)
6. **Dry-run output** — result of the parameter extraction script above
7. `git log --oneline`
8. **Next steps** — what's needed to run the first paper trade (API key setup, account creation)
