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


def _extract_from_structured_block(text: str) -> dict:
    """Extract parameters from the ## EXECUTION PARAMETERS block.

    Returns dict with keys: stop_loss, price_target, position_pct,
    entry_price, confidence, risk_reward_ratio.
    Any key may be None if not found.
    """
    result = {
        "stop_loss": None,
        "price_target": None,
        "position_pct": None,
        "entry_price": None,
        "confidence": None,
        "risk_reward_ratio": None,
    }

    block_match = re.search(
        r'##\s*EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return result

    block_text = block_match.group(1)

    stop_match = re.search(r'Stop-Loss:\s*\$?([\d,]+(?:\.\d+)?)', block_text, re.IGNORECASE)
    if stop_match:
        try:
            result["stop_loss"] = float(stop_match.group(1).replace(",", ""))
        except ValueError:
            pass

    target_match = re.search(r'Price Target:\s*\$?([\d,]+(?:\.\d+)?)', block_text, re.IGNORECASE)
    if target_match:
        try:
            result["price_target"] = float(target_match.group(1).replace(",", ""))
        except ValueError:
            pass

    entry_match = re.search(r'Entry Price:\s*\$?([\d,]+(?:\.\d+)?)', block_text, re.IGNORECASE)
    if entry_match:
        try:
            result["entry_price"] = float(entry_match.group(1).replace(",", ""))
        except ValueError:
            pass

    size_match = re.search(r'Position Size:\s*([\d.]+)%', block_text, re.IGNORECASE)
    if size_match:
        try:
            result["position_pct"] = float(size_match.group(1))
        except ValueError:
            pass

    rr_match = re.search(r'Risk/Reward Ratio:\s*([\d.]+)', block_text, re.IGNORECASE)
    if rr_match:
        try:
            result["risk_reward_ratio"] = float(rr_match.group(1))
        except ValueError:
            pass

    conf_match = re.search(r'Confidence:\s*(\w+)', block_text, re.IGNORECASE)
    if conf_match:
        conf = conf_match.group(1).lower()
        if conf in ("high", "strong"):
            result["confidence"] = "high"
        elif conf in ("low", "weak", "uncertain"):
            result["confidence"] = "low"
        else:
            result["confidence"] = "medium"

    return result


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

    # --- Try structured block first (highest priority) ---
    structured = _extract_from_structured_block(decision_text)
    if structured["stop_loss"] is not None:
        params.stop_loss = structured["stop_loss"]
    if structured["price_target"] is not None:
        params.price_target = structured["price_target"]
    if structured["position_pct"] is not None:
        params.position_pct = structured["position_pct"]
    if structured["entry_price"] is not None and params.entry_price is None:
        params.entry_price = structured["entry_price"]
    if structured["confidence"] is not None:
        params.confidence = structured["confidence"]
    if structured["risk_reward_ratio"] is not None:
        params.risk_reward_ratio = structured["risk_reward_ratio"]

    # --- Extract stop-loss (fallback regex if not found in structured block) ---
    if params.stop_loss is None:
        stop_patterns = [
            r'stop[- ]?loss[:\s]+\$?([\d,]+(?:\.\d+)?)(?!\s*%)',
            r'(?:hard|trailing)\s+stop[:\s]+(?:at\s+)?\$?([\d,]+(?:\.\d+)?)(?!\s*%)',
            r'stop[:\s]+(?:at\s+)?\$?([\d,]+(?:\.\d+)?)(?!\s*%)',
        ]
        for pattern in stop_patterns:
            for match in re.finditer(pattern, decision_text, re.IGNORECASE):
                try:
                    val = float(match.group(1).replace(",", ""))
                    if val > 10:
                        params.stop_loss = val
                        break
                except ValueError:
                    continue
            if params.stop_loss is not None:
                break

    # --- Extract price target (fallback regex) ---
    if params.price_target is None:
        target_patterns = [
            r'(?:price\s+)?target[:\s]+\$?([\d,]+(?:\.\d+)?)',
            r'target\s+(?:of\s+|at\s+)?\$?([\d,]+(?:\.\d+)?)',
            r'upside\s+(?:to\s+|target\s+)?\$?([\d,]+(?:\.\d+)?)',
            r'take[- ]?profit[:\s]+\$?([\d,]+(?:\.\d+)?)',
        ]
        for pattern in target_patterns:
            match = re.search(pattern, decision_text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1).replace(",", ""))
                    if val > 10:
                        params.price_target = val
                        break
                except ValueError:
                    continue

    # --- Extract position sizing (fallback regex) ---
    if params.position_pct is None:
        size_patterns = [
            r'([\d]+(?:\.[\d]+)?)\s*%\s*(?:of\s+)?(?:portfolio|allocation|position)',
            r'position\s+siz(?:e|ing)[:\s]+([\d]+(?:\.[\d]+)?)\s*%',
            r'allocat(?:e|ion)[:\s]+([\d]+(?:\.[\d]+)?)\s*%',
            r'portfolio\s+weight[:\s]+([\d]+(?:\.[\d]+)?)\s*%',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, decision_text, re.IGNORECASE)
            if match:
                try:
                    params.position_pct = float(match.group(1))
                    break
                except ValueError:
                    continue

    # --- Extract entry price (fallback regex if not already set) ---
    if params.entry_price is None:
        entry_patterns = [
            r'entry[:\s]+(?:at\s+)?\$?([\d,]+(?:\.\d+)?)',
            r'current\s+price[:\s]+\$?([\d,]+(?:\.\d+)?)',
            r'trading\s+at\s+\$?([\d,]+(?:\.\d+)?)',
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

    # Use extracted R/R from structured block if available; else calculate
    if params.risk_reward_ratio is None:
        if params.risk_pct and params.reward_pct and params.risk_pct > 0:
            params.risk_reward_ratio = params.reward_pct / params.risk_pct

    # --- Extract confidence (fallback regex) ---
    if params.confidence == "medium":
        confidence_match = re.search(
            r'(?:confidence|conviction)(?:\s+\w+)?:\s*(\w+)',
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
