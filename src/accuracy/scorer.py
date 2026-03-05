"""
Accuracy Scorer — calculates whether pipeline signals were correct based on
actual price movement recorded by PriceTracker.

All calculations are directional:
- BUY signals are correct when price rises
- SELL signals are correct when price falls
- HOLD signals are always 'correct' (neutral, no directional bet)

All return percentages are signed so that positive = profitable regardless
of whether the signal is BUY or SELL.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AccuracyScorer:
    """Scores signal accuracy based on price outcomes."""

    def score_outcome(self, outcome: dict, daily_highs: list = None,
                      daily_lows: list = None) -> dict:
        """Calculate all accuracy metrics for a single outcome record.

        Args:
            outcome: Row dict from signal_outcomes table.
            daily_highs: List of up to 10 daily high prices (T+1 … T+10).
                         Required for target_hit_first; other metrics use
                         only the stored high_t10/low_t10 extremes.
            daily_lows: Corresponding daily low prices.

        Returns:
            Dict of calculated fields to write back via PriceTracker.apply_scores().
            Only includes fields where sufficient data is available.
        """
        decision = (outcome.get("decision") or "HOLD").upper()
        entry    = outcome.get("entry_price")
        stop     = outcome.get("stop_loss")
        target   = outcome.get("price_target")
        p_t1     = outcome.get("price_t1")
        p_t5     = outcome.get("price_t5")
        p_t10    = outcome.get("price_t10")
        high10   = outcome.get("high_t10")
        low10    = outcome.get("low_t10")

        scores: dict = {}

        # ── Direction accuracy ────────────────────────────────────────────────
        if entry is not None:
            if p_t1 is not None:
                scores["direction_correct_t1"] = int(
                    _direction_correct(decision, entry, p_t1)
                )
            if p_t5 is not None:
                scores["direction_correct_t5"] = int(
                    _direction_correct(decision, entry, p_t5)
                )
            if p_t10 is not None:
                scores["direction_correct_t10"] = int(
                    _direction_correct(decision, entry, p_t10)
                )

        # ── Return percentages ────────────────────────────────────────────────
            if p_t1 is not None:
                scores["return_t1_pct"] = round(
                    _calculate_return(decision, entry, p_t1), 4
                )
            if p_t5 is not None:
                scores["return_t5_pct"] = round(
                    _calculate_return(decision, entry, p_t5), 4
                )
            if p_t10 is not None:
                scores["return_t10_pct"] = round(
                    _calculate_return(decision, entry, p_t10), 4
                )

        # ── Max favorable / adverse ───────────────────────────────────────────
            if high10 is not None and low10 is not None:
                scores["max_favorable_pct"] = round(
                    _max_favorable(decision, entry, high10, low10), 4
                )
                scores["max_adverse_pct"] = round(
                    _max_adverse(decision, entry, high10, low10), 4
                )

        # ── Target / stop hit ─────────────────────────────────────────────────
        if high10 is not None and low10 is not None:
            if target is not None:
                scores["target_hit"] = int(_target_hit(decision, target, high10, low10))
            if stop is not None:
                scores["stop_hit"] = int(_stop_hit(decision, stop, high10, low10))

            if (target is not None and stop is not None
                    and daily_highs is not None and daily_lows is not None):
                scores["target_hit_first"] = int(
                    _target_hit_first(decision, target, stop, daily_highs, daily_lows)
                )

        return scores


# ── Pure calculation helpers ──────────────────────────────────────────────────

def _direction_correct(decision: str, entry: float, actual: float) -> bool:
    """Return True if the price moved in the direction the signal predicted.

    BUY: correct when actual > entry (price went up).
    SELL: correct when actual < entry (price went down).
    HOLD: always correct (neutral — no directional bet).
    """
    if decision == "HOLD":
        return True
    if decision == "BUY":
        return actual > entry
    if decision == "SELL":
        return actual < entry
    return False


def _target_hit(decision: str, price_target: float,
                high_t10: float, low_t10: float) -> bool:
    """Return True if the price target was reached within T+10.

    BUY: target hit when high_t10 >= price_target (upside reached).
    SELL: target hit when low_t10 <= price_target (downside reached).
    """
    if decision == "BUY":
        return high_t10 >= price_target
    if decision == "SELL":
        return low_t10 <= price_target
    return False


def _stop_hit(decision: str, stop_loss: float,
              high_t10: float, low_t10: float) -> bool:
    """Return True if the stop-loss was breached within T+10.

    BUY: stop hit when low_t10 <= stop_loss (downside breached).
    SELL: stop hit when high_t10 >= stop_loss (upside breached).
    """
    if decision == "BUY":
        return low_t10 <= stop_loss
    if decision == "SELL":
        return high_t10 >= stop_loss
    return False


def _target_hit_first(decision: str, price_target: float, stop_loss: float,
                      daily_highs: list, daily_lows: list) -> bool:
    """Determine whether the target or stop was hit first chronologically.

    Scans T+1 … T+10 daily data day by day. If both target and stop are hit
    on the same day, returns False (conservative — assume stop hit intraday first).

    Args:
        decision: BUY or SELL.
        price_target: Signal's take-profit level.
        stop_loss: Signal's stop-loss level.
        daily_highs: List of daily high prices (T+1 first).
        daily_lows: List of daily low prices (T+1 first).

    Returns:
        True if target was hit before stop (favourable outcome).
    """
    for high, low in zip(daily_highs, daily_lows):
        if decision == "BUY":
            hit_target = high >= price_target
            hit_stop   = low  <= stop_loss
        elif decision == "SELL":
            hit_target = low  <= price_target
            hit_stop   = high >= stop_loss
        else:
            return False  # HOLD has no target/stop

        if hit_target and hit_stop:
            return False   # same day — conservatively assume stop first
        if hit_target:
            return True    # target reached before stop
        if hit_stop:
            return False   # stop hit first (bad outcome)

    return False  # neither hit within the window


def _calculate_return(decision: str, entry: float, actual: float) -> float:
    """Calculate directional return percentage.

    Positive values always mean profitable:
    BUY:  (actual - entry) / entry * 100
    SELL: (entry - actual) / entry * 100
    """
    if entry == 0:
        return 0.0
    if decision == "BUY":
        return (actual - entry) / entry * 100
    if decision == "SELL":
        return (entry - actual) / entry * 100
    return 0.0


def _max_favorable(decision: str, entry: float,
                   high_t10: float, low_t10: float) -> float:
    """Best move in the signal's direction within T+10 (always positive).

    BUY:  best upside  = (high_t10 - entry) / entry * 100
    SELL: best downside = (entry - low_t10) / entry * 100
    """
    if entry == 0:
        return 0.0
    if decision == "BUY":
        return (high_t10 - entry) / entry * 100
    if decision == "SELL":
        return (entry - low_t10) / entry * 100
    return 0.0


def _max_adverse(decision: str, entry: float,
                 high_t10: float, low_t10: float) -> float:
    """Worst move against the signal's direction within T+10 (always negative).

    BUY:  worst drawdown  = (entry - low_t10) / entry * 100  (stored as negative)
    SELL: worst rally     = (high_t10 - entry) / entry * 100 (stored as negative)
    """
    if entry == 0:
        return 0.0
    if decision == "BUY":
        return -abs((entry - low_t10) / entry * 100)
    if decision == "SELL":
        return -abs((high_t10 - entry) / entry * 100)
    return 0.0
