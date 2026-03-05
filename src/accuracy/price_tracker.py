"""
Price Outcome Tracker — records actual market prices after each signal
to measure prediction accuracy.

Workflow:
1. After each analysis, create_outcome() creates a pending record
2. AccuracyUpdater runs daily, fetches prices, calls update_prices()
3. Once T+10 data is available, AccuracyScorer calculates results
4. mark_complete() finalises the record

Price fetching uses yfinance daily OHLCV. Trading days are used for all
checkpoints — yfinance only returns trading days, so DataFrame index
positions 1/5/10 give the correct T+1/T+5/T+10 closes automatically.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


# ── Price fetching ────────────────────────────────────────────────────────────

def fetch_outcome_prices(ticker: str, signal_date: str) -> dict:
    """Fetch actual prices for accuracy tracking using yfinance.

    Returns a dict with available checkpoints. Keys with no data yet
    are absent (not None) so update_prices() can distinguish "not yet
    available" from "fetched and null".

    Args:
        ticker: Stock ticker symbol.
        signal_date: ISO date string (YYYY-MM-DD) the signal was generated.

    Returns:
        Dict with any subset of:
            price_at_signal, price_t1, price_t5, price_t10,
            high_t10, low_t10
        Plus the daily arrays needed by the scorer:
            daily_highs (list[float]), daily_lows (list[float])
    """
    # Download enough history: signal date + 15 calendar days covers T+10 trading days
    # even across weekends and holidays.
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(
        start=signal_date,
        period="25d",       # generous buffer for T+10 = ~14 calendar days
        auto_adjust=True,
    )

    if hist.empty:
        logger.warning("yfinance returned no data for %s from %s", ticker, signal_date)
        return {}

    prices: dict = {}

    # ── price_at_signal: first row available on or after signal_date ─────────
    # If signal_date is a non-trading day (weekend/holiday), the first row in
    # the DataFrame will be the next trading day's open session, but we want
    # the prior close. Check whether the first row's date matches signal_date.
    first_date = hist.index[0].date().isoformat()
    if first_date == signal_date:
        prices["price_at_signal"] = round(float(hist["Close"].iloc[0]), 4)
        # T+1 onwards are the rows after signal_date
        future = hist.iloc[1:]
    else:
        # signal_date was a non-trading day; first row is the next trading day.
        # Fall back: fetch one extra day prior to get the prior close.
        logger.debug(
            "%s: signal_date %s is a non-trading day; using prior close as price_at_signal",
            ticker, signal_date,
        )
        prior = ticker_obj.history(
            start=_subtract_days(signal_date, 7),
            end=signal_date,
            auto_adjust=True,
        )
        if not prior.empty:
            prices["price_at_signal"] = round(float(prior["Close"].iloc[-1]), 4)
        future = hist  # all rows are T+1 onwards

    # ── T+1, T+5, T+10 close prices ──────────────────────────────────────────
    closes = future["Close"]
    highs  = future["High"]
    lows   = future["Low"]

    if len(closes) >= 1:
        prices["price_t1"] = round(float(closes.iloc[0]), 4)
    if len(closes) >= 5:
        prices["price_t5"] = round(float(closes.iloc[4]), 4)
    if len(closes) >= 10:
        prices["price_t10"]  = round(float(closes.iloc[9]), 4)
        prices["high_t10"]   = round(float(highs.iloc[:10].max()), 4)
        prices["low_t10"]    = round(float(lows.iloc[:10].min()), 4)
        prices["daily_highs"] = [round(float(h), 4) for h in highs.iloc[:10]]
        prices["daily_lows"]  = [round(float(l), 4) for l in lows.iloc[:10]]

    return prices


def _subtract_days(date_str: str, days: int) -> str:
    """Return an ISO date string N days before date_str."""
    from datetime import date, timedelta
    d = date.fromisoformat(date_str)
    return (d - timedelta(days=days)).isoformat()


# ── PriceTracker ──────────────────────────────────────────────────────────────

class PriceTracker:
    """Tracks actual price outcomes for pipeline signals.

    Uses the existing PortfolioDatabase connection so all data lives in
    data/portfolio.db alongside analyses, orders, and portfolio_snapshots.
    """

    def __init__(self, db):
        """
        Args:
            db: PortfolioDatabase instance.
        """
        self._db = db

    def create_outcome(
        self,
        analysis_id: int,
        ticker: str,
        signal_date: str,
        decision: str,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        price_target: Optional[float] = None,
    ) -> int:
        """Create a pending outcome record for a new signal.

        Called automatically after each analysis run.

        Args:
            analysis_id: FK to analyses.id.
            ticker: Stock ticker symbol.
            signal_date: ISO date (YYYY-MM-DD) the signal was generated.
            decision: BUY / SELL / HOLD.
            entry_price: Signal's suggested entry price.
            stop_loss: Signal's stop-loss price.
            price_target: Signal's take-profit price target.

        Returns:
            The new outcome row ID.
        """
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT OR IGNORE INTO signal_outcomes (
                analysis_id, ticker, signal_date, decision,
                entry_price, stop_loss, price_target,
                status, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """
        with self._db._conn() as conn:
            cur = conn.execute(sql, (
                analysis_id, ticker, signal_date, decision,
                entry_price, stop_loss, price_target, now,
            ))
            row_id = cur.lastrowid
        logger.info(
            "Outcome created: id=%s  %s %s  %s  entry=%.2f  stop=%.2f  target=%.2f",
            row_id, decision, ticker, signal_date,
            entry_price or 0, stop_loss or 0, price_target or 0,
        )
        return row_id

    def update_prices(self, outcome_id: int, prices: dict) -> None:
        """Update price checkpoints for an outcome record.

        Only columns present in `prices` are updated (incremental).
        Automatically sets status to 'partial' if T+10 is not yet
        available, or 'complete' if it is.

        Args:
            outcome_id: Row ID in signal_outcomes.
            prices: Dict with any subset of price checkpoint keys.
        """
        allowed = {
            "price_at_signal", "price_t1", "price_t5", "price_t10",
            "high_t10", "low_t10",
        }
        updates = {k: v for k, v in prices.items() if k in allowed and v is not None}
        if not updates:
            return

        now  = datetime.now(timezone.utc).isoformat()
        # Determine new status
        new_status = "partial" if "price_t10" not in updates else "partial"
        # Will be set to 'complete' by mark_complete() after scoring

        set_clause = ", ".join(f"{col} = ?" for col in updates)
        vals = list(updates.values()) + [now, new_status, outcome_id]

        sql = f"""
            UPDATE signal_outcomes
               SET {set_clause}, last_updated = ?, status = ?
             WHERE id = ? AND status != 'complete'
        """
        with self._db._conn() as conn:
            conn.execute(sql, vals)

    def get_pending_outcomes(self) -> list:
        """Return all outcomes with status 'pending' or 'partial'."""
        sql = """
            SELECT * FROM signal_outcomes
             WHERE status IN ('pending', 'partial')
             ORDER BY signal_date ASC
        """
        with self._db._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_outcome(self, analysis_id: int) -> Optional[dict]:
        """Return the outcome record for a specific analysis, or None."""
        sql = "SELECT * FROM signal_outcomes WHERE analysis_id = ?"
        with self._db._conn() as conn:
            row = conn.execute(sql, (analysis_id,)).fetchone()
        return dict(row) if row else None

    def get_outcomes_for_ticker(self, ticker: str, limit: int = 50) -> list:
        """Return recent outcomes for a ticker, newest first."""
        sql = """
            SELECT * FROM signal_outcomes
             WHERE ticker = ?
             ORDER BY signal_date DESC
             LIMIT ?
        """
        with self._db._conn() as conn:
            rows = conn.execute(sql, (ticker, limit)).fetchall()
        return [dict(r) for r in rows]

    def mark_complete(self, outcome_id: int) -> None:
        """Mark an outcome as complete (all T+10 data available and scored)."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            UPDATE signal_outcomes
               SET status = 'complete', last_updated = ?
             WHERE id = ?
        """
        with self._db._conn() as conn:
            conn.execute(sql, (now, outcome_id))

    def mark_error(self, outcome_id: int, error: str) -> None:
        """Mark an outcome as errored (e.g. delisted ticker, no data)."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            UPDATE signal_outcomes
               SET status = 'error', error_message = ?, last_updated = ?
             WHERE id = ?
        """
        with self._db._conn() as conn:
            conn.execute(sql, (error, now, outcome_id))

    def apply_scores(self, outcome_id: int, scores: dict) -> None:
        """Write scorer output fields back to the outcome row.

        Args:
            outcome_id: Row ID in signal_outcomes.
            scores: Dict of calculated accuracy fields.
        """
        allowed = {
            "direction_correct_t1", "direction_correct_t5", "direction_correct_t10",
            "target_hit", "stop_hit", "target_hit_first",
            "return_t1_pct", "return_t5_pct", "return_t10_pct",
            "max_favorable_pct", "max_adverse_pct",
        }
        updates = {k: v for k, v in scores.items() if k in allowed}
        if not updates:
            return

        now = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{col} = ?" for col in updates)
        vals = list(updates.values()) + [now, outcome_id]
        sql = f"""
            UPDATE signal_outcomes
               SET {set_clause}, last_updated = ?
             WHERE id = ?
        """
        with self._db._conn() as conn:
            conn.execute(sql, vals)
