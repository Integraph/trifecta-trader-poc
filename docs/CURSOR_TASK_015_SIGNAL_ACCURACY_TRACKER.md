# Task 015: Signal Accuracy Tracker

**Priority:** HIGH — Every day without accuracy data is a day of lost measurement
**Depends on:** Task 012 (portfolio database), Task 013 (signal adapter)
**Parallel with:** Task 014 (daemon mode — tracker integrates into the daemon loop)

---

## Objective

Build a system that measures whether the pipeline's trading signals are correct. For every signal that includes price targets (entry, stop-loss, take-profit), track the actual price movement over the following 1, 5, and 10 trading days and calculate whether the signal was accurate.

This creates the feedback loop the pipeline needs: generate signal → track actual prices → measure accuracy → identify what to improve.

---

## Background

The pipeline generates signals with:
- **decision:** BUY, SELL, or HOLD
- **entry_price:** Suggested entry point
- **stop_loss:** Downside risk boundary
- **price_target:** Upside target (takeProfit)
- **quality_score:** 0-10 composite quality rating

The `analyses` table in `data/portfolio.db` stores every signal with these fields. What's missing is:
1. Recording actual prices at T+1, T+5, T+10 days after the signal
2. Calculating whether the signal's direction was correct
3. Calculating whether the price target or stop-loss was hit
4. Aggregating accuracy metrics across all signals

---

## Deliverable 1: Price Outcome Tracker (`src/accuracy/price_tracker.py`)

### New SQLite Table

Add to the existing `data/portfolio.db` database:

```sql
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL REFERENCES analyses(id),
    ticker          TEXT    NOT NULL,
    signal_date     TEXT    NOT NULL,       -- Date the signal was generated
    decision        TEXT    NOT NULL,       -- BUY/SELL/HOLD
    entry_price     REAL,                   -- Signal's suggested entry
    stop_loss       REAL,                   -- Signal's stop-loss
    price_target    REAL,                   -- Signal's take-profit target

    -- Actual prices at checkpoints
    price_at_signal REAL,                   -- Close price on signal date
    price_t1        REAL,                   -- Close price T+1 trading day
    price_t5        REAL,                   -- Close price T+5 trading days
    price_t10       REAL,                   -- Close price T+10 trading days

    -- Price extremes during the tracking window
    high_t10        REAL,                   -- Highest price in T+1 to T+10
    low_t10         REAL,                   -- Lowest price in T+1 to T+10

    -- Calculated outcome fields (populated by accuracy scorer)
    direction_correct_t1   INTEGER,         -- 1 if price moved in signal direction at T+1
    direction_correct_t5   INTEGER,         -- 1 if price moved in signal direction at T+5
    direction_correct_t10  INTEGER,         -- 1 if price moved in signal direction at T+10
    target_hit             INTEGER,         -- 1 if price_target was reached within T+10
    stop_hit               INTEGER,         -- 1 if stop_loss was breached within T+10
    target_hit_first       INTEGER,         -- 1 if target was hit before stop (favorable outcome)
    return_t1_pct          REAL,            -- % return at T+1
    return_t5_pct          REAL,            -- % return at T+5
    return_t10_pct         REAL,            -- % return at T+10
    max_favorable_pct      REAL,            -- Best % move in signal direction within T+10
    max_adverse_pct        REAL,            -- Worst % move against signal direction within T+10

    -- Tracking status
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/partial/complete/error
    last_updated    TEXT NOT NULL,
    error_message   TEXT,

    UNIQUE(analysis_id)
);
```

### Class Design

```python
"""
Price Outcome Tracker — records actual market prices after each signal
to measure prediction accuracy.

Workflow:
1. After each analysis, create a pending outcome record
2. A daily job fetches actual prices for all pending/partial outcomes
3. Once T+10 data is available, the accuracy scorer calculates results
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class PriceTracker:
    """Tracks actual price outcomes for pipeline signals."""

    def __init__(self, db: PortfolioDatabase):
        """Uses the existing portfolio database connection."""

    def create_outcome(self, analysis_id: int, ticker: str, signal_date: str,
                       decision: str, entry_price: float = None,
                       stop_loss: float = None, price_target: float = None) -> int:
        """
        Create a pending outcome record for a new signal.
        Called automatically after each analysis.
        Returns the outcome row ID.
        """

    def update_prices(self, outcome_id: int, prices: dict) -> None:
        """
        Update price checkpoints for an outcome.

        prices dict keys: price_at_signal, price_t1, price_t5, price_t10,
                          high_t10, low_t10
        Only updates non-None values (allows incremental updates).
        """

    def get_pending_outcomes(self) -> list[dict]:
        """
        Get all outcomes that need price updates.
        Returns outcomes where status is 'pending' or 'partial'.
        """

    def get_outcome(self, analysis_id: int) -> Optional[dict]:
        """Get the outcome record for a specific analysis."""

    def get_outcomes_for_ticker(self, ticker: str, limit: int = 50) -> list[dict]:
        """Get recent outcomes for a ticker, ordered by signal_date desc."""

    def mark_complete(self, outcome_id: int) -> None:
        """Mark an outcome as complete (all T+10 data available)."""

    def mark_error(self, outcome_id: int, error: str) -> None:
        """Mark an outcome as errored (e.g., delisted ticker)."""
```

### Price Fetching

Use the existing yfinance integration to fetch historical prices:

```python
def fetch_outcome_prices(ticker: str, signal_date: str) -> dict:
    """
    Fetch actual prices for accuracy tracking.

    Uses yfinance to get daily OHLCV data from signal_date through T+10.
    Returns dict with:
    - price_at_signal: Close on signal_date
    - price_t1: Close on next trading day
    - price_t5: Close on 5th trading day after signal
    - price_t10: Close on 10th trading day after signal
    - high_t10: Highest high in the T+1 to T+10 window
    - low_t10: Lowest low in the T+1 to T+10 window

    Returns partial results if not all checkpoints are available yet
    (e.g., signal was 3 days ago, so T+5 and T+10 are not yet available).
    """
```

**Important:** Use trading days, not calendar days. T+5 means 5 trading days (approximately 1 week). T+10 means 10 trading days (approximately 2 weeks). yfinance returns only trading days, so index position in the OHLCV DataFrame is the correct approach.

### Integration with Pipeline

After each analysis completes (in `run_analysis.py` and `run_batch.py`), automatically create a pending outcome:

```python
# After analysis and optional Supabase publish:
if tracker_enabled:
    from src.accuracy.price_tracker import PriceTracker
    tracker = PriceTracker(db)
    tracker.create_outcome(
        analysis_id=analysis_id,
        ticker=result["ticker"],
        signal_date=result["trade_date"],
        decision=result["decision"],
        entry_price=trade_params.entry_price if trade_params else None,
        stop_loss=trade_params.stop_loss if trade_params else None,
        price_target=trade_params.price_target if trade_params else None,
    )
```

This should be **automatic** (not opt-in) — every analysis creates an outcome record. The price fetching happens later in a separate job.

---

## Deliverable 2: Accuracy Scorer (`src/accuracy/scorer.py`)

### Calculation Logic

```python
"""
Accuracy Scorer — calculates whether signals were correct based on
actual price movement.
"""

class AccuracyScorer:
    """Scores signal accuracy based on price outcomes."""

    def score_outcome(self, outcome: dict) -> dict:
        """
        Calculate all accuracy metrics for a single outcome.

        Returns dict of calculated fields to update in the database:
        - direction_correct_t1/t5/t10
        - target_hit, stop_hit, target_hit_first
        - return_t1/t5/t10_pct
        - max_favorable_pct, max_adverse_pct
        """
```

### Direction Accuracy

A signal's direction is correct if the price moved in the predicted direction:

```python
def _direction_correct(decision: str, entry: float, actual: float) -> bool:
    """
    BUY signal is correct if actual > entry (price went up)
    SELL signal is correct if actual < entry (price went down)
    HOLD is always 'correct' (neutral, no directional bet)
    """
    if decision == "HOLD":
        return True
    if decision == "BUY":
        return actual > entry
    if decision == "SELL":
        return actual < entry
    return False
```

### Target/Stop Hit Detection

```python
def _target_hit(decision: str, price_target: float,
                high_t10: float, low_t10: float) -> bool:
    """
    BUY target hit if high_t10 >= price_target (price reached upside target)
    SELL target hit if low_t10 <= price_target (price reached downside target)
    """

def _stop_hit(decision: str, stop_loss: float,
              high_t10: float, low_t10: float) -> bool:
    """
    BUY stop hit if low_t10 <= stop_loss (price breached downside stop)
    SELL stop hit if high_t10 >= stop_loss (price breached upside stop)
    """

def _target_hit_first(decision: str, price_target: float, stop_loss: float,
                      daily_highs: list, daily_lows: list) -> bool:
    """
    Determine whether the target or stop was hit first by scanning
    daily price data chronologically.

    This requires the full daily OHLCV series, not just high/low extremes.
    If both target and stop were hit on the same day, return False
    (conservative — assume stop was hit intraday first).
    """
```

### Return Calculations

```python
def _calculate_return(decision: str, entry: float, actual: float) -> float:
    """
    Calculate directional return percentage.

    BUY: (actual - entry) / entry * 100  (positive = profitable)
    SELL: (entry - actual) / entry * 100  (positive = profitable)
    """

def _max_favorable(decision: str, entry: float,
                   high_t10: float, low_t10: float) -> float:
    """
    Best move in the signal's direction within the tracking window.

    BUY: (high_t10 - entry) / entry * 100
    SELL: (entry - low_t10) / entry * 100
    """

def _max_adverse(decision: str, entry: float,
                 high_t10: float, low_t10: float) -> float:
    """
    Worst move against the signal's direction within the tracking window.

    BUY: (entry - low_t10) / entry * 100  (how far price dropped)
    SELL: (high_t10 - entry) / entry * 100  (how far price rallied)
    """
```

---

## Deliverable 3: Daily Update Job (`src/accuracy/updater.py`)

### Overview

A job that runs daily to fetch prices for all pending/partial outcomes and score them once complete.

```python
"""
Daily accuracy updater — fetches actual prices and scores outcomes.

Designed to run as a scheduled job in the daemon (Task 014), or
standalone via CLI.
"""

class AccuracyUpdater:
    """Orchestrates daily price fetching and accuracy scoring."""

    def __init__(self, db: PortfolioDatabase):
        self.tracker = PriceTracker(db)
        self.scorer = AccuracyScorer()

    def run_update(self) -> dict:
        """
        Full update cycle:
        1. Get all pending/partial outcomes
        2. For each, fetch latest available prices
        3. If T+10 is now available, run accuracy scoring
        4. Mark completed outcomes
        5. Return summary dict

        Returns:
        {
            "total_pending": 15,
            "updated": 12,
            "newly_complete": 5,
            "errors": 1,
            "skipped": 2
        }
        """

    def backfill(self, days_back: int = 30) -> dict:
        """
        Backfill outcomes for historical analyses that don't have
        outcome records yet. Scans the analyses table for entries
        within the last N days and creates + populates outcomes.

        Useful for: running once after Task 015 ships to score
        all existing signals retroactively.
        """
```

### Daemon Integration

Add the updater as a scheduled job in the daemon (alongside the watchlist scheduler):

```python
# In PipelineDaemon or config/automation.yaml:
accuracy:
  enabled: true
  update_hour: 17          # 5:00 PM ET — after market close
  update_minute: 0
  timezone: "US/Eastern"
  weekdays_only: true
  backfill_on_first_run: true
```

The updater runs daily after market close, when all prices are final. It fetches closing prices for the day and updates all pending outcomes.

### CLI Entry Point

```
python -m src.accuracy.updater                    # Run one update cycle
python -m src.accuracy.updater --backfill 30      # Backfill last 30 days
python -m src.accuracy.updater --ticker AAPL      # Update outcomes for one ticker
python -m src.accuracy.updater --report           # Print accuracy summary
```

---

## Deliverable 4: Accuracy Reporter (`src/accuracy/reporter.py`)

### Overview

Generates accuracy reports from the scored outcomes data.

```python
"""
Accuracy Reporter — generates human-readable reports and machine-readable
summaries of pipeline signal accuracy.
"""

class AccuracyReporter:
    """Generates accuracy reports from outcome data."""

    def __init__(self, db: PortfolioDatabase):
        pass

    def summary(self, days: int = 30) -> dict:
        """
        Aggregate accuracy metrics over the last N days.

        Returns:
        {
            "period_days": 30,
            "total_signals": 45,
            "by_decision": {
                "BUY": {
                    "count": 25,
                    "direction_correct_t1": 0.68,   # 68% correct at T+1
                    "direction_correct_t5": 0.72,   # 72% correct at T+5
                    "direction_correct_t10": 0.64,  # 64% correct at T+10
                    "target_hit_rate": 0.48,        # 48% hit their target
                    "stop_hit_rate": 0.28,          # 28% hit their stop
                    "avg_return_t5_pct": 1.8,       # Average 1.8% return at T+5
                    "avg_return_t10_pct": 2.1,      # Average 2.1% return at T+10
                    "avg_max_favorable_pct": 4.2,
                    "avg_max_adverse_pct": -2.8,
                    "target_before_stop_rate": 0.56 # 56% hit target before stop
                },
                "SELL": { ... },
                "HOLD": { ... }
            },
            "by_quality_tier": {
                "high (8-10)": {
                    "count": 12,
                    "direction_correct_t5": 0.83,   # High-quality signals are more accurate
                    "avg_return_t5_pct": 2.9
                },
                "medium (6-8)": { ... },
                "low (0-6)": { ... }
            },
            "best_signals": [
                {"ticker": "NVDA", "decision": "BUY", "return_t10_pct": 8.5, "quality_score": 9.4},
                ...
            ],
            "worst_signals": [
                {"ticker": "TSLA", "decision": "BUY", "return_t10_pct": -5.2, "quality_score": 7.8},
                ...
            ]
        }
        """

    def ticker_report(self, ticker: str) -> dict:
        """
        Accuracy report for a specific ticker across all signals.

        Returns signal history with outcomes for trend analysis.
        """

    def print_summary(self, days: int = 30) -> None:
        """
        Print a formatted accuracy summary to stdout.

        Example output:
        ═══════════════════════════════════════════════════
        TRIFECTA TRADER — Signal Accuracy Report (30 days)
        ═══════════════════════════════════════════════════
        Total signals: 45 | Complete: 38 | Pending: 7

        Direction Accuracy:
          BUY  (25 signals): T+1: 68% | T+5: 72% | T+10: 64%
          SELL (12 signals): T+1: 58% | T+5: 67% | T+10: 58%
          HOLD  (8 signals): —

        Target/Stop Performance:
          BUY:  Target hit: 48% | Stop hit: 28% | Target first: 56%
          SELL: Target hit: 42% | Stop hit: 33% | Target first: 50%

        Return by Quality Tier (T+5 avg):
          High  (8-10): +2.9%  (12 signals)
          Medium (6-8): +1.2%  (18 signals)
          Low    (0-6): -0.4%  ( 8 signals)

        Top 3 Best: NVDA +8.5% | AAPL +6.2% | MSFT +4.1%
        Top 3 Worst: TSLA -5.2% | GOOGL -3.8% | AMD -2.9%
        ═══════════════════════════════════════════════════
        """
```

### Quality Tier Correlation

One of the most valuable metrics: do higher quality scores predict better outcomes? The reporter should break down accuracy by quality tier to validate the scoring system:

- **High (8-10):** Signals the pipeline was most confident about
- **Medium (6-8):** Moderate confidence signals
- **Low (0-6):** Low confidence signals

If high-quality signals are significantly more accurate than low-quality ones, the scoring system is working. If there's no correlation, the quality scorer needs improvement.

---

## New Files

```
src/accuracy/
    __init__.py
    price_tracker.py          # Deliverable 1
    scorer.py                 # Deliverable 2
    updater.py                # Deliverable 3
    reporter.py               # Deliverable 4
tests/
    test_price_tracker.py     # Unit tests for outcome tracking
    test_accuracy_scorer.py   # Unit tests for accuracy calculations
    test_accuracy_updater.py  # Unit tests for daily update job
    test_accuracy_reporter.py # Unit tests for reporting
```

## Modified Files

```
src/portfolio/database.py     # Add signal_outcomes table to SCHEMA_SQL
src/run_analysis.py           # Auto-create outcome after each analysis
src/run_batch.py              # Auto-create outcomes for batch analyses
config/automation.yaml        # Add accuracy updater schedule
```

---

## Exit Criteria

1. `signal_outcomes` table created in `data/portfolio.db`
2. `PriceTracker.create_outcome()` creates a pending record linked to an analysis
3. `PriceTracker.update_prices()` updates price checkpoints incrementally
4. `PriceTracker.get_pending_outcomes()` returns only pending/partial records
5. `fetch_outcome_prices()` returns correct prices using yfinance trading days
6. Partial results returned when not all checkpoints are available yet (e.g., T+1 only)
7. `AccuracyScorer.score_outcome()` correctly calculates direction_correct for BUY signals
8. `AccuracyScorer.score_outcome()` correctly calculates direction_correct for SELL signals
9. HOLD signals always marked as direction_correct
10. Target hit detection works for both BUY and SELL
11. Stop hit detection works for both BUY and SELL
12. `target_hit_first` correctly identifies whether target or stop was hit first
13. Return percentages are directional (positive = profitable for both BUY and SELL)
14. `max_favorable_pct` and `max_adverse_pct` calculate correctly
15. `AccuracyUpdater.run_update()` fetches prices and scores in one pass
16. `AccuracyUpdater.backfill()` creates outcomes for existing analyses
17. `AccuracyReporter.summary()` returns correct aggregate metrics
18. Quality tier breakdown shows accuracy by score range
19. `print_summary()` produces formatted output
20. Outcome records auto-created after each `run_analysis` and `run_batch` run
21. Daily updater integrates with daemon schedule (config in automation.yaml)
22. CLI `--report` flag prints accuracy summary
23. CLI `--backfill N` creates and populates outcomes for last N days
24. All new tests pass
25. All existing tests still pass (299+ from Task 014)
26. Zero vendor modifications

---

## Testing Strategy

### Price Tracker Tests (`test_price_tracker.py`)
- Create outcome and verify all fields stored
- Update prices incrementally (T+1 first, then T+5, then T+10)
- Get pending outcomes filters correctly
- Duplicate analysis_id raises or handles gracefully
- mark_complete and mark_error status transitions

### Accuracy Scorer Tests (`test_accuracy_scorer.py`)
- BUY direction correct: price up at T+1, T+5, T+10
- BUY direction wrong: price down
- SELL direction correct: price down
- SELL direction wrong: price up
- HOLD always correct
- Target hit for BUY: high reached target
- Target hit for SELL: low reached target
- Stop hit for BUY: low breached stop
- Stop hit for SELL: high breached stop
- Target before stop chronological detection
- Return calculations for BUY and SELL
- Max favorable/adverse for BUY and SELL
- Missing entry_price handled gracefully (all direction/return fields null)
- Missing stop/target handled gracefully (hit fields null)

### Updater Tests (`test_accuracy_updater.py`)
- Mock yfinance to return known prices
- run_update processes pending outcomes
- Partial data (only T+1 available) updates correctly
- Complete data (T+10 available) triggers scoring
- Error handling (ticker not found / delisted)
- Backfill creates records for existing analyses

### Reporter Tests (`test_accuracy_reporter.py`)
- Summary with known outcome data returns correct aggregates
- Quality tier breakdown bins correctly
- Empty data returns zeros (not errors)
- Ticker report filters correctly
- Best/worst signals sorted correctly

---

## Notes

- **Why T+1/T+5/T+10?** T+1 tests immediate direction. T+5 (one trading week) tests short-term accuracy. T+10 (two trading weeks) tests medium-term accuracy and whether targets/stops get hit. These are standard signal evaluation windows in quantitative finance.
- **Trading days, not calendar days.** yfinance returns only trading days, so DataFrame index position 1/5/10 gives the correct checkpoints. No need to manually skip weekends/holidays.
- **Backfill is essential.** When Task 015 ships, we'll have existing analyses from Tasks 001-014 testing. The backfill command creates outcome records for all of those and populates prices retroactively, giving us an instant accuracy baseline.
- **The quality tier correlation is the most important metric.** If high-quality signals (8-10) don't outperform low-quality signals (0-6), our quality scoring formula needs recalibration. This data will drive future pipeline improvements.
- **HOLD signals don't affect accuracy calculations meaningfully** — they're always "correct" directionally. The interesting metrics are BUY and SELL accuracy. HOLD count is reported for completeness.
