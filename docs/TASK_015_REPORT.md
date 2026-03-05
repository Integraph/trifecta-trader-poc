# Task 015 Report: Signal Accuracy Tracker

**Date:** 2026-03-05  
**Status:** Complete  
**Tests:** 93 new passing tests / 40 pre-existing failures (unchanged)

---

## Summary

Implemented a full signal accuracy tracking pipeline that measures whether the trading signals produced by the pipeline were correct by following actual price movement at T+1, T+5, and T+10 trading days after each analysis. All four deliverables are implemented and fully tested.

---

## Deliverables

### 1. `src/accuracy/__init__.py`
Package marker for the accuracy module.

### 2. `src/accuracy/price_tracker.py`
- **`PriceTracker` class** with six methods:
  - `create_outcome()` — creates a `pending` record in `signal_outcomes` after each analysis (using `INSERT OR IGNORE` to avoid duplicates on retry)
  - `update_prices()` — incremental price checkpoint updates, ignores `daily_highs`/`daily_lows` (transient only), skips `complete` rows
  - `get_pending_outcomes()` — returns all `pending` and `partial` rows ordered by signal date
  - `get_outcome()` — returns single row by `analysis_id`
  - `mark_complete()` / `mark_error()` — status transitions
  - `apply_scores()` — writes scorer output back to the row
- **`fetch_outcome_prices(ticker, signal_date)`** — uses `yfinance` to fetch OHLCV from `signal_date` through T+10 trading days
  - Returns partial results if T+10 not yet available
  - Falls back to prior trading day's close when `signal_date` is a non-trading day (weekend/holiday), logs at DEBUG level
  - Returns `{}` for unknown tickers or when yfinance returns no data
  - `daily_highs` / `daily_lows` are returned transiently (10-element lists) but not stored in DB

### 3. `src/accuracy/scorer.py`
- **`AccuracyScorer` class** with `score_outcome(outcome, daily_highs, daily_lows)` method
- **Pure helper functions** (all independently testable):
  - `_direction_correct(decision, entry, actual)` — BUY: actual > entry, SELL: actual < entry, HOLD: always True
  - `_target_hit(decision, price_target, high_t10, low_t10)`
  - `_stop_hit(decision, stop_loss, high_t10, low_t10)`
  - `_target_hit_first(decision, target, stop, daily_highs, daily_lows)` — chronological scan, conservative tie-breaking (same-day → False)
  - `_calculate_return(decision, entry, actual)` — signed, positive = profitable regardless of direction
  - `_max_favorable(decision, entry, high_t10, low_t10)` — always non-negative
  - `_max_adverse(decision, entry, high_t10, low_t10)` — always non-positive

### 4. `src/accuracy/updater.py`
- **`AccuracyUpdater` class**:
  - `run_update(ticker=None)` — fetches prices for all pending/partial outcomes, updates DB, scores and marks complete when T+10 is available, returns a summary dict with `{total_pending, updated, newly_complete, errors, skipped}`
  - `backfill(days_back=30)` — queries `analyses` LEFT JOIN `signal_outcomes` to find gaps, creates missing outcome records, then calls `run_update()`
- **CLI entry point** (`python -m src.accuracy.updater`) with flags:
  - `--backfill N` — backfill N days of history
  - `--ticker AAPL` — limit to one ticker
  - `--report [--days N]` — print accuracy summary via `AccuracyReporter`

### 5. `src/accuracy/reporter.py`
- **`AccuracyReporter` class**:
  - `summary(days=30)` — aggregates complete outcomes by decision and quality tier; includes best/worst 5 signals by T+10 return
  - `ticker_report(ticker)` — full signal history for a ticker with complete/pending counts
  - `print_summary(days=30)` — formatted stdout report
- **Quality tier correlation**: High (8-10), Medium (6-8), Low (0-6) breakdown to validate that quality scoring predicts accuracy

### 6. `src/portfolio/database.py` (modified)
Added `signal_outcomes` table to `SCHEMA_SQL`:
- FK to `analyses.id` with `UNIQUE(analysis_id)` constraint
- Price checkpoint columns: `price_at_signal`, `price_t1`, `price_t5`, `price_t10`, `high_t10`, `low_t10`
- Scored outcome columns: `direction_correct_t1/t5/t10`, `target_hit`, `stop_hit`, `target_hit_first`, `return_t1/t5/t10_pct`, `max_favorable_pct`, `max_adverse_pct`
- Status tracking: `status` (pending/partial/complete/error), `last_updated`, `error_message`

### 7. `src/run_analysis.py` (modified)
- **Always extracts trade parameters** inside `run_analysis()` (Option A from Q1 response) using `extract_trade_params_dual()`, regardless of `--publish`, `--execute`, or `--dry-run` flags
- Stores trade params in `result["trade_params"]` as a dict (graceful fallback to `None` on error)
- `tracker.log_analysis()` return value is now captured as `_analysis_id` so it's available for outcome creation
- **Auto-creates a pending `signal_outcome` record** after every analysis via `PriceTracker.create_outcome()`
- `main()` now re-uses `result["trade_params"]` instead of re-extracting for publish path

### 8. `src/run_batch.py` (no modification needed)
Outcomes are auto-created by `run_analysis()` on each per-ticker call. The batch runner requires no explicit changes.

### 9. `src/automation/daemon.py` (modified)
- Added `accuracy` key to `_CONFIG_DEFAULTS` with `update_hour: 17`, `update_minute: 0`, `timezone: "US/Eastern"`, `weekdays_only: True`, `backfill_on_first_run: True`
- New `_start_accuracy_scheduler(acc_cfg)` method wires the `AccuracyUpdater.run_update()` as a second `CronTrigger` job on the existing `APScheduler` instance at 5:00 PM ET, weekdays only
- `backfill_on_first_run=true` runs `AccuracyUpdater.backfill(90)` on daemon startup (non-fatal if it fails)

### 10. `config/automation.yaml` (modified)
Added `accuracy:` block:
```yaml
accuracy:
  enabled: true
  update_hour: 17
  update_minute: 0
  timezone: "US/Eastern"
  weekdays_only: true
  backfill_on_first_run: true
```

---

## Test Results

```
tests/test_price_tracker.py      25 passed
tests/test_accuracy_scorer.py    44 passed
tests/test_accuracy_updater.py   13 passed
tests/test_accuracy_reporter.py  11 passed
                                 ──────────
Total new                        93 passed / 0 failed

Full suite: 392 passed, 40 failed (pre-existing), 8 skipped
```

Pre-existing failures are all `ModuleNotFoundError: No module named 'langchain_google_genai'` — identical to the Task 014 baseline and unrelated to this task.

---

## Design Decisions Implemented (from Q1 response)

| Question | Choice Made |
|---|---|
| Trade param extraction | **Option A**: Always extract in `run_analysis()`, store in result dict |
| `target_hit_first` daily series | **Option A**: Fetch full OHLCV transiently; do NOT store in DB |
| `price_at_signal` on non-trading day | **Fall back to prior trading day's close**, log at DEBUG |
| Daemon integration scope | **Include in Task 015**: second CronTrigger at 5:00 PM ET |

---

## Exit Criteria Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | `signal_outcomes` table created in `data/portfolio.db` | ✅ |
| 2 | `price_tracker.py` implements `PriceTracker` class | ✅ |
| 3 | `create_outcome()` creates pending record | ✅ |
| 4 | `update_prices()` is incremental | ✅ |
| 5 | `update_prices()` does not overwrite complete rows | ✅ |
| 6 | `get_pending_outcomes()` returns pending + partial | ✅ |
| 7 | `fetch_outcome_prices()` uses yfinance | ✅ |
| 8 | Non-trading day fallback to prior close | ✅ |
| 9 | Partial results when T+10 not available | ✅ |
| 10 | `scorer.py` implements all 7 helper functions | ✅ |
| 11 | HOLD is always direction_correct | ✅ |
| 12 | Returns are signed (positive = profitable) | ✅ |
| 13 | `target_hit_first` uses daily arrays chronologically | ✅ |
| 14 | Same-day target+stop → conservative False | ✅ |
| 15 | `updater.py` `run_update()` end-to-end cycle | ✅ |
| 16 | `backfill()` uses LEFT JOIN to find gaps | ✅ |
| 17 | CLI `--backfill N`, `--ticker`, `--report` | ✅ |
| 18 | `reporter.py` `summary()` aggregates by decision | ✅ |
| 19 | Quality tier breakdown | ✅ |
| 20 | `ticker_report()` | ✅ |
| 21 | `print_summary()` formatted output | ✅ |
| 22 | `run_analysis.py` always extracts trade params | ✅ |
| 23 | Auto-creates outcome record after each analysis | ✅ |
| 24 | Daemon accuracy CronTrigger at 5:00 PM ET | ✅ |
| 25 | `config/automation.yaml` accuracy block | ✅ |
| 26 | 93 new tests passing, 0 new regressions | ✅ |

---

## Files Modified / Created

**New files:**
- `src/accuracy/__init__.py`
- `src/accuracy/price_tracker.py`
- `src/accuracy/scorer.py`
- `src/accuracy/updater.py`
- `src/accuracy/reporter.py`
- `tests/test_price_tracker.py`
- `tests/test_accuracy_scorer.py`
- `tests/test_accuracy_updater.py`
- `tests/test_accuracy_reporter.py`
- `docs/TASK_015_REPORT.md`

**Modified files:**
- `src/portfolio/database.py` — added `signal_outcomes` table to `SCHEMA_SQL`
- `src/run_analysis.py` — always extract trade params; auto-create outcome record
- `src/automation/daemon.py` — accuracy updater CronTrigger + defaults
- `config/automation.yaml` — added `accuracy:` block
