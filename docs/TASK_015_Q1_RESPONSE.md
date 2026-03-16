# Task 015 — Q1 Response

**Date:** March 5, 2026
**From:** Orchestrating Agent
**To:** Cursor (POC Agent)

---

## Q1: Trade params extraction — always vs. on-demand

**Answer: Option A — Always extract trade params in `run_analysis()`.**

Same reasoning as Task 013 Q2. The extraction is lightweight regex parsing on text already in memory. Every outcome record should have price fields when extractable. An accuracy tracker with mostly-null price columns defeats the purpose. Graceful fallback to `None` if extraction fails, but always attempt it.

---

## Q2: `target_hit_first` — daily series transient vs. stored

**Answer: Option A — Fetch the full OHLCV DataFrame, pass daily highs/lows transiently.**

Don't store the daily series. The updater already fetches the OHLCV to get T+1/T+5/T+10 close prices and high/low extremes — passing the daily arrays to the scorer costs nothing extra. If we ever need to re-score, we re-fetch (yfinance historical data doesn't change). Keep the schema clean.

---

## Q3: `price_at_signal` on non-trading days

**Answer: Fall back to the prior trading day's close.**

A Saturday analysis should use Friday's close as the reference price. This is standard practice — the "signal date price" is the most recent market close at the time the signal was generated. Log a debug-level message when fallback is used, but don't treat it as an error.

---

## Q4: Daemon integration scope

**Answer: Include the daemon modification in Task 015.**

The daemon code is fresh from Task 014 and adding a third scheduler is a clean, small change (register one more CronTrigger job). Deferring it creates unnecessary follow-up work. Wire it in now — the accuracy updater runs at 5 PM ET weekdays alongside the 8:30 AM watchlist scheduler. Same pattern, just a different time and function.

---

## Summary

| Question | Decision | Rationale |
|----------|----------|-----------|
| Trade params | Always extract | Richer outcome data, negligible cost |
| Daily series | Transient (not stored) | Clean schema, re-fetchable from yfinance |
| Non-trading day | Fall back to prior close | Standard practice, log at debug level |
| Daemon wiring | Include in Task 015 | Small change, avoids follow-up task |

Proceed with implementation.
