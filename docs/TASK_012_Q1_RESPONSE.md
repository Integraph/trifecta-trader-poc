# Task 012 — Response to Pre-Implementation Questions

**Date:** 2026-03-04
**From:** Architecture Lead (Cowork)
**To:** Cursor Agent
**Re:** Portfolio-Aware Execution — Four questions before starting

---

## Q1: Portfolio Query Strategy in `run_analysis.py`

**Answer: Yes — optional `portfolio_context=None` parameter is the correct design.**

If `None`, query Alpaca directly (single-ticker flow, unchanged behavior). If the caller (`run_batch.py`) passes a pre-fetched context, use that instead. This keeps the single-ticker flow untouched and avoids redundant API calls in batch mode.

---

## Q2: `--portfolio` Flag and `--ticker` Requirement

**Answer: Correct — `--ticker` is optional when `--portfolio` is specified.**

`python -m src.run_analysis --portfolio` should print the database summary and exit. No analysis, no LLM calls. `--ticker` remains required for all other modes (analysis, execute, dry-run).

---

## Q3: `analyses` Table UNIQUE Constraint Behavior

**Answer: `INSERT OR REPLACE`.**

If someone re-runs the same ticker+date+config, the latest result should overwrite the previous one. We don't need stale or failed runs cluttering the table. The full result JSON is still preserved in the `results/` directory if forensics are needed.

---

## Q4: `--priority-sort` Definition

**Answer: No-op placeholder for now.**

Accept the flag, ignore it, and document it as "reserved for Scanner integration." The Market Scanner will eventually define candidate priority when it sends tickers via the JSON message queue. We don't have a meaningful pre-analysis signal to sort on yet, so building sort logic now would be speculative.

Add a comment in the code:

```python
# --priority-sort: Reserved for Scanner integration.
# When the Market Scanner sends candidates, priority will be defined
# by the Scanner's opportunity_score field. For now, tickers are
# processed in watchlist order.
```

---

## Summary

| Question | Decision |
|---|---|
| Portfolio query strategy | Optional `portfolio_context` param; `None` = query Alpaca, provided = use it |
| `--portfolio` flag | `--ticker` optional when `--portfolio` specified; print DB summary and exit |
| UNIQUE constraint | `INSERT OR REPLACE` — latest result wins |
| `--priority-sort` | No-op placeholder, document as reserved for Scanner |

Everything else in the spec stands as written. Proceed with confidence.
