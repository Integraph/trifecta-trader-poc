# Bug Fix Report — Scheduler Page, Logs Page, Test Run Failures, and Scheduler Runtime Crash

**Date:** 2026-03-17
**Bugs Fixed:** 6 (2 frontend blank-page crashes + 3 backend eager-import failures + 1 scheduler runtime crash)

---

## Overview

Five separate bugs were identified and fixed across two sessions:

1. **Scheduler page went blank** — caused by a JSON key mismatch between the API response and the frontend fetch call, plus secondary field name mismatches in the history table columns.
2. **Logs page went blank** — caused by the same pattern: a different JSON key mismatch.
3. **Test runs failed with `No module named 'langchain_google_genai'`** — caused by an eager top-level import of the Google SDK inside the LLM client factory, which crashed every analysis even when using Anthropic or Ollama providers.
4. **Test runs failed with `No module named 'stockstats'`** — caused by an eager top-level import of `stockstats` in `stockstats_utils.py`, pulled in at startup via `y_finance.py`.
5. **Test suite failing with `No module named 'rank_bm25'`** — caused by an eager top-level import of `rank_bm25` in `memory.py`, pulled in at startup via `agents/__init__.py`.

---

## Bug 1 — Scheduler Page: Blank on Navigation

### Symptom

Clicking **Scheduler** in the sidebar caused the page to go completely blank.

### Root Cause

Two related problems in `SchedulerPage.tsx`:

**Problem A — Wrong wrapper key:**

The backend endpoint `GET /scheduler/history` returns:
```json
{ "runs": [...] }
```

But the frontend fetched using:
```typescript
apiGet<{ history: SchedulerHistoryItem[] }>('/scheduler/history', { days })
  .then(r => setHistory(r.history))  // r.history === undefined
```

`r.history` resolved to `undefined`. React's initial state (`[]`) rendered fine, but once the API responded and `setHistory(undefined)` was called, the next render passed `rows={undefined}` to `DataTable`. Inside `DataTable`, line 39 spread `undefined`:

```typescript
const sorted = [...rows]  // TypeError: undefined is not iterable
```

This unhandled error crashed the entire component tree, leaving a blank page with no visible error (React has no Error Boundary in this app).

**Problem B — Field name mismatches in table columns:**

Even after fixing the wrapper key, each history row's field names didn't match what the frontend column definitions expected:

| Frontend column `key` | Actual API field | Result if uncorrected |
|---|---|---|
| `total_analyses` | `tickers_processed` | Column shows `—` for every row |
| `avg_quality` | `avg_quality_score` | Column shows `—` for every row |
| `elapsed_seconds` | `total_elapsed_seconds` | Column shows `—` for every row |
| `total_cost_usd` | `total_cost_usd` | ✓ already correct |
| `trade_date` | `trade_date` | ✓ already correct |

These mismatches were invisible to TypeScript because the `DataTable` rows were cast through `as unknown as (SchedulerHistoryItem & Record<string, unknown>)[]`, which makes every string key valid at compile time.

### Fix

**`admin-ui/src/components/scheduler/SchedulerPage.tsx`:**

```typescript
// Before
apiGet<{ history: SchedulerHistoryItem[] }>('/scheduler/history', { days })
  .then(r => setHistory(r.history))

// After
apiGet<{ runs: SchedulerHistoryItem[] }>('/scheduler/history', { days })
  .then(r => setHistory(r.runs ?? []))
```

The `?? []` guard ensures that even if the API shape changes again, the component renders an empty table rather than crashing.

Column key corrections in `HIST_COLS`:
- `total_analyses` → `tickers_processed`
- `avg_quality` → `avg_quality_score` (also fixed in `render` accessor)
- `elapsed_seconds` → `total_elapsed_seconds` (also fixed in `render` accessor)

**`admin-ui/src/api/types.ts`** — `SchedulerHistoryItem` interface updated to match actual API field names:

```typescript
// Before
export interface SchedulerHistoryItem {
  trade_date: string;
  total_analyses: number;
  decisions: Record<string, number>;
  avg_quality: number | null;
  total_cost_usd: number | null;
  elapsed_seconds: number | null;
}

// After
export interface SchedulerHistoryItem {
  trade_date:             string;
  tickers_processed:      number;
  decisions:              Record<string, number>;
  avg_quality_score:      number | null;
  total_cost_usd:         number | null;
  total_elapsed_seconds:  number | null;
}
```

### Files Changed
- `admin-ui/src/components/scheduler/SchedulerPage.tsx`
- `admin-ui/src/api/types.ts`

---

## Bug 2 — Logs Page: Blank on Navigation

### Symptom

Clicking **Logs** in the sidebar caused the page to go completely blank.

### Root Cause

The same key mismatch pattern as Bug 1, but in the Logs page.

The backend endpoint `GET /logs/recent` returns:
```json
{ "lines": [...] }
```

But the frontend fetched using:
```typescript
apiGet<{ logs: LogEntry[] }>('/logs/recent', { lines: 200, level: 'DEBUG' })
  .then(r => setLines(r.logs))  // r.logs === undefined
```

`r.logs` resolved to `undefined`, so `setLines(undefined)` set the `lines` state to `undefined`. On the next render:

```typescript
const filtered = lines.filter(...)  // TypeError: Cannot read properties of undefined
```

This crashed the component tree, producing the blank page.

### Fix

**`admin-ui/src/components/logs/LogsPage.tsx`:**

```typescript
// Before
apiGet<{ logs: LogEntry[] }>('/logs/recent', { lines: 200, level: 'DEBUG' })
  .then(r => setLines(r.logs))

// After
apiGet<{ lines: LogEntry[] }>('/logs/recent', { lines: 200, level: 'DEBUG' })
  .then(r => setLines(r.lines ?? []))
```

The `?? []` guard ensures a safe empty array fallback.

### Files Changed
- `admin-ui/src/components/logs/LogsPage.tsx`

---

## Bug 3 — Test Run: `No module named 'langchain_google_genai'`

### Symptom

Every test run submitted through the Admin UI (Test Run page) failed immediately with:

```
Analysis: TSLA failed
No module named 'langchain_google_genai'
```

This happened regardless of which hybrid config was selected — even configs that use only Anthropic and Ollama (like `hybrid_haiku_tools`) were affected.

### Root Cause

The LLM client factory `vendor/TradingAgents/tradingagents/llm_clients/factory.py` had a **module-level (eager) import** of `GoogleClient`:

```python
# factory.py — original, broken
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .google_client import GoogleClient          # ← executed on every import of factory.py
```

And `google_client.py` immediately imported the optional SDK at its own module level:

```python
# google_client.py
from langchain_google_genai import ChatGoogleGenerativeAI  # ← crashes if not installed
```

The result: **importing `factory.py` always triggered the Google SDK import**, even when creating an Anthropic or Ollama client. Since `langchain-google-genai` is not installed in this environment, the import raised `ModuleNotFoundError` and crashed the entire `create_llm_client()` call path before any LLM was even instantiated.

The execution path that hit this:

```
run_analysis()
  → create_hybrid_llms()
    → create_llm_client(provider="anthropic", ...)   # only anthropic requested
      → import factory.py                            # triggers eager GoogleClient import
        → import google_client.py                   # triggers langchain_google_genai import
          → ModuleNotFoundError                      # crash before any API call is made
```

### Fix

**`vendor/TradingAgents/tradingagents/llm_clients/factory.py`** — removed the top-level `GoogleClient` import and replaced it with a **lazy import** inside the `google` provider branch:

```python
# Before — eager import at module top level
from .google_client import GoogleClient

# Inside factory function:
if provider_lower == "google":
    return GoogleClient(model, base_url, **kwargs)
```

```python
# After — lazy import only when google provider is actually requested
if provider_lower == "google":
    try:
        from .google_client import GoogleClient
    except ImportError as e:
        raise ImportError(
            f"Google provider requires 'langchain-google-genai'. "
            f"Install it with: pip install langchain-google-genai\n"
            f"Original error: {e}"
        ) from e
    return GoogleClient(model, base_url, **kwargs)
```

The `try/except ImportError` re-raises with a **clear, actionable error message** telling the developer exactly which package to install, rather than surfacing a raw `ModuleNotFoundError` deep in the stack.

### Why Other Providers Weren't Affected

`AnthropicClient` and `OpenAIClient` have no optional SDK dependencies — `langchain-anthropic` and `langchain-openai` are always installed. Only the `GoogleClient` depends on the optional `langchain-google-genai` package.

### Side Effect: 40 Previously-Failing Tests Now Pass

The Task 019 spec documented "40 pre-existing Google GenAI failures." Those test failures were caused by the same eager import — test modules that imported any part of the LLM pipeline would inadvertently trigger `google_client.py` → `langchain_google_genai`, failing before any test logic ran. The lazy import fix resolved all of them.

### Files Changed
- `vendor/TradingAgents/tradingagents/llm_clients/factory.py`

---

---

## Bug 4 — Test Run: `No module named 'stockstats'`

### Symptom

After the Google GenAI lazy import fix (Bug 3), test runs still failed immediately with:

```
Analysis: TSLA failed
No module named 'stockstats'
```

Again, this happened regardless of the hybrid config selected.

### Root Cause

`vendor/TradingAgents/tradingagents/dataflows/stockstats_utils.py` had a **module-level import** of `stockstats`:

```python
# stockstats_utils.py — original, broken
from stockstats import wrap   # ← executed at module import time
```

`y_finance.py` imported `StockstatsUtils` from this file **at module level**:

```python
# y_finance.py — original, broken
from .stockstats_utils import StockstatsUtils  # ← triggers stockstats import
```

The execution chain:

```
run_analysis()
  → import y_finance              # triggers module-level import of StockstatsUtils
    → import stockstats_utils     # triggers module-level: from stockstats import wrap
      → ModuleNotFoundError       # crash — stockstats is not installed
```

Since `stockstats` is not installed in this environment, the import raised `ModuleNotFoundError` before any analysis logic ran.

### Fix

**`vendor/TradingAgents/tradingagents/dataflows/stockstats_utils.py`** — removed the top-level `from stockstats import wrap` and moved it to a **lazy import inside `get_stock_stats`**:

```python
# Before — eager import at module top level
from stockstats import wrap

# After — lazy import with actionable error message
def get_stock_stats(self, symbol, indicator, curr_date):
    try:
        from stockstats import wrap
    except ImportError as e:
        raise ImportError(
            "stockstats is required for technical indicator calculations. "
            "Install it with: pip install stockstats\n"
            f"Original error: {e}"
        ) from e
    ...
```

**`vendor/TradingAgents/tradingagents/dataflows/y_finance.py`** — removed the module-level `from .stockstats_utils import StockstatsUtils` and moved it **inside `get_stockstats_indicator`**, the only function that uses it:

```python
# Before — module-level
from .stockstats_utils import StockstatsUtils

# After — lazy, inside the function that needs it
def get_stockstats_indicator(symbol, indicator, curr_date):
    from .stockstats_utils import StockstatsUtils  # lazy: stockstats is optional
    ...
```

### Files Changed
- `vendor/TradingAgents/tradingagents/dataflows/stockstats_utils.py`
- `vendor/TradingAgents/tradingagents/dataflows/y_finance.py`

---

## Bug 5 — Test Suite: `No module named 'rank_bm25'`

### Symptom

After fixing Bugs 3 and 4, the test suite still showed 38 failures across `test_config.py`, `test_cost_optimization.py`, `test_hybrid_llm.py`, `test_pipeline.py`, and `test_portfolio.py`, all with:

```
ModuleNotFoundError: No module named 'rank_bm25'
```

### Root Cause

`vendor/TradingAgents/tradingagents/agents/utils/memory.py` had a **module-level import** of `rank_bm25`:

```python
# memory.py — original, broken
from rank_bm25 import BM25Okapi  # ← executed at module import time
```

`tradingagents/agents/__init__.py` re-exported `FinancialSituationMemory` at its top level:

```python
# agents/__init__.py
from .utils.memory import FinancialSituationMemory  # ← triggers rank_bm25 import
```

This meant importing **anything from `tradingagents.agents`** — including the graph, agents, or the main pipeline — triggered the `rank_bm25` import. Since `rank_bm25` is not installed in this environment, all such tests failed before any test logic ran.

### Fix

**`vendor/TradingAgents/tradingagents/agents/utils/memory.py`** — removed the top-level import and introduced a `_get_bm25()` lazy loader:

```python
# Before — eager import at module top level
from rank_bm25 import BM25Okapi

# After — lazy loader function
def _get_bm25():
    """Lazy loader for BM25Okapi to avoid eager import of rank_bm25."""
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except ImportError as e:
        raise ImportError(
            "rank_bm25 is required for financial memory retrieval. "
            "Install it with: pip install rank-bm25\n"
            f"Original error: {e}"
        ) from e
```

The only internal call site (`_rebuild_index`) was updated to call through the loader:

```python
# Before
self.bm25 = BM25Okapi(tokenized_docs)

# After
BM25Okapi = _get_bm25()
self.bm25 = BM25Okapi(tokenized_docs)
```

### Side Effect: 38 Previously-Failing Tests Now Pass

Exactly 38 test cases that failed with `rank_bm25 ModuleNotFoundError` now pass, matching the reduction from 40 failures → 2 failures in the final run.

### Files Changed
- `vendor/TradingAgents/tradingagents/agents/utils/memory.py`

---

## Test Results After All Fixes

**Session 1 (Bugs 1–3):**

```
Total tests run:   593
Passed:            546
Failed:             40  (38 × rank_bm25 + 2 × local Ollama format)
Skipped:             8
Resolved:           40  (Google GenAI failures eliminated by lazy import fix)
```

**Session 2 (Bugs 4–5):**

```
Total tests run:   606
Passed:            596
Failed:              2  (both in test_local_tool_calling.py — Ollama model output format, unrelated to code changes)
Skipped:             8
Resolved:           38  (rank_bm25 + stockstats failures eliminated by lazy import fixes)
```

The 2 remaining failures (`test_tool_calling_basic[mistral-small:22b]` and `test_tool_calling_multi_tool[mistral-small:22b]`) are caused by the local Ollama `mistral-small:22b` model returning JSON-formatted tool calls as text rather than structured function calls — this is a model behavior issue, not a code defect.

## Build Verification

```
tsc --noEmit:  ✓ 0 errors
npm run build: ✓ 0 errors
```

---

---

## Bug 6 — Scheduler Watchlist Scan Crash: `'list' object has no attribute 'upper'`

### Symptom

Every scheduled watchlist scan (and every "Run Watchlist Now" trigger) failed with:

```
ERROR src.automation.scheduler  Watchlist scan failed: 'list' object has no attribute 'upper'
```

In the Scheduler page UI, **Last Run** showed `Result: error`, `Tickers: 0`.
The signal adapter still managed to build one signal for a ticker called `"Default Watchlist"`
(the watchlist name itself being treated as a ticker symbol).

### Root Cause

`src/run_batch.py` `load_watchlist()` has always returned a **2-element tuple** `(name, tickers)`:

```python
def load_watchlist(path: str) -> tuple[str, list[str]]:
    ...
    return name, [t.upper().strip() for t in tickers if t]
```

The CLI entry point (`run_batch.py` line 441) correctly unpacks this:

```python
watchlist_name, tickers = load_watchlist(args.watchlist)  # ← correct
```

But `src/automation/scheduler.py` assigned the return value to a single variable:

```python
tickers = load_watchlist(watchlist_path)   # ← bug: tickers is (name, [list])
```

`tickers` was now the full tuple `("Default Watchlist", ["AAPL", "MSFT", ...])`.
When `run_batch` iterated `for i, ticker in enumerate(tickers)`:

| Iteration | `ticker` value | `.upper()` result |
|-----------|---------------|-------------------|
| 0 | `"Default Watchlist"` (string) | `"DEFAULT WATCHLIST"` — accepted as a ticker, producing the bogus signal |
| 1 | `["AAPL", "MSFT", ...]` (the list) | **`AttributeError: 'list' object has no attribute 'upper'`** → crash |

This also caused `watchlist_name` passed to `run_batch` to use the YAML file stem
(`"default"`) rather than the display name (`"Default Watchlist"`), slightly
misreporting the batch summary.

### Fix

**`src/automation/scheduler.py`** — one-line tuple unpack:

```python
# Before
tickers = load_watchlist(watchlist_path)

# After — matches how run_batch.py CLI entry point has always called it
watchlist_display_name, tickers = load_watchlist(watchlist_path)
```

`watchlist_display_name` is then forwarded to `run_batch(watchlist_name=watchlist_display_name, ...)`
so batch summaries and signal IDs use the human-readable name from the YAML file.

**`tests/test_scheduler.py`** — two changes:

1. Corrected the existing mock in `test_handles_run_batch_exception` from
   `return_value=["AAPL"]` to `return_value=("Default Watchlist", ["AAPL"])` — the mock
   was masking the bug because it was returning a plain list instead of the real tuple.

2. Added a new regression test `test_load_watchlist_tuple_is_unpacked_correctly` that
   captures what `run_batch` receives and asserts:
   - `tickers` is `["AAPL", "MSFT"]` (the list, not the tuple)
   - `watchlist_name` is `"Default Watchlist"` (the display name, not the file stem)

### Why This Wasn't Caught Earlier

The pre-existing `test_handles_run_batch_exception` mocked `load_watchlist` to return
`["AAPL"]` (a plain list) instead of `("Default Watchlist", ["AAPL"])` (the correct tuple).
This mismatch meant the test exercised the wrong code path — the bug only surfaced when
the real `load_watchlist` was called with a real watchlist file at 08:30.

### Test Results

```
tests/test_scheduler.py          20 passed  (was 15 — 5 new/updated)
tests/test_admin_scheduler.py     8 passed
Total: 28 passed in 0.58s
```

### Files Changed
- `src/automation/scheduler.py`
- `tests/test_scheduler.py`

---

## Summary Table

| Bug | Page/Feature | Root Cause | Fix |
|-----|-------------|-----------|-----|
| 1a | Scheduler UI | `r.history` → should be `r.runs` | Changed fetch key; added `?? []` guard |
| 1b | Scheduler UI | Column keys don't match API field names | Updated `HIST_COLS` keys + `SchedulerHistoryItem` type |
| 2  | Logs UI | `r.logs` → should be `r.lines` | Changed fetch key; added `?? []` guard |
| 3  | Test Run (all) | Eager `GoogleClient` import crashes non-Google providers | Made `GoogleClient` import lazy inside `factory.py` |
| 4  | Test Run (all) | Eager `stockstats` import in `stockstats_utils.py` via `y_finance.py` | Made import lazy inside `get_stock_stats` and `get_stockstats_indicator` |
| 5  | All tests | Eager `rank_bm25` import in `memory.py` via `agents/__init__.py` | Introduced `_get_bm25()` lazy loader; call inside `_rebuild_index` |
| 6  | Scheduler runtime | `load_watchlist()` tuple not unpacked — whole tuple passed as `tickers` | Unpacked to `(watchlist_display_name, tickers)` in `scheduler.py` |

---

## Prevention

Bugs 1 and 2 (key mismatches) were not caught by TypeScript because `DataTable` accepts `Record<string, unknown>` rows. A runtime validation step or a shared API response type would prevent this class of bug.

Bugs 3, 4, and 5 share the same root cause: optional dependencies imported eagerly at module level. The Python convention — never import optional SDKs at module level; always import lazily — eliminates this class of failure.

Bug 6 (tuple unpack) was masked by a test mock that returned the wrong type (`list` instead of `tuple`). Mocks should always match the real function's return type signature precisely — mismatched mocks allow bugs to silently pass the test suite while failing in production.
