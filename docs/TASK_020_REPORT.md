# Task 020 Report — Admin UI End-to-End Bug Fixes

**Date:** 2026-03-17
**Bugs Fixed:** 6 (1 blocker + 3 medium + 1 medium-enhancement + 1 low)
**Tests updated:** 4 (test_admin_health.py — signature + behavior change)

---

## Summary

All six bugs from the Task 020 specification were identified, fixed, and verified.
The fixes span three backend files, five frontend files, one new backend module,
and one updated test file.

---

## Bug 1 — BLOCKER: `run_analysis()` Does Not Accept `publish` Keyword

### Root Cause

`src/admin/test_run.py` line 214 passed `publish=publish` to `run_analysis()`,
but that function has no `publish` parameter. Publishing is handled separately via
the module-level `_publish_signal()` function after the analysis completes.

### Fix

**`src/admin/test_run.py` — `_run_analysis_safe()`:**

```python
# Before — crash on every call
result = run_analysis(
    ticker=ticker, trade_date=trade_date, hybrid=hybrid_config,
    use_cache=True, cost_breakdown=True,
    publish=publish,   # ← TypeError: unexpected keyword argument
    debug=False,
)
# ...
"published": publish,   # ← always reflected the request, not the outcome
```

```python
# After — correct call + actual publish logic
from src.run_analysis import run_analysis, _publish_signal

result = run_analysis(
    ticker=ticker, trade_date=trade_date, hybrid=hybrid_config,
    use_cache=True, cost_breakdown=True, debug=False,
)

published = False
if publish:
    try:
        trade_params = result.get("trade_params_obj")
        _publish_signal(result, trade_params)
        published = True
    except Exception as e:
        logger.warning("Publish failed for %s: %s", ticker, e)

# ...
"published": published,   # ← reflects actual outcome
```

### Files Changed
- `src/admin/test_run.py`

---

## Bug 2 — Analyses Detail Panel Never Refreshes

### Root Cause

`admin-ui/src/components/analyses/AnalysesPage.tsx` `DetailPanel` component used
`useState()` with an async callback to fire an API request. `useState`'s initializer
runs exactly once during initial mount and does not re-run when the `id` prop changes.
It also runs synchronously during the render phase, making any async operations inside
it unreliable.

### Fix

**`admin-ui/src/components/analyses/AnalysesPage.tsx`:**

```tsx
// Before — wrong hook, no dependency, never re-fetches
import { useState } from 'react';
// ...
useState(() => {
  setLoading(true);
  apiGet<AnalysisDetail>(`/analyses/${id}`)
    .then(r => setDetail(r))
    .catch(() => {})
    .finally(() => setLoading(false));
});

// After — correct hook with [id] dependency array
import { useEffect, useState } from 'react';
// ...
useEffect(() => {
  setLoading(true);
  apiGet<AnalysisDetail>(`/analyses/${id}`)
    .then(r => setDetail(r))
    .catch(() => {})
    .finally(() => setLoading(false));
}, [id]);
```

### Files Changed
- `admin-ui/src/components/analyses/AnalysesPage.tsx`

---

## Bug 3 — Configuration Page: Scheduler Hour/Minute Field Name Mismatch

### Root Cause

The YAML config (`config/automation.yaml`) and the backend parser (`src/admin/config.py`)
use `watchlist_hour` / `watchlist_minute` as field names. The frontend
`ConfigPage.tsx` was reading and writing `hour` / `minute` instead — a different pair
of keys. Saves appeared to work but wrote the wrong YAML keys, which the scheduler
daemon never read.

### Fix

**`admin-ui/src/components/config/ConfigPage.tsx` (lines 119–120):**

```tsx
// Before — wrong field names
<FieldRow label="Hour (24h)" value={s.hour ?? 8}   onChange={v => set(['scheduler','hour'], v)} ... />
<FieldRow label="Minute"     value={s.minute ?? 30} onChange={v => set(['scheduler','minute'], v)} ... />

// After — correct field names matching YAML / backend
<FieldRow label="Hour (24h)" value={s.watchlist_hour ?? 8}   onChange={v => set(['scheduler','watchlist_hour'], v)} ... />
<FieldRow label="Minute"     value={s.watchlist_minute ?? 30} onChange={v => set(['scheduler','watchlist_minute'], v)} ... />
```

**`admin-ui/src/api/types.ts` — `AutomationConfig.scheduler`:**
Renamed `hour` / `minute` → `watchlist_hour` / `watchlist_minute` to match the API.

**`admin-ui/src/components/config/config-help.ts`:**
Renamed tooltip keys `scheduler.hour` / `scheduler.minute` → `scheduler.watchlist_hour` / `scheduler.watchlist_minute`.

### Files Changed
- `admin-ui/src/components/config/ConfigPage.tsx`
- `admin-ui/src/components/config/config-help.ts`
- `admin-ui/src/api/types.ts`

---

## Bug 4 — Queue Page: Score Column Always Shows "—"

### Root Cause

The backend `src/admin/queue.py` returns the score field as `opportunity_score`.
The frontend `QueuePage.tsx` column definition used `key: 'score'` and `r.score`,
which is always `undefined` because the field is named `opportunity_score`.

### Fix

**`admin-ui/src/components/queue/QueuePage.tsx`:**

```tsx
// Before — wrong key, always undefined
{ key: 'score', label: 'Score', sortable: true,
  render: r => <>{r.score != null ? (r.score as number).toFixed(2) : '—'}</> },

// After — correct key matching backend field name
{ key: 'opportunity_score', label: 'Score', sortable: true,
  render: r => <>{r.opportunity_score != null ? (r.opportunity_score as number).toFixed(2) : '—'}</> },
```

### Files Changed
- `admin-ui/src/components/queue/QueuePage.tsx`

---

## Bug 5 — "Degraded" Status Badge When API Is Healthy

### Root Cause (Two separate issues)

**Issue A — Standalone mode (no daemon attached):**

When the API was started with `python -m src.admin.app` (no daemon),
`compute_health_color()` received `daemon_running=False`, immediately returning
`("red", "unhealthy")`. The badge showed "Unhealthy" even though standalone operation
is a perfectly normal state.

**Issue B — Daemon mode with unconfigured optional integrations:**

When the daemon was running, `compute_health_color()` had two yellow conditions that
fired aggressively:

```python
# Old — degrades any time Supabase isn't write-enabled (even if not configured)
if not supabase_write_enabled:
    return "yellow", "degraded"

# Old — degrades any time Ollama is not reachable (even if not in use)
if not ollama_reachable:
    return "yellow", "degraded"
```

Both Supabase and Ollama are optional integrations. When not configured, they should
not cause the pipeline to report as degraded. A default deployment without Supabase
credentials or a running Ollama instance should show **green/Healthy**.

### Fix

**`src/admin/health.py` — `compute_health_color()`:**

```python
# Removed: ollama_reachable parameter entirely
# Changed: supabase condition now requires configured=True to degrade

# Before
if not supabase_write_enabled:
    return "yellow", "degraded"
if not ollama_reachable:
    return "yellow", "degraded"

# After — only degrade if Supabase is configured but write is explicitly off
if supabase_configured and not supabase_write_enabled:
    return "yellow", "degraded"
# Ollama unreachable: reported in subsystems only, does not affect overall status
```

**`src/admin/health.py` — `get_health()` standalone early-return:**

```python
# New: detect standalone mode and return immediately with a neutral response
mode = "full" if daemon_running else "standalone"
if not daemon_running:
    return {
        "status": "standalone",
        "color":  "blue",
        "mode":   "standalone",
        ...
    }
```

**Frontend — `StatusDot.tsx`:** Added `'blue'` color (`bg-blue-400`).

**Frontend — `admin-ui/src/api/types.ts`:** Added `'blue'` to `HealthResponse.color`
and `mode?: 'standalone' | 'full'` field.

**Frontend — `HealthBadge.tsx`:** Added label mapping (`standalone` → `"Standalone"`),
handled new `'blue'` color, and shows a `(dev)` tag in standalone mode.

### Status behavior after fix

| Scenario | Status | Color | Badge |
|----------|--------|-------|-------|
| No daemon (standalone dev) | `standalone` | blue | Standalone (dev) |
| Daemon running, all green | `healthy` | green | Healthy |
| Supabase configured but write off | `degraded` | yellow | Degraded |
| Supabase not configured | `healthy` | green | Healthy |
| Ollama not reachable | `healthy` | green | Healthy (reported in subsystems) |
| Scheduler enabled but stopped | `degraded` | red | Degraded |

### Files Changed
- `src/admin/health.py`
- `admin-ui/src/components/shared/StatusDot.tsx`
- `admin-ui/src/api/types.ts`
- `admin-ui/src/components/health/HealthBadge.tsx`

---

## Bug 6 — No Startup Dependency Verification

### Problem

After the lazy-import fixes in the prior bug fix session, missing packages were only
discovered at runtime during analysis — producing a raw traceback deep in the stack.

### Fix

**New file: `src/admin/startup_checks.py`**

```python
REQUIRED_PACKAGES = [
    ("langchain_anthropic",   "langchain-anthropic",   "Anthropic LLM provider"),
    ("langchain_openai",      "langchain-openai",      "OpenAI-compatible LLM provider"),
    ("langchain_google_genai","langchain-google-genai","Google Gemini LLM provider"),
    ("stockstats",            "stockstats",            "Technical indicator calculations"),
    ("rank_bm25",             "rank-bm25",             "Financial memory retrieval"),
    ("apscheduler",           "apscheduler",           "Scheduled watchlist scans"),
    ("supabase",              "supabase",              "Supabase signal publishing"),
    ("yfinance",              "yfinance",              "Yahoo Finance data fetching"),
]

def check_dependencies() -> list[dict]: ...   # returns {package, pip_name, installed}
def get_missing() -> list[str]: ...           # returns pip names of missing packages
def log_missing_warnings() -> None: ...       # logs WARNINGs for missing packages
```

**`src/admin/app.py` — `on_startup` handler:**

```python
from src.admin.startup_checks import log_missing_warnings
log_missing_warnings()
```

Logs a single `WARNING` at startup listing every missing package and the exact
`pip install` command to fix them.

**`src/admin/health.py` — new `/health/dependencies` endpoint:**

```python
@health_router.get("/health/dependencies")
async def dependency_check():
    results = check_dependencies()
    return {
        "status":   "ok" | "missing_packages",
        "all_ok":   bool,
        "missing":  [pip_name, ...],
        "packages": [{package, pip_name, required_for, installed}, ...],
    }
```

### Files Changed / Created
- `src/admin/startup_checks.py` (**new**)
- `src/admin/app.py`
- `src/admin/health.py`

---

## Test Updates

`tests/test_admin_health.py` was updated to match the new `compute_health_color()`
signature and the new standalone-mode behavior:

| Test | Change |
|------|--------|
| `test_yellow_when_ollama_unreachable` | Renamed → `test_ollama_unreachable_does_not_degrade`; now asserts GREEN (correct) |
| `test_yellow_when_supabase_write_disabled` | Renamed → `test_yellow_when_supabase_configured_but_write_disabled`; passes `supabase_configured=True` |
| `test_green_when_supabase_not_configured` | **New test** — asserts green when `supabase_configured=False` |
| `test_health_color_red_when_no_daemon` | Renamed → `test_health_standalone_mode_when_no_daemon`; asserts `color=="blue"`, `status=="standalone"`, `mode=="standalone"` |
| All `_color()` defaults | `ollama_reachable` removed, `supabase_configured=True` added |

All 15 health tests pass.

---

## Build Verification

```
tsc --noEmit:  ✓  0 errors
npm run build: ✓  built in 2.05s, 0 errors
pytest (health + task018 + task019 + admin tests): ✓  all pass, exit_code=0
```

---

## Exit Criteria Status

| # | Bug | Status |
|---|-----|--------|
| 1 | `publish` keyword error | ✅ Fixed — `run_analysis()` called without `publish`; `_publish_signal` invoked post-analysis |
| 2 | Analyses detail panel | ✅ Fixed — `useEffect([id])` re-fetches on row change |
| 3 | Scheduler hour/minute | ✅ Fixed — reads/writes `watchlist_hour`/`watchlist_minute`; type updated |
| 4 | Queue score column | ✅ Fixed — column key updated to `opportunity_score` |
| 5 | Degraded status | ✅ Fixed — standalone shows blue "Standalone"; Supabase/Ollama no longer degrade when unconfigured |
| 6 | Dependency check | ✅ Done — `startup_checks.py` created; warnings at startup; `/health/dependencies` endpoint live |

---

## Files Modified — Complete List

### New files
| File | Purpose |
|------|---------|
| `src/admin/startup_checks.py` | Dependency verification module |

### Backend
| File | Bug(s) | Change |
|------|--------|--------|
| `src/admin/test_run.py` | 1 | Removed `publish=` from `run_analysis()` call; added `_publish_signal()` post-call |
| `src/admin/health.py` | 5, 6 | Standalone early-return; fixed Supabase/Ollama color logic; added `/health/dependencies` |
| `src/admin/app.py` | 6 | Call `log_missing_warnings()` at startup |

### Frontend
| File | Bug(s) | Change |
|------|--------|--------|
| `admin-ui/src/components/analyses/AnalysesPage.tsx` | 2 | `useState` → `useEffect([id])`; added `useEffect` import |
| `admin-ui/src/components/config/ConfigPage.tsx` | 3 | `s.hour`/`s.minute` → `s.watchlist_hour`/`s.watchlist_minute` |
| `admin-ui/src/components/config/config-help.ts` | 3 | Renamed tooltip keys to match |
| `admin-ui/src/api/types.ts` | 3, 5 | `AutomationConfig` field names; `HealthResponse` gains `mode` + `'blue'` color |
| `admin-ui/src/components/queue/QueuePage.tsx` | 4 | Column key `'score'` → `'opportunity_score'` |
| `admin-ui/src/components/shared/StatusDot.tsx` | 5 | Added `'blue'` color support |
| `admin-ui/src/components/health/HealthBadge.tsx` | 5 | Standalone badge label + blue color handling |

### Tests
| File | Change |
|------|--------|
| `tests/test_admin_health.py` | Updated 4 tests for new signature/behavior; added 1 new test |
