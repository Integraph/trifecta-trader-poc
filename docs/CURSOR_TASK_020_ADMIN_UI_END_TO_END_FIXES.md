# Cursor Task 020 — Admin UI End-to-End Bug Fixes

**Priority:** HIGH
**Depends on:** Tasks 016, 017, 018, 019, Bug Fix Report (docs/BUG_FIX_REPORT_SCHEDULER_LOGS_TESTRUN.md)
**Scope:** Backend API + Frontend Admin UI
**Date:** 2026-03-17

---

## Context

The previous bug fix session (documented in `docs/BUG_FIX_REPORT_SCHEDULER_LOGS_TESTRUN.md`)
resolved five issues: Scheduler blank page, Logs blank page, and three eager-import
`ModuleNotFoundError` crashes (langchain_google_genai, stockstats, rank_bm25).

During continued user acceptance testing, **six additional bugs** were discovered.
Three are blocking errors in the Test Run page. The remaining three are data display
bugs that cause fields to show incorrect or missing values across other pages.

The third screenshot from UAT clearly shows:

```
Analysis: TSLA failed
run_analysis() got an unexpected keyword argument 'publish'
```

This error occurs on every Single Run and A/B Comparison attempt, making the
Test Run page completely non-functional even after the previous import fixes.

---

## Bug 1 — BLOCKER: `run_analysis()` Does Not Accept `publish` Keyword

**Severity:** BLOCKER
**Affected pages:** Test Run → Single Run, Test Run → A/B Compare
**Symptom:** Every analysis fails with `run_analysis() got an unexpected keyword argument 'publish'`

### Root Cause

`src/admin/test_run.py` line 214 passes `publish=publish` to `run_analysis()`,
but `run_analysis()` (defined in `src/run_analysis.py` lines 222-227) does not
accept a `publish` parameter:

```python
# src/run_analysis.py lines 222-227 — function signature
def run_analysis(ticker: str, trade_date: str, provider: str = "anthropic",
                 deep_model: str = None, quick_model: str = None,
                 debug: bool = True, hybrid: str = None,
                 use_cache: bool = True, cost_breakdown: bool = True,
                 portfolio_context: Optional[dict] = None,
                 batch_mode: bool = False) -> dict:
```

Note: there is NO `publish` parameter. Publishing is handled separately in the
CLI's `__main__` block (lines 773-774) via the module-level function
`_publish_signal()` (defined at line 442).

The problematic call in test_run.py:

```python
# src/admin/test_run.py lines 208-216
result = run_analysis(
    ticker=ticker,
    trade_date=trade_date,
    hybrid=hybrid_config,
    use_cache=True,
    cost_breakdown=True,
    publish=publish,        # ← THIS PARAMETER DOES NOT EXIST
    debug=False,
)
```

Both the Single Run endpoint (line 69-74, calls `_run_analysis_safe`) and
the A/B Compare endpoint (lines 112-116, same function) are affected.

### Required Fix

Modify `_run_analysis_safe()` in `src/admin/test_run.py` (lines 197-237):

1. **Remove** `publish=publish` from the `run_analysis()` call on line 214.
2. **Add** a post-analysis call to `_publish_signal()` when `publish=True`.
3. **Change** `"published": publish` in the return dict to `"published": published`
   so it reflects whether publish actually succeeded, not just whether it was requested.

Here is the corrected function (replace lines 197-237):

```python
def _run_analysis_safe(
    ticker: str,
    hybrid_config: str,
    publish: bool,
    trade_date: str,
) -> dict:
    """Run a single analysis and return a sanitised result dict."""
    from src.run_analysis import run_analysis, _publish_signal
    import time

    start = time.time()
    result = run_analysis(
        ticker=ticker,
        trade_date=trade_date,
        hybrid=hybrid_config,
        use_cache=True,
        cost_breakdown=True,
        debug=False,
    )
    elapsed = round(time.time() - start, 1)

    # Publish to Supabase if requested (mirrors CLI --publish behaviour)
    published = False
    if publish:
        try:
            trade_params = result.get("trade_params_obj")  # raw TradeParams if present
            _publish_signal(result, trade_params)
            published = True
        except Exception as e:
            logger.warning("Publish failed for %s: %s", ticker, e)

    qs = result.get("quality_score", {}) or {}
    tp = result.get("trade_params", {}) or {}
    cb = result.get("cost_breakdown", {}) or {}

    return {
        "ticker":          ticker,
        "trade_date":      trade_date,
        "hybrid_config":   hybrid_config,
        "decision":        result.get("decision"),
        "quality_score":   qs,
        "trade_params":    tp,
        "cost_breakdown":  {
            "total_usd":   cb.get("total_usd"),
            "by_provider": cb.get("by_provider", {}),
        },
        "elapsed_seconds": elapsed,
        "published":       published,
        "result_file":     result.get("result_file"),
    }
```

Key changes:
- `publish=publish` removed from `run_analysis()` call
- `_publish_signal` imported alongside `run_analysis`
- Post-analysis publish call guarded by `if publish`, wrapped in try/except
- Return dict uses `published` (actual outcome) instead of `publish` (requested)

### Verification

1. Start the admin API: `cd trifecta-trader-poc && python -m src.run_daemon --api`
2. Open `http://localhost:5174/test-run`
3. Enter ticker `AAPL`, select `hybrid_haiku_tools`, leave Publish unchecked
4. Click **Run Analysis**
5. **Expected:** Analysis completes with a result card showing decision, quality score, cost, and elapsed time. No `unexpected keyword argument` error.
6. Repeat with Publish checked — verify the result shows `Published: Yes` if Supabase is configured, or a warning in the API console if not.

---

## Bug 2 — Analyses Detail Panel Uses `useState` Instead of `useEffect`

**Severity:** HIGH
**Affected page:** Analyses → row detail expansion
**Symptom:** Clicking an analysis row either fails to load detail data or shows
"Failed to load" permanently. The detail never re-fetches when clicking a different row.

### Root Cause

`admin-ui/src/components/analyses/AnalysesPage.tsx` lines 50-56 use `useState()`
with a callback to fire an API request:

```tsx
// Current code — BROKEN (lines 50-56)
useState(() => {
    setLoading(true);
    apiGet<AnalysisDetail>(`/analyses/${id}`)
      .then(r => setDetail(r))
      .catch(() => {})
      .finally(() => setLoading(false));
  });
```

`useState`'s initializer runs once during initial render and does NOT re-run when
`id` changes. It also runs synchronously during the render phase, making the async
call inside it unreliable. The correct hook is `useEffect`.

### Required Fix

1. Replace lines 50-56 with:

```tsx
useEffect(() => {
    setLoading(true);
    apiGet<AnalysisDetail>(`/analyses/${id}`)
      .then(r => setDetail(r))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [id]);
```

2. Add `useEffect` to the React import at the top of the file. Currently line 1:

```tsx
import { useCallback, useMemo, useState } from 'react';
```

Change to:

```tsx
import { useCallback, useEffect, useMemo, useState } from 'react';
```

### Verification

1. Navigate to **Analyses** in the sidebar
2. Click any row in the analyses table
3. **Expected:** Detail panel expands and shows full analysis data (decision, quality
   breakdown, trade params, cost, etc.)
4. Click a different row
5. **Expected:** Detail panel updates with the new analysis data (not stale)

---

## Bug 3 — Configuration Page: Scheduler Hour/Minute Field Name Mismatch

**Severity:** MEDIUM
**Affected page:** Configuration → Automation Config → Scheduler section
**Symptom:** The Hour and Minute fields display default values (8, 30) regardless
of what's saved in the YAML. Edits appear to save but have no effect on the
actual scheduler because they're written to the wrong YAML keys.

### Root Cause

The YAML config file `config/automation.yaml` uses `watchlist_hour` and
`watchlist_minute` as the field names (lines 3-4):

```yaml
scheduler:
  enabled: true
  watchlist_hour: 8
  watchlist_minute: 30
```

The backend `src/admin/config.py` reads these exact names (line 165-170):

```python
def _fmt_schedule(cfg: dict) -> str:
    hour      = cfg.get("watchlist_hour", 8)
    minute    = cfg.get("watchlist_minute", 30)
```

But the frontend `ConfigPage.tsx` reads/writes `hour` and `minute` (lines 119-120):

```tsx
<FieldRow label="Hour (24h)" value={s.hour ?? 8}   onChange={v => set(['scheduler','hour'], v)} type="number" ... />
<FieldRow label="Minute"     value={s.minute ?? 30} onChange={v => set(['scheduler','minute'], v)} type="number" ... />
```

When the admin saves, the backend receives `{"scheduler": {"hour": 9}}` and
writes `scheduler.hour: 9` to the YAML. The scheduler daemon ignores this field
and keeps reading `scheduler.watchlist_hour`.

### Required Fix

Update `admin-ui/src/components/config/ConfigPage.tsx` lines 119-120:

**Before:**
```tsx
<FieldRow label="Hour (24h)" value={s.hour ?? 8}   onChange={v => set(['scheduler','hour'], v)} type="number"  info={SETTINGS_HELP['scheduler.hour']} />
<FieldRow label="Minute"     value={s.minute ?? 30} onChange={v => set(['scheduler','minute'], v)} type="number"  info={SETTINGS_HELP['scheduler.minute']} />
```

**After:**
```tsx
<FieldRow label="Hour (24h)" value={s.watchlist_hour ?? 8}   onChange={v => set(['scheduler','watchlist_hour'], v)} type="number"  info={SETTINGS_HELP['scheduler.watchlist_hour']} />
<FieldRow label="Minute"     value={s.watchlist_minute ?? 30} onChange={v => set(['scheduler','watchlist_minute'], v)} type="number"  info={SETTINGS_HELP['scheduler.watchlist_minute']} />
```

Also update the SETTINGS_HELP keys in `admin-ui/src/components/config/config-help.ts`.
If entries exist for `scheduler.hour` / `scheduler.minute`, rename them to
`scheduler.watchlist_hour` / `scheduler.watchlist_minute`. The tooltip text
can stay the same — just the key must match.

### Verification

1. Navigate to **Configuration** in the sidebar
2. In the Scheduler section, confirm Hour shows `8` and Minute shows `30`
   (matching `config/automation.yaml`)
3. Change Hour to `9`, click **Save**
4. Reload the page — Hour should still show `9`
5. Open `config/automation.yaml` — confirm it now says `watchlist_hour: 9`
6. Change it back to `8` and save

---

## Bug 4 — Queue Page: Score Column Always Shows "—"

**Severity:** MEDIUM
**Affected page:** Queue → Pending Candidates table
**Symptom:** The Score column displays "—" for every candidate, even when the
backend data contains valid `opportunity_score` values.

### Root Cause

The backend `src/admin/queue.py` line 122 returns the field as `opportunity_score`:

```python
candidates.append({
    "filename":          path.name,
    "ticker":            msg.get("ticker"),
    "priority":          msg.get("priority", "medium"),
    "opportunity_score": msg.get("opportunity_score"),   # ← field name
    ...
})
```

But the frontend `QueuePage.tsx` line 120 defines the column key as `score`:

```tsx
{ key: 'score', label: 'Score', sortable: true,
  render: r => <>{r.score != null ? (r.score as number).toFixed(2) : '—'}</> },
```

Since `r.score` is always `undefined` (the field is `r.opportunity_score`), the
render function always falls through to `'—'`.

### Required Fix

Update `admin-ui/src/components/queue/QueuePage.tsx` line 120:

**Before:**
```tsx
{ key: 'score', label: 'Score', sortable: true,
  render: r => <>{r.score != null ? (r.score as number).toFixed(2) : '—'}</> },
```

**After:**
```tsx
{ key: 'opportunity_score', label: 'Score', sortable: true,
  render: r => <>{r.opportunity_score != null ? (r.opportunity_score as number).toFixed(2) : '—'}</> },
```

### Verification

1. Enqueue a candidate that has an opportunity_score (or manually add a test message
   to the `queue/` directory with `"opportunity_score": 0.85`)
2. Navigate to **Queue** in the sidebar
3. **Expected:** Score column shows `0.85` instead of `—`

---

## Bug 5 — "Degraded" Status Badge When API Is Healthy

**Severity:** LOW
**Affected page:** Dashboard / global header (top-right status badge)
**Symptom:** The screenshots show "Degraded" with an orange dot even though
the API is running and responding to all requests correctly.

### Root Cause (investigate)

The health endpoint likely checks subsystem status (scheduler running, Supabase
connected, queue reader active) and returns "degraded" when any subsystem is
not fully operational. When the API is started in standalone mode
(`python -m src.admin.app`), the scheduler and queue reader are not running,
which triggers the degraded status even though that's expected behavior.

### Required Fix

Check `src/admin/health.py` and the frontend `DashboardPage.tsx` (or wherever
the status badge is rendered). The fix should:

1. Add a `mode` field to the health response: `"standalone"` when started via
   `src.admin.app` directly, `"full"` when started via `src.run_daemon --api`.
2. The frontend should display "Standalone" (blue/neutral badge) when mode is
   standalone, "Healthy" (green) when all subsystems are up, and "Degraded"
   (orange) only when a subsystem that should be running is failing.

If the daemon IS running and status still shows "Degraded", investigate which
subsystem check is failing and fix the condition.

### Verification

1. Start the API in standalone mode: `python -m src.admin.app`
2. Open `http://localhost:5174`
3. **Expected:** Status badge does NOT show "Degraded". It shows "Standalone"
   or similar neutral indicator.
4. Start the API in full daemon mode: `python -m src.run_daemon --api`
5. **Expected:** Status badge shows "Healthy" (green) if all subsystems start.

---

## Bug 6 — Startup Dependency Check (Enhancement)

**Severity:** MEDIUM (prevents confusing errors)
**Affected:** All pages that trigger analysis

### Problem

When required Python packages are not installed, the admin API starts successfully
but every test run fails with a raw `ModuleNotFoundError` traceback. The previous
bug fix session made imports lazy, which prevents startup crashes but means the
error is only discovered at runtime during analysis.

### Required Enhancement

Create `src/admin/startup_checks.py`:

```python
"""
Startup dependency verification.

Called during app creation to log warnings about missing packages.
The API still starts — but the admin gets an immediate heads-up.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = [
    ("langchain_google_genai", "langchain-google-genai"),
    ("stockstats",             "stockstats"),
    ("rank_bm25",              "rank-bm25"),
    ("langchain_anthropic",    "langchain-anthropic"),
    ("langchain_openai",       "langchain-openai"),
    ("apscheduler",            "apscheduler"),
    ("supabase",               "supabase"),
]


def check_dependencies() -> list[dict]:
    """Return a list of {package, pip_name, installed: bool}."""
    results = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            results.append({"package": module_name, "pip_name": pip_name, "installed": True})
        except ImportError:
            results.append({"package": module_name, "pip_name": pip_name, "installed": False})
            logger.warning("Missing optional dependency: pip install %s", pip_name)
    return results


def get_missing() -> list[str]:
    """Return pip install names for all missing packages."""
    return [r["pip_name"] for r in check_dependencies() if not r["installed"]]
```

Integrate into `src/admin/app.py` inside `create_app()`:

```python
from src.admin.startup_checks import get_missing

missing = get_missing()
if missing:
    logger.warning(
        "Missing packages — some features may not work: %s\n"
        "Fix with: pip install %s",
        ", ".join(missing),
        " ".join(missing),
    )
```

Add a health sub-endpoint to `src/admin/health.py`:

```python
@health_router.get("/health/dependencies")
async def dependency_check():
    from src.admin.startup_checks import check_dependencies
    results = check_dependencies()
    all_ok = all(r["installed"] for r in results)
    return {"status": "ok" if all_ok else "missing_packages", "packages": results}
```

### Verification

1. Uninstall one optional package (e.g., `pip uninstall stockstats`)
2. Start the API
3. **Expected:** Console shows `WARNING: Missing optional dependency: pip install stockstats`
4. Visit `http://localhost:8420/health/dependencies`
5. **Expected:** JSON response shows `stockstats` with `"installed": false`
6. Reinstall: `pip install stockstats`

---

## Files to Modify — Summary

### New files

| File | Purpose |
|------|---------|
| `src/admin/startup_checks.py` | Dependency verification module |

### Backend modifications

| File | Bug(s) | Change |
|------|--------|--------|
| `src/admin/test_run.py` | 1 | Remove `publish=` from `run_analysis()` call; add post-analysis `_publish_signal()` |
| `src/admin/health.py` | 5, 6 | Add `/health/dependencies` endpoint; improve degraded vs standalone status |
| `src/admin/app.py` | 6 | Call `get_missing()` at startup, log warnings |

### Frontend modifications

| File | Bug(s) | Change |
|------|--------|--------|
| `admin-ui/src/components/analyses/AnalysesPage.tsx` | 2 | `useState` → `useEffect`; add `useEffect` import |
| `admin-ui/src/components/config/ConfigPage.tsx` | 3 | `s.hour`/`s.minute` → `s.watchlist_hour`/`s.watchlist_minute` |
| `admin-ui/src/components/config/config-help.ts` | 3 | Update SETTINGS_HELP keys to match |
| `admin-ui/src/components/queue/QueuePage.tsx` | 4 | Column key `'score'` → `'opportunity_score'` |
| `admin-ui/src/components/dashboard/DashboardPage.tsx` | 5 | Update status badge to handle standalone mode |

---

## Exit Criteria

| # | Bug | Test |
|---|-----|------|
| 1 | `publish` keyword error | Single Run completes for AAPL with `hybrid_haiku_tools`. Result card shows decision + quality + cost. No `unexpected keyword argument` error. A/B Compare also works. |
| 2 | Analyses detail panel | Click an analysis row → detail panel loads data. Click a different row → detail updates. |
| 3 | Scheduler hour/minute | Config page shows correct values from YAML. Edit + Save + Reload persists. `automation.yaml` shows `watchlist_hour` (not `hour`). |
| 4 | Queue score column | Pending candidates with `opportunity_score` show numeric values, not "—". |
| 5 | Degraded status | Standalone mode shows neutral badge. Full daemon mode shows green "Healthy". |
| 6 | Dependency check | Missing packages logged at startup. `/health/dependencies` returns package status. |

### Build gates

- `tsc --noEmit` passes with 0 errors
- `npm run build` passes with 0 errors
- All existing Python tests pass (except the 2 known Ollama model format failures)

---

## Task Report

When complete, create `docs/TASK_020_REPORT.md` with:
1. Summary of all changes made (before/after for each bug)
2. Any additional issues discovered during the fix
3. Test results confirming each exit criterion passes
4. Build verification output (tsc + npm run build + pytest summary)
