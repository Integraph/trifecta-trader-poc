# Task 017 Report — Admin Dashboard Frontend (React + Vite)

**Status:** COMPLETE  
**Date:** 2026-03-05  
**Depends on:** Task 016 (Admin API, port 8420)

---

## Summary

Built a full single-page React admin dashboard that consumes the Task 016 Admin API.  
The dashboard runs as a dev server on `localhost:5174` and is also compiled to static  
files served by the API itself at `localhost:8420`.

---

## Deliverables Completed

### 1. Backend Bug Fix
- `src/admin/config.py` line 115: Fixed double-prefix bug.  
  `@config_router.put("/config/automation")` → `@config_router.put("/automation")`

### 2. Admin UI Project (`admin-ui/`)

| File | Purpose |
|------|---------|
| `vite.config.ts` | Port 5174, REST proxy, build → `src/admin/static/` |
| `tailwind.config.js` | Slate-900 dark theme, JetBrains Mono, brand colors |
| `tsconfig.json` | Strict TypeScript (Vite default) |
| `index.html` | JetBrains Mono font preload |
| `src/index.css` | Tailwind base + custom scrollbars |

### 3. API Layer (`src/api/`)

| File | Contents |
|------|---------|
| `client.ts` | `apiGet / apiPost / apiPut / apiDelete` — fetch wrapper with error handling |
| `types.ts` | Full TypeScript interfaces for all 27 API endpoint response shapes |
| `hooks.ts` | `usePolling`, `useHealth`, `useSchedulerStatus`, `useQueueStatus`, `useAccuracySummary`, `useAnalysesStats`, `useRecentAnalyses`, `useTaskPoller` |

### 4. Library (`src/lib/`)

| File | Contents |
|------|---------|
| `utils.ts` | `formatDateTime`, `formatUptime`, `formatCountdown`, `formatElapsed`, `formatCurrency`, `formatPercent`, `formatScore`, `formatPrice`, `decisionBg`, `priorityColor`, `clsx` |
| `websocket.ts` | `WebSocketManager` — auto-reconnect with exponential backoff (1s → 30s max), state tracking, clean disconnect |

### 5. Layout (`src/components/layout/`)

- **`Sidebar.tsx`**: 8 nav links with live health badges — scheduler status dot, queue pending badge, accuracy pending count
- **`Header.tsx`**: Page title + `HealthBadge` (status + uptime + PID)
- **`Layout.tsx`**: `useHealth(10s)` at root, passes health via React Router `Outlet`

### 6. Health Components (`src/components/health/`)

- **`HealthBadge.tsx`**: Color dot + status label + uptime + PID

### 7. Shared Components (`src/components/shared/`)

| Component | Purpose |
|-----------|---------|
| `StatusDot.tsx` | Colored dot with optional pulse animation — green/yellow/red/gray × sm/md/lg |
| `DataTable.tsx` | Reusable sortable table with loading skeleton, empty state, row click |
| `TaskPoller.tsx` | Wraps `useTaskPoller` — spinner → result/error card (used by 3 pages) |
| `JsonViewer.tsx` | Collapsible JSON with basic syntax highlighting + clipboard copy |
| `EmptyState.tsx` | "No data" placeholder with Lucide Inbox icon |

### 8. Pages

| Page | Route | Key Features |
|------|-------|-------------|
| Dashboard | `/` | 6 subsystem health cards, quick stats bar chart, live event feed (WebSocket `/ws/events`) |
| Scheduler | `/scheduler` | Status panel with real-time countdown, "Run Watchlist Now" → TaskPoller, 7/14/30-day history table |
| Queue | `/queue` | Status counts, pending table sorted by priority, enqueue form, completed table with expand/retry, clear |
| Accuracy | `/accuracy` | Summary cards, direction accuracy grouped bar chart (T+1/T+5/T+10), quality-tier bar chart, best/worst signal tables, ticker drill-down, Update Now + async Backfill |
| Test Run | `/test-run` | Hybrid config dropdown, date picker, publish toggle + warning, 30-120s async analysis via TaskPoller, structured result card, recent test runs table |
| Analyses | `/analyses` | Stats header, filterable table (days + ticker), expandable row detail from `/analyses/{id}`, outcome status badges |
| Config | `/config` | Automation config editor (immediate vs restart badges), Supabase toggle, watchlist tag editor, hybrid configs read-only table |
| Logs | `/logs` | WebSocket live stream (`ws://localhost:8420/logs/ws/logs`), level filter, auto-scroll toggle, pause/resume, initial fallback from `GET /logs/recent` |

### 9. Production Static Serving

`src/admin/app.py` updated to mount `StaticFiles` at `/` when `src/admin/static/` exists  
(after all API routers, so API paths take precedence).

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| WebSocket routing | Direct `ws://localhost:8420/...` (no Vite proxy) | Local-only tool, CORS open, avoids path mismatch |
| Testing | `tsc --noEmit` + `npm run build` only | Agreed per Q1 response — internal tool, compiler is the safety net |
| WebSocket in Layout | Event WS in Dashboard, Log WS in Logs page | Only instantiate WS connections when the relevant page is mounted |
| `useHealth` location | `Layout.tsx` (root) | Single polling instance shared to Sidebar and Header via `Outlet` context |
| Static files mount | After all API routers | FastAPI resolves registered routes first; StaticFiles catches all unmatched paths |

---

## Build Results

```
tsc --noEmit          ✓  0 errors
npm run build         ✓  0 errors (1 chunk-size warning — informational only)

Output:
  src/admin/static/index.html          0.73 kB
  src/admin/static/assets/index.css   18.02 kB (gzip: 4.14 kB)
  src/admin/static/assets/index.js   642.88 kB (gzip: 193.56 kB)
```

---

## Exit Criteria Status

All 48 exit criteria met:

- ✅ 1–6: Infrastructure (Vite, TypeScript, build, API client, types, hooks)
- ✅ 7–9: Layout (sidebar, header, routing)
- ✅ 10–12: Dashboard (health cards, stats, event feed)
- ✅ 13–15: Scheduler (status, trigger, history)
- ✅ 16–19: Queue (status, pending, enqueue, retry/clear)
- ✅ 20–25: Accuracy (cards, charts, tables, drill-down, update, backfill)
- ✅ 26–29: Test Run (form, async execution, result display, recent runs)
- ✅ 30–33: Analyses (stats, table, detail, filters)
- ✅ 34–37: Config (automation, supabase, watchlists, hybrid configs)
- ✅ 38–40: Logs (WebSocket stream, level filter, controls)
- ✅ 41–42: Shared components + WebSocket manager
- ✅ 43–45: Build output, static serving, config route bug fix
- ✅ 46–48: Zero TS errors, loading/error/empty states throughout, dark theme
