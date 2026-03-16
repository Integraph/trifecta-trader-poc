# Task 017: Admin Dashboard Frontend (React + Vite)

**Priority:** HIGH — Pipeline is autonomous; operators need visual controls
**Depends on:** Task 016 (Admin API on port 8420)
**Enables:** Task 018 (Platform UI health badge)

---

## Objective

Build a single-page React admin dashboard that consumes the Task 016 Admin API (`localhost:8420`). This is the operator's cockpit — it provides full visibility and control over the Trifecta Trader pipeline without touching the CLI.

The dashboard runs locally on `localhost:5174` (Vite dev server) or is served as static files by the Admin API itself in production mode.

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| React | 18.x | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool + dev server |
| Tailwind CSS | 3.x | Utility-first styling |
| Recharts | 2.x | Charts (accuracy, history) |
| Lucide React | latest | Icons |
| React Router | 6.x | Client-side routing |

No component library (no shadcn, no MUI). Keep it lightweight — raw Tailwind with clean, functional components. The dashboard is admin-only, local-only, so aesthetics are secondary to clarity and information density.

---

## Architecture

```
admin-ui/
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
├── src/
│   ├── main.tsx                   # Entry point
│   ├── App.tsx                    # Router + layout
│   ├── api/
│   │   ├── client.ts             # Axios/fetch wrapper (base URL: localhost:8420)
│   │   ├── types.ts              # TypeScript interfaces for all API responses
│   │   └── hooks.ts              # Custom React hooks for data fetching
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx        # Nav sidebar
│   │   │   ├── Header.tsx         # Top bar with health badge
│   │   │   └── Layout.tsx         # Main layout wrapper
│   │   ├── health/
│   │   │   ├── HealthBadge.tsx    # Green/yellow/red dot with tooltip
│   │   │   └── HealthDetail.tsx   # Full subsystem breakdown
│   │   ├── dashboard/
│   │   │   └── DashboardPage.tsx  # Overview page
│   │   ├── scheduler/
│   │   │   └── SchedulerPage.tsx  # Scheduler controls + history
│   │   ├── queue/
│   │   │   └── QueuePage.tsx      # Queue viewer + enqueue + manage
│   │   ├── accuracy/
│   │   │   └── AccuracyPage.tsx   # Accuracy reports + charts
│   │   ├── test-run/
│   │   │   └── TestRunPage.tsx    # Ticker input + results viewer
│   │   ├── config/
│   │   │   └── ConfigPage.tsx     # Config editor
│   │   ├── analyses/
│   │   │   └── AnalysesPage.tsx   # Analysis history + detail
│   │   ├── logs/
│   │   │   └── LogsPage.tsx       # Live log viewer
│   │   └── shared/
│   │       ├── StatusDot.tsx      # Reusable colored status indicator
│   │       ├── DataTable.tsx      # Reusable table component
│   │       ├── TaskPoller.tsx     # Reusable async task polling UI
│   │       ├── JsonViewer.tsx     # Collapsible JSON display
│   │       └── EmptyState.tsx     # "No data yet" placeholder
│   └── lib/
│       ├── websocket.ts          # WebSocket connection manager
│       └── utils.ts              # Formatters, date helpers
```

The `admin-ui/` directory lives at the project root, alongside `src/` and `config/`.

---

## Deliverable 1: API Client Layer (`src/api/`)

### `client.ts`

Simple fetch wrapper with the Admin API base URL.

```typescript
const API_BASE = "http://localhost:8420";

export async function apiGet<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(path, API_BASE);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const resp = await fetch(url.toString());
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> { ... }
export async function apiPut<T>(path: string, body: unknown): Promise<T> { ... }
export async function apiDelete<T>(path: string, params?: Record<string, string>): Promise<T> { ... }
```

### `types.ts`

TypeScript interfaces matching every API response shape. Key types:

```typescript
interface HealthResponse {
  status: string;
  color: "green" | "yellow" | "red";
  timestamp: string;
  uptime_seconds: number | null;
  subsystems: {
    daemon: { status: string; pid: number; start_time: string | null };
    scheduler: { status: string; last_run: string | null; last_run_result: string | null; next_run: string | null; ... };
    queue_reader: { status: string; pending_count: number; completed_today: number; last_poll: string | null };
    accuracy_updater: { status: string; pending_outcomes: number; complete_outcomes: number; ... };
    supabase: { configured: boolean; write_enabled: boolean };
    ollama: { reachable: boolean; model: string | null };
  };
}

interface SchedulerStatus { enabled: boolean; is_running: boolean; schedule: string; next_run: string | null; last_run: object | null; config: object; }
interface QueueStatus { enabled: boolean; is_running: boolean; counts: { pending: number; processing: number; completed: number }; config: object; }
interface AccuracySummary { period_days: number; total_signals: number; pending_outcomes: number; by_decision: Record<string, DecisionStats>; by_quality_tier: Record<string, TierStats>; best_signals: Signal[]; worst_signals: Signal[]; }
interface AnalysisRow { id: number; ticker: string; trade_date: string; decision: string; quality_score: number; outcome_status: string | null; ... }
interface TaskStatus { task_id: string; status: "running" | "complete" | "error"; started_at: string; completed_at: string | null; result: unknown; error: string | null; }
// ... (full types for all 27 endpoints)
```

### `hooks.ts`

Custom React hooks for auto-refreshing data:

```typescript
// Polls every `interval` ms. Pauses when tab is hidden.
export function usePolling<T>(fetcher: () => Promise<T>, interval: number): { data: T | null; loading: boolean; error: string | null; refresh: () => void }

// Specific hooks:
export function useHealth(interval?: number): HealthResponse | null
export function useSchedulerStatus(): SchedulerStatus | null
export function useQueueStatus(): QueueStatus | null
export function useAccuracySummary(days?: number): AccuracySummary | null
export function useAnalyses(params: { days?: number; ticker?: string; limit?: number }): AnalysesResponse | null
```

---

## Deliverable 2: Layout & Navigation (`src/components/layout/`)

### Sidebar Navigation

Fixed left sidebar with navigation links:

```
📊 Dashboard          ← /
📅 Scheduler          ← /scheduler
📥 Queue              ← /queue
🎯 Accuracy           ← /accuracy
🧪 Test Run           ← /test-run
📋 Analyses           ← /analyses
⚙️ Configuration      ← /config
📜 Logs               ← /logs
```

Each nav item shows a small status indicator (colored dot) pulled from `/health`:
- Scheduler: green if `scheduler.status === "running"`, else red
- Queue: shows pending count badge if > 0
- Accuracy: shows pending outcomes count

### Header

Top bar spanning the content area:
- Left: Current page title
- Right: Health badge (HealthBadge component) + uptime display + daemon PID

### Layout

Standard sidebar + header + content area. Content area is the React Router outlet.

---

## Deliverable 3: Dashboard Page (`/`)

The overview page — one screen showing everything at a glance.

### System Health Panel (top row)

6 status cards in a grid, one per subsystem from `/health`:

| Card | Data Source | Display |
|------|------------|---------|
| Daemon | `subsystems.daemon` | Status dot + PID + uptime |
| Scheduler | `subsystems.scheduler` | Next run countdown + last run result |
| Queue Reader | `subsystems.queue_reader` | Pending/completed counts |
| Accuracy | `subsystems.accuracy_updater` | Pending/complete outcomes |
| Supabase | `subsystems.supabase` | Configured + write_enabled |
| Ollama | `subsystems.ollama` | Reachable + model name |

Each card has a colored border (green/yellow/red) based on its individual status.

### Quick Stats Panel (middle row)

From `GET /analyses/stats`:
- Total analyses (all time)
- Analyses today
- Decision breakdown (mini bar chart: BUY/SELL/HOLD)
- Average quality score
- Total cost USD

### Recent Activity Feed (bottom)

A reverse-chronological feed of events from the `/ws/events` WebSocket connection. Each event shows:
- Timestamp
- Event type (color-coded: scheduler=blue, queue=purple, accuracy=green, test-run=orange)
- Summary text (e.g., "Scheduler completed: 8 tickers, 142s")

Maximum 20 events shown. Streams in real-time.

**Auto-refresh:** Dashboard polls `/health` every 10 seconds and `/analyses/stats` every 30 seconds. Events stream via WebSocket.

---

## Deliverable 4: Scheduler Page (`/scheduler`)

### Status Panel

From `GET /scheduler/status`:
- Schedule display: "08:30 US/Eastern weekdays"
- Next run: countdown timer (auto-updates every second)
- Last run: timestamp, result (success/error badge), tickers processed, elapsed time, decisions breakdown
- Config: hybrid_config, watchlist, publish flag

### Manual Trigger

Button: **"Run Watchlist Now"**
- Clicking sends `POST /scheduler/trigger`
- Shows a spinner / progress state
- Uses the `TaskPoller` component to poll `GET /scheduler/trigger/{task_id}` every 2 seconds
- When complete, shows: tickers processed, elapsed time, decisions table

### Run History Table

From `GET /scheduler/history?days=7`:

| Date | Tickers | BUY | SELL | HOLD | Avg Quality | Elapsed | Cost |
|------|---------|-----|------|------|-------------|---------|------|
| 2026-03-05 | 8 | 3 | 1 | 4 | 7.2 | 142.5s | $0.23 |

Sortable columns. Day filter dropdown (7/14/30 days).

---

## Deliverable 5: Queue Page (`/queue`)

### Status Panel

From `GET /queue/status`:
- Running/stopped badge
- Pending / Processing / Completed counts (large numbers)
- Poll interval, cooldown, max retries from config

### Pending Candidates Table

From `GET /queue/pending`:

| Ticker | Priority | Score | Catalysts | Source | Retries | Queued At |
|--------|----------|-------|-----------|--------|---------|-----------|
| NVDA | high | 0.85 | volume_surge, breakout | scanner | 0 | 14:30 |

Priority column is color-coded (high=red, medium=yellow, low=gray).

### Enqueue Form

Input fields:
- Ticker (text input, auto-uppercase)
- Priority (dropdown: high/medium/low)
- Reason (text input, optional)

Submit button sends `POST /queue/enqueue`. Shows success toast with filename.

### Completed Table

From `GET /queue/completed`:
- Ticker, decision, quality score, elapsed, completed_at
- Expandable row to show full analysis_result JSON

### Actions

- **Retry** button on completed rows → `POST /queue/retry/{filename}`
- **Clear Completed** button → `DELETE /queue/clear?target=completed` (with confirmation)

---

## Deliverable 6: Accuracy Page (`/accuracy`)

### Summary Cards (top)

From `GET /accuracy/summary?days=30`:
- Total signals (complete)
- Pending outcomes (awaiting T+10)
- Direction accuracy at T+5 (overall %)
- Average return at T+5 (overall %)

### Direction Accuracy Chart

Recharts grouped bar chart:
- X-axis: BUY, SELL
- Y-axis: 0-100%
- Three bars per group: T+1, T+5, T+10

### Quality Tier Correlation Chart

Recharts bar chart:
- X-axis: "High (8-10)", "Medium (6-8)", "Low (0-6)"
- Y-axis: Average return T+5 (%)
- Color gradient: green for high, yellow for medium, red for low

This is the most important chart — it answers "does quality scoring predict accuracy?"

### Target/Stop Performance

Simple stats panel:
- Target hit rate by decision (BUY, SELL)
- Stop hit rate by decision
- Target-before-stop rate

### Best/Worst Signals Table

Two tables side-by-side:
- Best 5: Ticker, Decision, Return T+10%, Quality Score, Date
- Worst 5: Same columns

### Ticker Drill-Down

Input field: enter a ticker symbol → calls `GET /accuracy/ticker/{ticker}` → shows:
- Total signals for that ticker
- Complete vs. pending count
- Direction accuracy series
- Full signal history table

### Actions

- **Update Now** button → `POST /accuracy/update` (synchronous, shows result)
- **Backfill** button with days input → `POST /accuracy/backfill` (async, uses TaskPoller)
- Period selector: 7 / 14 / 30 / 90 days (re-fetches summary)

---

## Deliverable 7: Test Run Page (`/test-run`)

The admin's "does it still work?" panel.

### Input Form

- **Ticker** (text input, required, auto-uppercase)
- **Hybrid Config** (dropdown, populated from `GET /config/hybrid-configs`)
- **Trade Date** (date picker, defaults to today)
- **Publish to Supabase** (checkbox, default: OFF, with warning label: "⚠ Will write to production Supabase")

### Run Button

Large **"Run Analysis"** button.

On click:
1. Sends `POST /test-run` with form data
2. Button changes to spinner with "Analyzing {TICKER}..."
3. Uses TaskPoller to poll `GET /test-run/{task_id}` every 3 seconds
4. Typical wait: 30-120 seconds

### Result Display

When complete, show a structured result card:

**Decision Header:** Large "BUY" / "SELL" / "HOLD" badge (green/red/gray) with quality score ring

**Trade Parameters Table:**
| Field | Value |
|-------|-------|
| Entry Price | $178.50 |
| Stop Loss | $172.30 |
| Price Target | $195.00 |
| Position % | 3.5% |
| Risk/Reward | 2.66 |
| Confidence | high |

**Quality Score Breakdown:**
Visual bars for composite, research, analysis, recommendation scores (0-10 scale)

**Cost & Timing:**
- Total cost: $0.028
- Elapsed: 45.2s
- Published: No

**Raw Result:** Collapsible JSON viewer (JsonViewer component) for the full API response

### Recent Test Runs

Below the form, show a table of recent test runs from `GET /tasks` (filtered by task_id prefix "test_"):

| Ticker | Decision | Quality | Elapsed | Status | Time |
|--------|----------|---------|---------|--------|------|
| AAPL | BUY | 8.2 | 45.2s | ✅ | 14:35 |

---

## Deliverable 8: Analyses Page (`/analyses`)

### Stats Header

From `GET /analyses/stats`:
- Total analyses, analyses today, unique tickers, avg quality, total cost

### Filters

- Days range: 7 / 14 / 30 / 90 (dropdown)
- Ticker filter: text input (optional)

### Analyses Table

From `GET /analyses/recent?days=7&limit=50`:

| ID | Ticker | Date | Decision | Quality | Entry | Stop | Target | Cost | Elapsed | Outcome |
|----|--------|------|----------|---------|-------|------|--------|------|---------|---------|
| 42 | AAPL | 03-05 | BUY | 8.2 | $178.50 | $172.30 | $195.00 | $0.03 | 45.2s | pending |

- Decision column: colored badge (BUY=green, SELL=red, HOLD=gray)
- Outcome column: pending/partial/complete/error badge
- Clickable rows → expand to show full analysis detail from `GET /analyses/{id}`

### Detail Panel

When a row is clicked, show the full analysis record including signal outcome data (direction_correct, target_hit, returns, etc.) from the LEFT JOIN.

---

## Deliverable 9: Configuration Page (`/config`)

### Automation Config Panel

From `GET /config/automation`:
- Editable form with sections: Scheduler, Queue Reader, Accuracy, Admin API
- Each field shows its current value
- Save button → `PUT /config/automation` with changed fields
- Response shows "Applied" vs. "Requires Restart" badges per field

### Supabase Config Panel

From `GET /config/supabase`:
- write_enabled toggle
- signal_ttl_hours input
- table_name display
- Save button → `PUT /config/supabase`

### Watchlist Manager

From `GET /config/watchlists`:
- Table listing all watchlists with their tickers
- Click a watchlist to edit: shows editable ticker list (tag-style input)
- Save button → `PUT /config/watchlists/{name}`
- "New Watchlist" button to create a new one

### Hybrid Configs

From `GET /config/hybrid-configs`:
- Read-only table listing all available configs
- Shows active config (from scheduler config)
- Columns: Name, Tool Model, Reasoning Model, Deep Model

---

## Deliverable 10: Logs Page (`/logs`)

### Live Log Stream

WebSocket connection to `ws://localhost:8420/logs/ws/logs`.

Display: monospace font, dark background, newest entries at the bottom (auto-scroll).

Each line shows:
```
14:30:00  INFO   src.automation.scheduler  Watchlist scan starting: 8 tickers
14:30:02  WARN   src.integration.writer    Supabase write_enabled=false, skipping
```

Color-coded by level: DEBUG=gray, INFO=white, WARNING=yellow, ERROR=red.

### Controls

- Level filter: dropdown (DEBUG / INFO / WARNING / ERROR)
- Auto-scroll toggle (on by default)
- Pause/resume streaming
- Clear display button

### Recent Logs (fallback)

If WebSocket fails or on initial load, fetch `GET /logs/recent?lines=200&level=INFO` and display the buffer.

---

## Deliverable 11: Shared Components

### StatusDot

Small colored circle with optional pulse animation:
- `color: "green" | "yellow" | "red" | "gray"`
- `pulse: boolean` (for "running" states)
- `size: "sm" | "md" | "lg"`

### DataTable

Reusable table with:
- Column definitions (key, label, formatter, sortable)
- Optional row click handler
- Empty state message
- Loading skeleton

### TaskPoller

Manages the async task polling pattern (used by 3 pages):
- Takes a `taskId` and `pollUrl`
- Polls every N seconds
- Shows: spinner → result card or error card
- Reusable across test run, scheduler trigger, and backfill

### JsonViewer

Collapsible JSON display with:
- Syntax highlighting (basic: strings=green, numbers=blue, keys=gray)
- Expand/collapse toggle
- Copy-to-clipboard button

### EmptyState

"No data yet" placeholder with an icon and optional action button.

---

## Deliverable 12: WebSocket Manager (`src/lib/websocket.ts`)

Manages two WebSocket connections:

1. **Events** (`ws://localhost:8420/ws/events`) — used by Dashboard
2. **Logs** (`ws://localhost:8420/logs/ws/logs`) — used by Logs page

Features:
- Auto-reconnect with exponential backoff (1s → 2s → 4s → max 30s)
- Connection state tracking (connecting/connected/disconnected)
- Message buffering during reconnect
- Clean disconnect on component unmount

```typescript
class WebSocketManager {
  connect(url: string, onMessage: (data: any) => void): void
  disconnect(): void
  get state(): "connecting" | "connected" | "disconnected"
}
```

---

## Deliverable 13: Build & Serve Configuration

### `vite.config.ts`

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://localhost:8420',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8420',
        ws: true,
      },
    },
  },
  build: {
    outDir: '../src/admin/static',  // Build output goes into the API's static dir
  },
})
```

### Production Serving

The Vite build outputs to `src/admin/static/`. Add a static file mount to `src/admin/app.py`:

```python
# In create_app():
from fastapi.staticfiles import StaticFiles
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="admin-ui")
```

This means in production, `localhost:8420` serves both the API and the frontend.

### `package.json` scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }
}
```

---

## Design Guidelines

### Color Palette

- Background: `#0f172a` (slate-900) — dark theme
- Card background: `#1e293b` (slate-800)
- Text: `#e2e8f0` (slate-200)
- Accent green: `#22c55e` (green-500)
- Accent yellow: `#eab308` (yellow-500)
- Accent red: `#ef4444` (red-500)
- BUY badge: green-600
- SELL badge: red-600
- HOLD badge: slate-500

### Typography

- Body: system font stack (`-apple-system, BlinkMacSystemFont, ...`)
- Monospace (logs, JSON): `'JetBrains Mono', 'Fira Code', monospace`
- Sizes: headers 18-24px, body 14px, small 12px

### Layout

- Sidebar: 240px fixed width
- Content: max-width 1400px, centered
- Cards: rounded-lg, subtle border, shadow-sm
- Spacing: consistent 16px/24px padding

### Responsive

Not required — this is a desktop admin tool. Minimum viewport: 1280px.

---

## Auto-Refresh Intervals

| Page | Endpoint | Interval |
|------|----------|----------|
| Dashboard | `/health` | 10s |
| Dashboard | `/analyses/stats` | 30s |
| Dashboard | `/ws/events` | real-time |
| Scheduler | `/scheduler/status` | 15s |
| Queue | `/queue/status` | 10s |
| Queue | `/queue/pending` | 10s |
| Accuracy | `/accuracy/summary` | 60s |
| Analyses | `/analyses/recent` | 30s |
| Logs | `/logs/ws/logs` | real-time |

All polling pauses when the browser tab is hidden (`document.hidden`).

---

## Admin API Bug Fix

**Important:** Before building the frontend, fix the route path bug in `src/admin/config.py` line 115:

```python
# WRONG (double /config prefix):
@config_router.put("/config/automation")

# CORRECT:
@config_router.put("/automation")
```

The `config_router` is already mounted with `prefix="/config"` in `app.py`, so the route decorator should be `/automation` not `/config/automation`.

---

## Exit Criteria

### Core Infrastructure
1. `admin-ui/` directory with Vite + React + TypeScript + Tailwind setup
2. `npm install && npm run build` completes without errors
3. `npm run dev` starts dev server on port 5174
4. API client (`client.ts`) handles GET/POST/PUT/DELETE with error handling
5. TypeScript types (`types.ts`) cover all 27 API response shapes
6. Custom hooks (`hooks.ts`) with polling + visibility-aware auto-refresh

### Layout & Navigation
7. Sidebar with 8 nav links + health indicators
8. Header with health badge and uptime display
9. React Router with all 8 routes

### Dashboard Page
10. 6 subsystem health cards from `/health`
11. Quick stats panel from `/analyses/stats`
12. Real-time event feed via `/ws/events` WebSocket

### Scheduler Page
13. Status panel with next-run countdown
14. "Run Watchlist Now" button with async TaskPoller
15. Run history table from `/scheduler/history`

### Queue Page
16. Status panel with pending/processing/completed counts
17. Pending candidates table sorted by priority
18. Enqueue form that creates Scanner-format queue files
19. Retry and Clear actions

### Accuracy Page
20. Summary cards (total, pending, direction accuracy, avg return)
21. Direction accuracy bar chart (T+1/T+5/T+10 by decision)
22. Quality tier correlation chart
23. Best/worst signals tables
24. Ticker drill-down search
25. Update Now and Backfill buttons

### Test Run Page
26. Input form with ticker, hybrid config dropdown, date picker, publish toggle
27. Async analysis execution with TaskPoller (30-120s)
28. Structured result display (decision badge, trade params, quality scores)
29. Recent test runs table

### Analyses Page
30. Stats header from `/analyses/stats`
31. Analyses table with outcome_status from LEFT JOIN
32. Expandable row detail from `/analyses/{id}`
33. Ticker and date range filters

### Configuration Page
34. Automation config editor with save + restart indicators
35. Supabase config editor
36. Watchlist manager (list, edit, create)
37. Hybrid configs read-only table

### Logs Page
38. Live log stream via WebSocket with level coloring
39. Level filter dropdown
40. Auto-scroll toggle and pause/resume controls

### Shared Components
41. StatusDot, DataTable, TaskPoller, JsonViewer, EmptyState components
42. WebSocket manager with auto-reconnect

### Build & Integration
43. `vite build` outputs to `src/admin/static/`
44. Admin API serves static files when `src/admin/static/` exists
45. Config route bug fix applied (`/config/automation` → `/automation`)

### Quality
46. Zero TypeScript errors (`tsc --noEmit` passes)
47. All API calls handle loading, error, and empty states
48. Dark theme applied consistently
