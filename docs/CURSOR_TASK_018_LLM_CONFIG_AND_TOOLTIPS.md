# Task 018 — LLM Configuration Editor, Sanity Check, A/B Comparison & Settings Tooltips

**Type:** Feature (Admin API + Admin UI)
**Depends on:** Task 017 (Admin Dashboard Frontend)
**Repo:** trifecta-trader-poc
**Priority:** High

---

## Objective

Extend the admin dashboard with four capabilities:

1. **Editable LLM Configurations** — Externalize hybrid LLM configs from Python code to YAML, add CRUD API endpoints, and build a full editor UI so admins can create, edit, clone, and delete LLM configurations without touching code.
2. **LLM Sanity Check** — A per-config "Test Connection" button that validates reachability and basic response capability for each provider/model in the config before running a full analysis.
3. **A/B Comparison** — Run the same ticker analysis with two different LLM configs side by side and compare results (decision, quality, trade params, cost, latency) in a single view.
4. **Settings Info Tooltips** — Add an `(i)` icon next to every label on the Config page that shows a popover with a description, valid ranges, and caveats for each setting.

---

## Background

Currently, hybrid LLM configs are hardcoded as `HybridLLMConfig` instances in `src/hybrid_llm.py` → `CONFIGS` dict (12 presets). The admin UI shows them in a read-only table. To test a new model, a developer must edit Python code and restart — this blocks non-developer admins and makes A/B comparisons cumbersome.

The goal is: **edit → verify → compare → deploy** entirely from the browser.

---

## Part 1 — Externalize Hybrid LLM Configs to YAML

### 1.1 New file: `config/hybrid_llm.yaml`

Create a YAML file that mirrors the current `CONFIGS` dict. On first load, if the file doesn't exist, auto-generate it from the existing Python `CONFIGS`.

**Schema per config entry:**

```yaml
configs:
  hybrid_haiku_tools:
    tool_provider: "anthropic"
    tool_model: "claude-haiku-4-5-20251001"
    reasoning_quick_provider: "ollama"
    reasoning_quick_model: "qwen2.5:14b"
    reasoning_deep_provider: "anthropic"
    reasoning_deep_model: "claude-sonnet-4-5-20250929"
    enhance_local: true
    enhance_style: "financial_analysis"
    enhance_deep: true
    enhance_deep_style: "execution_params_only"

  all_cloud:
    tool_provider: "anthropic"
    tool_model: "claude-sonnet-4-5-20250929"
    reasoning_quick_provider: "anthropic"
    reasoning_quick_model: "claude-sonnet-4-5-20250929"
    reasoning_deep_provider: "anthropic"
    reasoning_deep_model: "claude-sonnet-4-5-20250929"
    enhance_local: false
    enhance_style: "financial_analysis"
    enhance_deep: false
    enhance_deep_style: "execution_params_only"

  # ... all 12 existing configs migrated
```

### 1.2 Modify `src/hybrid_llm.py`

**Keep the existing Python `CONFIGS` dict as the fallback/default**. Add a new function:

```python
def load_configs() -> Dict[str, HybridLLMConfig]:
    """Load hybrid configs from YAML, falling back to Python defaults.

    On first call, if config/hybrid_llm.yaml doesn't exist, generate it
    from the Python CONFIGS dict and write to disk.
    """
```

- `load_configs()` reads `config/hybrid_llm.yaml`
- If the file doesn't exist, it calls `_generate_yaml_from_defaults()` which writes the Python `CONFIGS` to YAML, then returns them
- Each YAML entry is converted to a `HybridLLMConfig` instance
- The module-level `CONFIGS` dict should still work for backward compatibility — on import, set `CONFIGS = load_configs()`

**Important:** `run_analysis.py` line 274 does `hybrid_config = CONFIGS[hybrid]` — this must continue to work. Since `CONFIGS` is reassigned at module level on import, existing code paths are preserved.

### 1.3 Add `save_config()` and `delete_config()` helpers

```python
def save_config(name: str, config: HybridLLMConfig) -> None:
    """Save/update a single config entry. Writes full YAML file."""

def delete_config(name: str) -> None:
    """Remove a config entry. Raises KeyError if not found."""

def reload_configs() -> Dict[str, HybridLLMConfig]:
    """Re-read YAML from disk. Use after external edits."""
```

These operate on the YAML file and update the in-memory `CONFIGS` dict.

---

## Part 2 — Admin API Endpoints for Hybrid LLM Configs

### 2.1 Modify existing endpoint

**`GET /config/hybrid-configs`** — Already exists but only reads from Python. Modify to use `load_configs()`:

Response shape remains the same but now includes all YAML fields:

```json
{
  "configs": [
    {
      "name": "hybrid_haiku_tools",
      "tool_provider": "anthropic",
      "tool_model": "claude-haiku-4-5-20251001",
      "reasoning_quick_provider": "ollama",
      "reasoning_quick_model": "qwen2.5:14b",
      "reasoning_deep_provider": "anthropic",
      "reasoning_deep_model": "claude-sonnet-4-5-20250929",
      "enhance_local": true,
      "enhance_style": "financial_analysis",
      "enhance_deep": true,
      "enhance_deep_style": "execution_params_only"
    }
  ],
  "active": "hybrid_haiku_tools",
  "providers": ["anthropic", "ollama", "openai", "google", "xai", "openrouter"],
  "enhance_styles": ["financial_analysis", "structured", "few_shot", "execution_params_only"]
}
```

Add `providers` (from the factory) and `enhance_styles` (from `enhanced_llm.py`) so the UI can populate dropdowns without hardcoding values.

### 2.2 New endpoints

**`POST /config/hybrid-configs`** — Create a new config

Request body:
```json
{
  "name": "my_new_config",
  "tool_provider": "anthropic",
  "tool_model": "claude-haiku-4-5-20251001",
  "reasoning_quick_provider": "ollama",
  "reasoning_quick_model": "llama3.1:8b",
  "reasoning_deep_provider": "anthropic",
  "reasoning_deep_model": "claude-sonnet-4-5-20250929",
  "enhance_local": false,
  "enhance_style": "financial_analysis",
  "enhance_deep": false,
  "enhance_deep_style": "execution_params_only"
}
```

Validation:
- `name` must be non-empty, alphanumeric + underscores, max 64 chars
- `name` must not already exist (409 Conflict)
- `tool_provider`, `reasoning_quick_provider`, `reasoning_deep_provider` must be in the known providers list
- `enhance_style` and `enhance_deep_style` must be in the known styles list

Response: `201` with the created config object.

**`PUT /config/hybrid-configs/{name}`** — Update an existing config

Same body as POST (minus `name`). Returns the updated config.
- 404 if `name` doesn't exist
- Immediately updates in-memory `CONFIGS` dict + writes YAML

**`DELETE /config/hybrid-configs/{name}`** — Delete a config

- 404 if not found
- 409 if config is currently set as `active` in `automation.yaml` (scheduler or queue_reader `hybrid_config`)
- Returns `204 No Content`

**`POST /config/hybrid-configs/{name}/clone`** — Clone a config

Request body:
```json
{
  "new_name": "my_cloned_config"
}
```

- Copies all fields from `{name}` to `{new_name}`
- Same name validation as create
- Returns `201` with the new config

### 2.3 Sanity check endpoint

**`POST /config/hybrid-configs/{name}/sanity-check`** — Test connectivity for all 3 LLM slots

This endpoint tests each provider/model in the config by sending a minimal prompt and checking for a valid response.

Logic per slot (tool_calling, reasoning_quick, reasoning_deep):

1. **Ollama** — `POST http://localhost:11434/api/generate` with `{"model": "<model>", "prompt": "Say OK", "stream": false}`. Timeout 10s. Check for 200 + non-empty `response` field.
2. **Anthropic** — `POST https://api.anthropic.com/v1/messages` with a minimal 1-token request using the `ANTHROPIC_API_KEY` env var. Timeout 10s.
3. **OpenAI / OpenRouter / XAI** — `POST` to the appropriate base URL with a minimal chat completion. Timeout 10s. Use appropriate env var (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`).
4. **Google** — `POST` via the google client with a minimal request. Timeout 10s.

**Response:**

```json
{
  "config_name": "my_new_config",
  "overall": "pass",
  "checks": {
    "tool_calling": {
      "provider": "anthropic",
      "model": "claude-haiku-4-5-20251001",
      "status": "pass",
      "latency_ms": 342,
      "error": null
    },
    "reasoning_quick": {
      "provider": "ollama",
      "model": "llama3.1:8b",
      "status": "fail",
      "latency_ms": null,
      "error": "Connection refused: Ollama not running"
    },
    "reasoning_deep": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-5-20250929",
      "status": "pass",
      "latency_ms": 891,
      "error": null
    }
  }
}
```

`overall` is `"pass"` if all 3 slots pass, `"partial"` if some pass, `"fail"` if none pass.

**Important:** The sanity check should NOT use the full `create_hybrid_llms()` pipeline or `TradingAgentsGraph` — it should do lightweight HTTP-level checks only. This avoids importing langchain and keeps the check fast (< 15s total). Use raw `requests` or `httpx` calls.

**Implementation notes:**
- For Ollama: `POST http://localhost:11434/api/generate` body `{"model": "...", "prompt": "Respond with only the word OK.", "stream": false}`
- For Anthropic: `POST https://api.anthropic.com/v1/messages` with `{"model": "...", "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK"}]}` and header `x-api-key: $ANTHROPIC_API_KEY`, `anthropic-version: 2023-06-01`
- For OpenAI/OpenRouter/XAI: `POST <base_url>/chat/completions` with `{"model": "...", "max_tokens": 5, "messages": [{"role": "user", "content": "Say OK"}]}` and `Authorization: Bearer $API_KEY`
  - OpenAI base: `https://api.openai.com/v1`
  - OpenRouter base: `https://openrouter.ai/api/v1`
  - XAI base: `https://api.x.ai/v1`
- For Google: Use the `google.generativeai` library if available, otherwise skip with `"status": "skip", "error": "google SDK not installed"`
- Each check runs independently; one failure doesn't prevent others from running
- All 3 checks run concurrently (use `asyncio.gather` or `ThreadPoolExecutor`)

---

## Part 3 — Admin UI: Editable LLM Config Panel

### 3.1 Replace the existing `HybridConfigsPanel`

The current read-only table becomes a full editor. Structure:

**Config List (left/top)**
- Clickable cards/buttons for each config name (same style as watchlist selector)
- Active config highlighted with blue badge
- "New Config" button

**Config Editor (main area)** — when a config is selected:

```
┌─────────────────────────────────────────────────────────┐
│  hybrid_haiku_tools                    [active badge]    │
│                                                         │
│  ┌─ Tool Calling ───────────────────────────────────┐   │
│  │  Provider: [anthropic ▼]   Model: [claude-ha...] │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Reasoning Quick ────────────────────────────────┐   │
│  │  Provider: [ollama    ▼]   Model: [qwen2.5:14b ] │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Reasoning Deep ─────────────────────────────────┐   │
│  │  Provider: [anthropic ▼]   Model: [claude-so...] │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Enhancement ────────────────────────────────────┐   │
│  │  Enhance local: [✓]   Style: [financial_analysis]│   │
│  │  Enhance deep:  [✓]   Style: [execution_params]  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  [Sanity Check]  [Clone]  [Save]  [Delete]              │
└─────────────────────────────────────────────────────────┘
```

- **Provider dropdown:** Populated from `providers` array in GET response
- **Model field:** Free text input (models change constantly — don't restrict)
- **Enhancement section:** Checkboxes for `enhance_local` and `enhance_deep`, dropdowns for styles (from `enhance_styles` array)
- **Sanity Check button:** Calls `POST /config/hybrid-configs/{name}/sanity-check`, shows a 3-row status card with per-slot results (green check / red X / spinner)
- **Clone button:** Opens a small inline input for the new name, calls the clone endpoint
- **Save button:** Calls PUT to update, shows success/error toast
- **Delete button:** Confirmation dialog, calls DELETE. Disabled if config is active.

### 3.2 New Config flow

"New Config" button at the top shows a form with:
- Name field (validated: alphanumeric + underscores)
- All 6 provider/model fields pre-filled with sensible defaults (copy from `hybrid_haiku_tools`)
- Enhancement toggles defaulting to off
- "Create" button → POST → shows in the config list

### 3.3 Sanity Check UI

When "Sanity Check" is clicked:

1. Button shows spinner, disabled
2. Results appear in a card below the config editor:

```
┌─ Sanity Check Results ──────────────────────────────┐
│  ✅ Tool Calling      anthropic / claude-haiku  342ms│
│  ❌ Reasoning Quick   ollama / qwen2.5:14b      —   │
│     └─ Connection refused: Ollama not running        │
│  ✅ Reasoning Deep    anthropic / claude-sonnet 891ms│
│                                                      │
│  Overall: PARTIAL (2/3 passed)                       │
└──────────────────────────────────────────────────────┘
```

- Green checkmark + latency for pass
- Red X + error message for fail
- Overall summary line with pass/partial/fail badge

### 3.4 TypeScript types

Add to `types.ts`:

```typescript
export interface HybridConfigFull {
  name: string;
  tool_provider: string;
  tool_model: string;
  reasoning_quick_provider: string;
  reasoning_quick_model: string;
  reasoning_deep_provider: string;
  reasoning_deep_model: string;
  enhance_local: boolean;
  enhance_style: string;
  enhance_deep: boolean;
  enhance_deep_style: string;
}

export interface HybridConfigsResponse {
  configs: HybridConfigFull[];
  active: string | null;
  providers: string[];
  enhance_styles: string[];
}

export interface SanityCheckSlot {
  provider: string;
  model: string;
  status: 'pass' | 'fail' | 'skip';
  latency_ms: number | null;
  error: string | null;
}

export interface SanityCheckResult {
  config_name: string;
  overall: 'pass' | 'partial' | 'fail';
  checks: {
    tool_calling: SanityCheckSlot;
    reasoning_quick: SanityCheckSlot;
    reasoning_deep: SanityCheckSlot;
  };
}
```

---

## Part 4 — Settings Info Tooltips

### 4.1 Shared `InfoTooltip` component

Create `src/components/shared/InfoTooltip.tsx`:

- Renders a small `(i)` icon (use `Info` from lucide-react, ~14px, `text-slate-500`)
- On click: toggles a popover positioned to the right of the icon
- Popover has dark background (`bg-slate-700`), rounded corners, max-width 300px
- Shows the help text (supports simple markdown: bold, line breaks)
- Clicking outside or pressing Escape closes it
- Accessible: `role="tooltip"`, `aria-describedby`

### 4.2 Modify `FieldRow` component

Add an optional `info?: string` prop to `FieldRow`:

```tsx
function FieldRow({
  label, value, onChange, type = 'text', info,
}: {
  label: string;
  value: string | number | boolean;
  onChange: (v: string | number | boolean) => void;
  type?: 'text' | 'number' | 'boolean';
  info?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2 ...">
      <span className="text-xs text-slate-400 w-48 shrink-0 flex items-center gap-1.5">
        {label}
        {info && <InfoTooltip text={info} />}
      </span>
      {/* ... existing input logic */}
    </div>
  );
}
```

### 4.3 Help text content

Define a `SETTINGS_HELP` constant object in the ConfigPage (or a separate `config-help.ts` file). Cover every field:

**Scheduler section:**
- `Enabled` — "Master switch for the daily watchlist scanner. When disabled, no scheduled scans will run. The queue reader and accuracy updater operate independently."
- `Hour` / `Minute` — "Time of day (24h format) in the configured timezone when the watchlist scan runs. Default 8:30 AM ET gives ~60 minutes before market open for analyses to complete."
- `Hybrid config` — "Which LLM configuration to use for scheduled analyses. This determines which AI models handle tool-calling, quick reasoning, and deep reasoning (Risk Judge). Edit configs in the LLM Configuration section below."
- `Watchlist` — "Name of the watchlist file to scan (from config/watchlists/). Each watchlist contains a list of ticker symbols to analyze."
- `Publish` — "When enabled, analysis results are automatically published to Supabase for the Platform UI. Disable for testing or when Supabase is not configured."

**Queue Reader section:**
- `Enabled` — "Master switch for the file-based queue reader. When enabled, the daemon polls the queue directory for candidate JSON files from the Market Scanner."
- `Poll interval (s)` — "How often (in seconds) to check for new queue candidates. Lower values mean faster processing but more filesystem I/O. Range: 5-300. Takes effect immediately."
- `Max retries` — "Maximum number of retry attempts for a failed analysis before marking the candidate as permanently failed. Range: 0-10. Takes effect immediately."
- `Cooldown (s)` — "Minimum wait time (in seconds) between consecutive analyses to respect LLM rate limits. Range: 10-600. Takes effect immediately."

**Accuracy section:**
- `Enabled` — "Master switch for the signal accuracy tracker. When enabled, tracks price movements at T+1, T+5, and T+10 trading days after each signal."
- `Backfill on start` — "When enabled, automatically scores all existing untracked analyses when the daemon first starts. Useful after adding accuracy tracking to an existing deployment."

**Admin API section:**
- `Port` — "TCP port for the Admin API server. Default 8420. Requires daemon restart to take effect."

**Supabase section:**
- `Write enabled` — "When enabled, analyses are published to the Supabase signals table. Disable to run analyses without publishing. Takes effect immediately."
- `Signal TTL (hrs)` — "How many hours a published signal remains active in Supabase before being considered stale. The Platform UI uses this to filter signals. Range: 1-168 (1 week)."

**Watchlist Manager section:** (add info to the section header, not per-field)
- Section-level info: "Watchlists define which tickers the scheduler analyzes. Each watchlist is a YAML file in config/watchlists/. The active watchlist is set in the Scheduler config above."

**LLM Configuration section:**
- Section-level info: "Hybrid LLM configurations define which AI providers and models handle each agent role. Tool-calling agents need models that support function calling (e.g., Anthropic Claude, OpenAI GPT-4). Reasoning agents can use any model including local Ollama models."

---

## Part 5 — A/B Comparison (Test Run Enhancement)

### 5.1 New API endpoint

**`POST /test-run/ab`** — Submit two parallel test runs with different LLM configs

Request body:
```json
{
  "ticker": "AAPL",
  "trade_date": "2026-03-05",
  "config_a": "hybrid_haiku_tools",
  "config_b": "my_new_config",
  "publish": false
}
```

Response (`202`):
```json
{
  "ab_id": "ab_AAPL_20260305_143022",
  "task_id_a": "test_AAPL_20260305_143022_a",
  "task_id_b": "test_AAPL_20260305_143022_b",
  "status": "running",
  "ticker": "AAPL",
  "started_at": "2026-03-05T14:30:22Z"
}
```

Implementation:
- Validates both config names exist (404 if either is missing)
- Submits two tasks to `TaskManager` — they run concurrently if `max_workers >= 2` (which it is — max_workers=2)
- Each task calls `_run_analysis_safe()` with a different `hybrid_config`
- The `ab_id` is a grouping ID returned so the UI can poll both tasks

**`GET /test-run/ab/{ab_id}`** — Poll A/B comparison status

Response:
```json
{
  "ab_id": "ab_AAPL_20260305_143022",
  "ticker": "AAPL",
  "trade_date": "2026-03-05",
  "status": "complete",
  "config_a": {
    "name": "hybrid_haiku_tools",
    "task_id": "test_AAPL_20260305_143022_a",
    "status": "complete",
    "result": { ... }
  },
  "config_b": {
    "name": "my_new_config",
    "task_id": "test_AAPL_20260305_143022_b",
    "status": "complete",
    "result": { ... }
  }
}
```

- Overall `status` is `"running"` if either task is still running, `"complete"` if both are done (complete or error)
- Each side includes its full result (same shape as the existing single test run)

### 5.2 A/B Comparison UI

Add a new tab/mode to the Test Run page. The page gets two modes:

**Mode toggle** at the top: `[Single Run]  [A/B Compare]`

**A/B Compare form:**

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚗️  A/B LLM Comparison                                        │
│                                                                 │
│  Ticker: [AAPL]     Trade Date: [2026-03-05]                   │
│                                                                 │
│  Config A: [hybrid_haiku_tools  ▼]  (current)                  │
│  Config B: [my_new_config       ▼]                              │
│                                                                 │
│  [ ] Publish to Supabase  ⚠️ Writes to production              │
│                                                                 │
│  [Run A/B Comparison]                                           │
└─────────────────────────────────────────────────────────────────┘
```

- Config A defaults to the currently active config
- Config B defaults to the next one in the list (or empty)
- Both dropdowns populated from `GET /config/hybrid-configs`

**Side-by-side results:**

When both complete, display results in a two-column layout:

```
┌─ Config A: hybrid_haiku_tools ──────┬─ Config B: my_new_config ──────────┐
│                                     │                                     │
│  Decision: BUY                      │  Decision: BUY                      │
│  Quality:  7.8/10                   │  Quality:  6.2/10                   │
│                                     │                                     │
│  ┌─ Quality Breakdown ────────┐     │  ┌─ Quality Breakdown ────────┐     │
│  │ data_citation    8.5  ████ │     │  │ data_citation    5.0  ███  │     │
│  │ reasoning_depth  7.0  ███▌ │     │  │ reasoning_depth  6.5  ███  │     │
│  │ risk_assessment  8.0  ████ │     │  │ risk_assessment  7.0  ███▌ │     │
│  │ ...                        │     │  │ ...                        │     │
│  └────────────────────────────┘     │  └────────────────────────────┘     │
│                                     │                                     │
│  ┌─ Trade Params ─────────────┐     │  ┌─ Trade Params ─────────────┐     │
│  │ Entry:  $185.20            │     │  │ Entry:  $185.20            │     │
│  │ Stop:   $175.94            │     │  │ Stop:   $178.00            │     │
│  │ Target: $210.00            │     │  │ Target: $205.00            │     │
│  │ R/R:    2.7:1              │     │  │ R/R:    1.9:1              │     │
│  └────────────────────────────┘     │  └────────────────────────────┘     │
│                                     │                                     │
│  Cost: $0.042   Elapsed: 45.2s     │  Cost: $0.008   Elapsed: 32.1s     │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

Key UI behaviors:
- While running, each column shows a spinner independently (one may finish before the other)
- When both complete, show a **Comparison Summary** bar at the top highlighting key differences:
  - Same/different decision (highlighted if different)
  - Quality score delta (e.g., "+1.6" or "-1.6" with green/red color)
  - Cost delta (e.g., "A is 5.3x more expensive")
  - Speed delta (e.g., "B is 29% faster")
- If one side errors, show the error on that side while still showing the other's result
- Expandable `JsonViewer` for full results on each side

### 5.3 A/B state management

Store A/B metadata in a simple in-memory dict on the server side (similar to `TaskManager`):

```python
# In test_run.py or a small ab_store module
_AB_STORE: Dict[str, dict] = {}  # ab_id → {ticker, trade_date, config_a, config_b, task_id_a, task_id_b}
```

The GET endpoint uses `_AB_STORE[ab_id]` to find the two task IDs, then queries `TaskManager` for each task's status and result. No database needed — A/B results are ephemeral (same as regular test runs).

Cap at 20 entries, evict oldest when full (same pattern as `TaskManager`).

### 5.4 TypeScript types

Add to `types.ts`:

```typescript
export interface ABCompareRequest {
  ticker: string;
  trade_date: string;
  config_a: string;
  config_b: string;
  publish?: boolean;
}

export interface ABCompareResponse {
  ab_id: string;
  ticker: string;
  trade_date: string;
  status: 'running' | 'complete';
  config_a: {
    name: string;
    task_id: string;
    status: 'running' | 'complete' | 'error';
    result: unknown;
  };
  config_b: {
    name: string;
    task_id: string;
    status: 'running' | 'complete' | 'error';
    result: unknown;
  };
}
```

---

## Deliverables

| # | Deliverable | Scope |
|---|-------------|-------|
| 1 | `config/hybrid_llm.yaml` — auto-generated from Python defaults | Python |
| 2 | `src/hybrid_llm.py` — `load_configs()`, `save_config()`, `delete_config()`, `reload_configs()` with YAML read/write, backward-compatible `CONFIGS` dict | Python |
| 3 | `src/admin/config.py` — new endpoints: `POST /config/hybrid-configs`, `PUT /config/hybrid-configs/{name}`, `DELETE /config/hybrid-configs/{name}`, `POST /config/hybrid-configs/{name}/clone` | Python |
| 4 | `src/admin/sanity_check.py` — new file: `POST /config/hybrid-configs/{name}/sanity-check` with per-provider HTTP checks | Python |
| 5 | Modified `GET /config/hybrid-configs` to use `load_configs()` + return `providers` and `enhance_styles` | Python |
| 6 | `src/admin/test_run.py` — new endpoints: `POST /test-run/ab` and `GET /test-run/ab/{ab_id}` for A/B comparison | Python |
| 7 | `admin-ui/src/api/types.ts` — new types: `HybridConfigFull`, `HybridConfigsResponse`, `SanityCheckSlot`, `SanityCheckResult`, `ABCompareRequest`, `ABCompareResponse` | TypeScript |
| 8 | `admin-ui/src/components/shared/InfoTooltip.tsx` — reusable tooltip component | TypeScript |
| 9 | Modified `admin-ui/src/components/config/ConfigPage.tsx` — editable LLM config panel replacing read-only table, sanity check UI, info tooltips on all settings | TypeScript |
| 10 | Modified `admin-ui/src/components/test-run/TestRunPage.tsx` — A/B comparison mode with side-by-side results | TypeScript |
| 11 | `admin-ui/src/components/config/config-help.ts` — help text definitions | TypeScript |
| 12 | Tests for `load_configs`, `save_config`, `delete_config`, `reload_configs` | Python |
| 13 | Tests for all new/modified API endpoints (CRUD + sanity check + A/B) | Python |
| 14 | `npm run build` succeeds with 0 errors, `tsc --noEmit` clean | TypeScript |

---

## Exit Criteria

### Python — hybrid_llm.py changes (8 criteria)

1. `config/hybrid_llm.yaml` is auto-generated on first `load_configs()` call when file doesn't exist
2. Generated YAML contains all 12 existing configs with all 10 fields each
3. `load_configs()` returns `Dict[str, HybridLLMConfig]` from YAML
4. `save_config(name, config)` writes to YAML and updates in-memory `CONFIGS`
5. `delete_config(name)` removes from YAML and in-memory `CONFIGS`; raises `KeyError` if not found
6. `reload_configs()` re-reads from disk and returns updated dict
7. Module-level `CONFIGS = load_configs()` preserves backward compatibility — `from src.hybrid_llm import CONFIGS` still works
8. `run_analysis.py` line `CONFIGS[hybrid]` still resolves correctly

### Python — API endpoints (15 criteria)

9. `GET /config/hybrid-configs` returns full config objects (all 10 fields) plus `providers` and `enhance_styles` arrays
10. `POST /config/hybrid-configs` creates a new config with validation: name format, uniqueness (409), valid providers, valid enhance styles
11. `PUT /config/hybrid-configs/{name}` updates existing config; 404 if not found
12. `DELETE /config/hybrid-configs/{name}` removes config; 404 if not found; 409 if currently active in automation.yaml
13. `POST /config/hybrid-configs/{name}/clone` creates a copy with new name; 404 if source not found; 409 if new_name exists
14. Sanity check endpoint sends minimal prompt to each provider and returns per-slot pass/fail/skip with latency
15. Sanity check handles Ollama unreachable gracefully (fail, not crash)
16. Sanity check handles missing API keys gracefully (fail with descriptive error, not crash)
17. Sanity check runs all 3 slots concurrently (not sequentially)
18. `POST /test-run/ab` validates both config names, submits two concurrent tasks via TaskManager, returns `ab_id` + both `task_id`s
19. `GET /test-run/ab/{ab_id}` returns combined status: `"running"` if either task is active, `"complete"` when both are done; includes full results for each side
20. A/B store caps at 20 entries, evicts oldest when full
21. All existing tests still pass (no regressions in the 199 admin + accuracy + daemon tests)
22. New endpoint tests cover: create, create duplicate (409), update, update nonexistent (404), delete, delete active (409), clone, sanity check pass, sanity check fail, A/B submit, A/B poll, A/B with invalid config (404)
23. At least 30 new test cases for the new endpoints and hybrid_llm changes

### TypeScript — Admin UI (22 criteria)

24. `HybridConfigFull`, `HybridConfigsResponse`, `SanityCheckSlot`, `SanityCheckResult`, `ABCompareRequest`, `ABCompareResponse` types added to `types.ts`
25. Config list shows all configs as selectable cards/buttons with active badge
26. Selecting a config opens the editor with all 10 fields populated
27. Provider dropdowns populated from API `providers` array (not hardcoded)
28. Enhance style dropdowns populated from API `enhance_styles` array (not hardcoded)
29. Model fields are free-text inputs (not restricted to known models)
30. "New Config" button shows creation form with defaults, name validation, calls POST
31. "Save" button calls PUT, shows success/error toast
32. "Delete" button shows confirmation, calls DELETE, disabled if active, removes from list
33. "Clone" button prompts for new name, calls clone endpoint, shows new config in list
34. "Sanity Check" button calls the sanity-check endpoint, shows spinner while running
35. Sanity check results card shows per-slot status (green check / red X), latency, and error messages
36. Overall pass/partial/fail badge in sanity check results
37. Test Run page has mode toggle: `[Single Run]` / `[A/B Compare]`
38. A/B form has two config dropdowns (Config A defaults to active, Config B to next in list), shared ticker + date + publish fields
39. A/B results render in a two-column side-by-side layout with each side showing decision, quality breakdown, trade params, cost, elapsed
40. Each column shows a spinner independently while its analysis is running
41. Comparison summary bar shows key deltas: same/different decision, quality delta (colored), cost multiplier, speed difference
42. If one side errors, error is shown on that side; other side still renders normally
43. `InfoTooltip` component renders `(i)` icon, click toggles popover, click-outside closes
44. Every field in Automation, Queue Reader, Accuracy, Admin API, and Supabase panels has an `info` tooltip
45. Help text is accurate and matches the definitions in Part 4.3 of this spec

### Build & Quality (4 criteria)

46. `tsc --noEmit` — 0 errors
47. `npm run build` — 0 errors
48. All new Python tests pass
49. No regressions: existing 199 admin + accuracy + daemon tests still pass

---

## Implementation Notes

### Sanity check — keep it lightweight

The sanity check MUST NOT import langchain, tradingagents, or any heavy ML library. It should use only `requests` or `httpx` for HTTP calls. The goal is a fast (< 15s) connectivity test, not a model quality evaluation. The existing `_check_ollama()` in `health.py` is a good reference pattern.

### Provider base URLs

For the sanity check HTTP calls, use these base URLs:
- **Ollama:** `http://localhost:11434/api/generate`
- **Anthropic:** `https://api.anthropic.com/v1/messages`
- **OpenAI:** `https://api.openai.com/v1/chat/completions`
- **OpenRouter:** `https://openrouter.ai/api/v1/chat/completions`
- **XAI:** `https://api.x.ai/v1/chat/completions`
- **Google:** Use `google.generativeai` SDK if installed, otherwise return `status: "skip"`

### API key env vars

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY` (or `OPENROUTER_API_KEY`)
- `XAI_API_KEY`
- `GOOGLE_API_KEY`

If the env var is missing for a provider, the sanity check slot should return `status: "fail"` with `error: "API key not configured (missing XXXX_API_KEY)"`.

### YAML write safety

When writing `hybrid_llm.yaml`, always:
1. Load existing content
2. Merge/modify in memory
3. Write atomically (write to temp file, then rename)
4. Log the change

### InfoTooltip positioning

Use a simple absolute-positioned div. The popover should appear to the right of the icon by default. If it would overflow the viewport, flip to the left. Keep it simple — no need for a tooltip library.

### Backward compatibility

The `src/hybrid_llm.py` changes must be backward-compatible:
- `from src.hybrid_llm import CONFIGS` must still return a dict
- `from src.hybrid_llm import HybridLLMConfig` must still work
- `from src.hybrid_llm import create_hybrid_llms` must still work
- CLI usage via `--hybrid hybrid_haiku_tools` must still resolve

### A/B comparison concurrency

The `TaskManager` already has `max_workers=2`, so both A/B analyses can run truly in parallel. This is important — if max_workers were 1, the second analysis would queue behind the first, defeating the purpose of A/B testing. Do NOT change max_workers.

If the user submits multiple A/B comparisons rapidly, they'll queue in the ThreadPoolExecutor. This is acceptable — the UI should disable the "Run A/B Comparison" button while any A/B test is in progress.

### A/B result matching

Both sides of the A/B comparison must use the **exact same** `ticker` and `trade_date` to ensure an apples-to-apples comparison. The API should enforce this — both tasks inherit ticker and trade_date from the A/B request body, not from separate inputs.

---

## File Inventory (expected changes)

### New files
- `config/hybrid_llm.yaml` (auto-generated)
- `src/admin/sanity_check.py`
- `admin-ui/src/components/shared/InfoTooltip.tsx`
- `admin-ui/src/components/config/config-help.ts`
- `tests/test_hybrid_llm_yaml.py`
- `tests/test_admin_sanity_check.py`
- `tests/test_admin_ab_compare.py`

### Modified files
- `src/hybrid_llm.py` (add load/save/delete/reload functions)
- `src/admin/config.py` (add CRUD endpoints, modify GET hybrid-configs)
- `src/admin/test_run.py` (add A/B endpoints + `_AB_STORE`)
- `src/admin/app.py` (mount sanity_check_router)
- `admin-ui/src/api/types.ts` (add new types)
- `admin-ui/src/components/config/ConfigPage.tsx` (major rewrite of LLM section + tooltips)
- `admin-ui/src/components/test-run/TestRunPage.tsx` (add A/B mode toggle + side-by-side results)
- `tests/test_admin_config.py` (add tests for new endpoints)
- `tests/test_admin_test_run.py` (add A/B tests)
