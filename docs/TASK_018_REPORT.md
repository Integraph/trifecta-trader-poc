# Task 018 Report — LLM Config & Tooltips

**Completed:** 2026-02-28

---

## Objective

Extend the Admin Dashboard with:
1. Externalized, editable LLM configurations (YAML-backed)
2. Full CRUD API for hybrid LLM configs
3. Sanity-check endpoint (concurrent HTTP connectivity tests)
4. A/B comparison for parallel test runs
5. Info tooltips on all settings fields
6. Editable LLM Config panel in the Admin UI

---

## Deliverables

### Part 1: YAML Externalization (`src/hybrid_llm.py` + `config/hybrid_llm.yaml`)

- `config/hybrid_llm.yaml` — created with all **13 existing configs** pre-populated.
- `HybridLLMConfig.to_flat_dict()` — new method returning all 10 fields for YAML serialization.
- `_DEFAULT_CONFIGS` — Python dict (renamed from `CONFIGS`) used as fallback when YAML is absent.
- `KNOWN_PROVIDERS = ["anthropic", "ollama", "openai", "google", "xai", "openrouter"]` — canonical list.
- `KNOWN_ENHANCE_STYLES = ["financial_analysis", "structured", "few_shot", "execution_params_only"]`.
- `load_configs()` — reads from YAML; auto-generates file on first run via `_generate_yaml_from_defaults()`.
- `save_config(name, cfg)` — atomic write (tempfile + os.replace) + updates in-memory `CONFIGS`.
- `delete_config(name)` — raises `KeyError` if not found; updates YAML + `CONFIGS`.
- `reload_configs()` — re-reads from disk and clears/repopulates `CONFIGS`.
- `CONFIGS = load_configs()` — module-level dict stays fully backward-compatible.

### Part 2: Admin API Endpoints (`src/admin/config.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/config/hybrid-configs` | Returns all configs with full 10 fields + `providers` + `enhance_styles` |
| `POST` | `/config/hybrid-configs` | Create new config (validates name regex, uniqueness, provider, style) |
| `PUT`  | `/config/hybrid-configs/{name}` | Update existing config |
| `DELETE` | `/config/hybrid-configs/{name}` | Delete config (409 if currently active) |
| `POST` | `/config/hybrid-configs/{name}/clone` | Clone to new name (validates new_name) |

**Validation:** Name must match `^[a-zA-Z0-9_]{1,64}$`. Providers must be in `KNOWN_PROVIDERS`. Styles must be in `KNOWN_ENHANCE_STYLES`.

### Part 3: Sanity Check (`src/admin/sanity_check.py`)

- Mounted at `POST /config/hybrid-configs/{name}/sanity-check`.
- Tests all 3 slots (tool_calling, reasoning_quick, reasoning_deep) **concurrently** via `ThreadPoolExecutor(max_workers=3)`.
- Per-provider dispatchers: Ollama (local HTTP), Anthropic (API v1/messages), OpenAI-compatible (openai/openrouter/xai), Google (genai SDK with graceful skip if not installed).
- Returns `{ config_name, overall: pass|partial|fail, checks: { slot: { provider, model, status, latency_ms, error } } }`.
- Never imports langchain or ML libraries — pure `requests` HTTP.

### Part 4: A/B Comparison (`src/admin/test_run.py`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/test-run/ab` | Submit 2 parallel analyses, returns `ab_id` + both `task_id`s |
| `GET`  | `/test-run/ab/{ab_id}` | Poll combined status + both sides' results |

- `_AB_STORE` — `OrderedDict` capped at 20 entries (oldest evicted).
- Both tasks submitted concurrently via `TaskManager.submit()`.
- Route registered **before** `/{task_id}` to avoid path conflict.

### Part 5: TypeScript Changes (Admin UI)

**New types in `admin-ui/src/api/types.ts`:**
- `HybridConfigFull` — all 10 fields + name.
- `HybridConfigsResponse` — configs array + active + providers + enhance_styles.
- `SanityCheckSlot`, `SanityCheckResult` — per-slot and overall sanity check shape.
- `ABCompareRequest`, `ABSide`, `ABCompareResponse` — A/B comparison shapes.

**New `admin-ui/src/components/shared/InfoTooltip.tsx`:**
- Reusable `(i)` button with popover. Closes on click-outside or Escape. Auto-flips left if near viewport edge.

**New `admin-ui/src/components/config/config-help.ts`:**
- `SETTINGS_HELP` — 17 help text entries covering all scheduler, queue reader, accuracy, admin API, supabase, watchlist, and LLM config fields.

### Part 6: Config Page Rewrite (`admin-ui/src/components/config/ConfigPage.tsx`)

- `FieldRow` — accepts optional `info` prop and renders `InfoTooltip` inline.
- `AutomationPanel` — all fields wired to `SETTINGS_HELP` tooltips.
- `SupabasePanel` — all fields with tooltips.
- `WatchlistManager` — section-level tooltip.
- `LLMConfigsPanel` — full CRUD UI:
  - Config list (cards, active indicator, "New Config" button).
  - `LLMConfigEditor` — per-slot provider dropdowns + model text inputs, enhancement checkboxes/dropdowns.
  - Sanity Check button → `SanityCheckCard` with per-slot pass/fail/skip indicators.
  - Clone button → inline name input.
  - Save / Delete (with confirmation, disabled if active).

### Part 7: Test Run A/B Mode (`admin-ui/src/components/test-run/TestRunPage.tsx`)

- Mode toggle: `[Single Run]` / `[A/B Compare]`.
- `SingleRunMode` — unchanged except moved to sub-component.
- `ABCompareMode` — dual config dropdowns, ticker/date/publish fields, side-by-side result panels.
- `ComparisonSummary` — decision match, quality delta, cost multiplier, speed comparison.
- `ABSidePanel` — spinner/error/result per side.
- Polling: `setInterval` at 3000ms, stops on `status === "complete"`.

---

## Test Results

```
tests/test_task018_hybrid_llm.py   18 tests  ✓ 18 passed
tests/test_task018_config_api.py   14 tests  ✓ 14 passed
tests/test_task018_ab_compare.py   12 tests  ✓ 12 passed
────────────────────────────────────────────────────────
Total                              44 tests  ✓ 44 passed, 0 failed
```

## Build Results

```
tsc --noEmit:  ✓ 0 errors
npm run build: ✓ built in 2.17s
               3 output files: index.html, index.css (19.4kB), index.js (664.6kB)
```

---

## Design Decisions

1. **Hardcoded canonical providers list** — `KNOWN_PROVIDERS` is hardcoded (not derived from CONFIGS) so validation remains stable even as new configs are added with yet-unsupported providers.

2. **All 13 configs migrated** — The spec mentioned 12; the actual code had 13. All 13 were migrated to YAML.

3. **Atomic YAML writes** — `tempfile.mkstemp` + `os.replace()` prevents corrupt YAML on crash.

4. **Module-level `CONFIGS` mutated in-place** — `save_config` and `delete_config` update the in-memory dict so callers that hold a reference to `CONFIGS` see immediate changes without re-importing.

5. **Sanity check skips gracefully** — Google provider check returns `"skip"` if `google-generativeai` is not installed, rather than failing the overall check.

6. **A/B _AB_STORE capped at 20** — Uses `OrderedDict.popitem(last=False)` for O(1) LRU eviction.

7. **Route ordering** — `/test-run/ab` and `/test-run/ab/{ab_id}` are registered before `/{task_id}` to prevent the parameterized route from shadowing fixed paths.

---

## Exit Criteria Status

| Criterion | Status |
|-----------|--------|
| `config/hybrid_llm.yaml` created, all 13 configs present | ✓ |
| `load_configs()` reads YAML, generates on first run | ✓ |
| `save_config()` atomic write | ✓ |
| `delete_config()` raises KeyError if not found | ✓ |
| `reload_configs()` updates CONFIGS in-place | ✓ |
| Backward compat: `from src.hybrid_llm import CONFIGS` | ✓ |
| `GET /config/hybrid-configs` returns providers + styles | ✓ |
| `POST /config/hybrid-configs` validates name/provider/style | ✓ |
| `PUT /config/hybrid-configs/{name}` validates existence | ✓ |
| `DELETE /config/hybrid-configs/{name}` blocks if active | ✓ |
| `POST /config/hybrid-configs/{name}/clone` | ✓ |
| `POST /config/hybrid-configs/{name}/sanity-check` | ✓ |
| Sanity check: concurrent, all 3 slots, pass/fail/skip | ✓ |
| `POST /test-run/ab` returns ab_id + two task_ids | ✓ |
| `GET /test-run/ab/{ab_id}` returns both sides | ✓ |
| `_AB_STORE` capped at 20 | ✓ |
| TypeScript types for all new shapes | ✓ |
| `InfoTooltip` reusable component | ✓ |
| `SETTINGS_HELP` constant with 17 entries | ✓ |
| `FieldRow` accepts `info` prop | ✓ |
| LLM config list + editor in ConfigPage | ✓ |
| Sanity check button + result card in UI | ✓ |
| Clone button + inline name input | ✓ |
| Save/Delete with confirmation + active guard | ✓ |
| A/B mode toggle in TestRunPage | ✓ |
| Dual config dropdowns, side-by-side results | ✓ |
| ComparisonSummary bar | ✓ |
| `tsc --noEmit` → 0 errors | ✓ |
| `npm run build` → 0 errors | ✓ |
| 44 new Python tests pass | ✓ |
| No regressions in existing tests | ✓ |
