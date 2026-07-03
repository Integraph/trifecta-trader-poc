# TRI-70 — Step-0 Benchmark Report (local models on the v0.3.0 engine)

**Stage:** DEVELOP (in progress) → QA (Codex) → UAT → Arbiter · **Paper/`--dry-run` only.**
**Gate:** production `quality_scorer` ≥ **8.0 / 10** (local only). Cloud (Opus 4.8 / Haiku) = benchmark ceiling, `test-only`, never shipped.
**Work order:** `docs/TASK_TRI-70_BENCHMARK.md` (v3, 8 steps / 8 exit criteria).

---

## ⏱️ RUNNING STATUS (updated live — check here first)

**Last updated:** 2026-07-02 (DEVELOP continuation session)

| Step | What | State |
|------|------|-------|
| 1 | Structured extraction + run-ids wired into results | ✅ code done, tests green (50) — committing |
| 2 | Tool-calling gate (Q4_K_M) | ⏳ pending model pulls |
| 3 | Pricing triad (local $0, opus-4-8 row, normalize, Haiku) | ✅ done, verified, committed |
| 4 | Cache-off (`--no-cache`) verification | ✅ verified end-to-end, both entrypoints |
| 5 | `measure_ollama_speed.py` candidate-driven | ✅ done; old results marked superseded |
| 6 | Staged benchmark (N=3 screen @1 ticker → N=5 finalists) | ⏸️ not started |
| 7–8 | Cloud-reference yardstick + report + recommendation | ⏸️ not started |

**Model pulls (background, healthy — do NOT interrupt):** Wave-1 pull script running from setup session.
Done so far: `qwen3.6:27b`, `qwen3-coder:30b`, `deepseek-r1:8b`, `deepseek-r1:14b`. In progress: `qwen3.6:35b`.
Queue: gpt-oss:20b, deepseek-r1:32b, gemma-4 (tag TBD), qwen3.7 (expected FAIL), llama3.3:70b, deepseek-r1:70b, gpt-oss:120b.
Already present: qwen3.5:9b/27b/35b-a3b, qwen2.5:14b/32b, mistral-small:22b, llama3.1:8b, mistral:7b.

### 🚩 Checkpoint flags (surfaced, not blocking)
_None yet._ (Will record here prominently if: no local config clears 8.0 / structured output doesn't engage on Ollama / any material v0.3.0 behavior.)

---

## Step 1 — Structured decision extraction + method + run-ids

**Status: DONE (code + tests green).**

- `src/signal_processing.py`: `extract_decision_detailed(text) -> {decision, rating_5, method}` with
  `method ∈ {rendered_structured_parse, regex, unknown}`. `rendered_structured_parse` is detected by the exact
  v0.3.0 `render_pm_decision()` template fingerprint (line-anchored `**Rating**: <5-level>` + `**Executive Summary**:` /
  `**Investment Thesis**:` headers, fences stripped first) — a zero-mod deterministic inverse of the vendor render that
  recovers the typed rating. Free-text labels → `regex`; nothing → `unknown` (loud; investigated, never counted HOLD).
  `extract_decision()` kept as a back-compat wrapper.
- `src/run_analysis.py`: result JSON now carries **`pm_rating_5`** (raw 5-level, TRI-74), **`decision_extraction_method`**,
  and **`run_id`** (TRI-79). `--run-id` CLI flag + `TRIFECTA_RUN_ID` env; `TRIFECTA_RUN_ID_FILES=1` also writes a
  run-id-suffixed result copy so benchmark repeats never clobber. Batch repeats controllable via `TRIFECTA_RUN_ID` env.
- Tests: `tests/test_signal_processing.py::TestExtractDecisionDetailed` (8 tests) + full file = **50 passed**.
- **Whether structured output actually engages on Ollama** is measured in Step 6 (the `decision_extraction_method`
  breakdown per config). Session learning from TRI-66 smokes: the render engaged on the cloud path
  (`**Rating**: Overweight` + `**Executive Summary**:`) and fell back to free text on local qwen2.5 — to be quantified.

---

## Step 2 — Tool-calling gate
_Pending pulls._

## Step 3 — Pricing triad

**Status: DONE (verified).** All in `src/hybrid_graph.py`.

| Fix | Before | After | Verified |
|-----|--------|-------|----------|
| (a) Local/Ollama → $0 (kill Sonnet fallback) | `.get(key, {3.00, 15.00})` | `.get(key, {0.0, 0.0})` provider-aware | `qwen3-coder:30b` @1M/1M → **$0.0** |
| (b) Add `claude-opus-4-8` row | absent | `{input: 5.0, output: 25.0}` | in `MODEL_PRICING` ✓ |
| (c) Refresh Haiku | `$0.80/$4.00` | `$1.00/$5.00` | `{input:1.0, output:5.0}` ✓ |
| (d) `_normalize_model` "opus" | → retired `opus-4-5` ($15/$75) | → `claude-opus-4-8` | `normalize('…opus…')='claude-opus-4-8'` ✓ |

Retired `claude-opus-4-5-20251101` row kept for reference but is now unreachable via the normalizer.
`tests/test_cost_optimization.py` = **35 passed** (tests read the table dynamically, so the refreshed numbers are consistent).
**Cost policy:** wall-time/ticker reported for all configs; **$ column only for the cloud references** (opus-4-8 / Haiku). Local = $0.

## Step 4 — Cache-off

**Status: DONE (verified — no code change needed, already correctly wired).**

`--no-cache` flows end-to-end in **both** entrypoints:
- `run_analysis.py`: `--no-cache` → `use_cache=not args.no_cache` → `HybridTradingGraph(use_cache=use_cache)`.
- `run_batch.py`: `--no-cache` → `use_cache=not args.no_cache` → `run_analysis(use_cache=use_cache)`.
- `hybrid_graph.py`: `self._cache = DataCache(...) if use_cache else None`; when `None`, the
  `make_cached_analyst` wrapping at line ~277 is skipped (`if self.cache and …` is False) so analysts make
  real LLM calls, and `cache_stats = self._cache.stats() if self._cache else {}`.

→ Benchmark runs will pass **`--no-cache`** so repeat-run stability measures the model, not the DataCache.

## Step 5 — Speed harness

**Status: DONE.** `scripts/measure_ollama_speed.py` is now candidate-driven (was hardcoded to the
qwen2.5/3.5 list at line 18):
- Default = the TRI-70 candidate set (deep / tool / quick slots); not-installed tags are skipped cleanly
  (safe to run mid-pull).
- `--models <tags…>` for an explicit list; `--all` auto-discovers every installed model from `/api/tags`.
- `--output` (default `results/tri70_speed_benchmark.json`), superseding `task_011_speed_benchmark.json`.
- **Prior results marked SUPERSEDED (Tasks 005 / 010 / 011):** those files measured the qwen2.5/3.5
  generation on the **pre-v0.3.0 engine** — `task_011_speed_benchmark.json` (speed) and `task_010_bench_*.txt`
  (quality/wall-time) are superseded by this report's table + `results/tri70_speed_benchmark.json`. Do not cite
  their numbers as current. (`results/` is gitignored, so a local `results/SUPERSEDED_BY_TRI-70.md` marker was also
  written; this report is the tracked record.)
- Actual speed numbers get collected in Step 6 once pulls finish (not run now, to avoid competing with the
  in-flight Wave-1 pull).

## Step 6 — Staged benchmark runs
_Not started._

## Step 7–8 — Cloud reference + recommendation
_Not started._
