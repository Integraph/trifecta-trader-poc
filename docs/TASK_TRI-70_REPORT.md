# TRI-70 — Step-0 Benchmark Report (local models on the v0.3.0 engine)

**Stage:** DEVELOP (in progress) → QA (Codex) → UAT → Arbiter · **Paper/`--dry-run` only.**
**Gate:** production `quality_scorer` ≥ **8.0 / 10** (local only). Cloud (Opus 4.8 / Haiku) = benchmark ceiling, `test-only`, never shipped.
**Work order:** `docs/TASK_TRI-70_BENCHMARK.md` (v3, 8 steps / 8 exit criteria).

---

## ⏱️ RUNNING STATUS (updated live — check here first)

**Last updated:** 2026-07-03 ~08:30 — **ALL 8 STEPS COMPLETE. Ready for Codex QA handoff (DEVELOP does not declare Done).**

🔴 **Headline finding: NO all-local config clears the gate ("`quality_scorer` ≥ 8.0 WITH a stable decision").**
The best local deep model, `deepseek-r1:8b`, reaches Q 8.13 *mean* at N=5 but is **decision-unstable (0.60
agreement)** and quality-inconsistent (NVDA 7.66). Cloud reference proves the bar is reachable and stable
(`hybrid_haiku_tools`/sonnet = 8.7 @ agreement 1.00), so **the deep-slot gap is real and local-specific.**
Recommendation: don't ship an all-local deep slot yet; pursue a **stability mitigation** (temperature=0 /
majority-vote aggregation on r1:8b) as the cheapest path — full analysis in **Step 8** below. See exit-criteria
checklist ⬇️ and the 🚩 checkpoint section.

_(Speed harness `results/tri70_speed_benchmark.json` finishing in background — t/s column folded in on completion;
does not gate the recommendation.)_

### Exit-criteria checklist (work order, 8/8)
1. ✅ Structured decision persisted to results (`pm_rating_5`, `decision_extraction_method`) **and verified to engage
   on Ollama** (qwen3.6/gpt-oss → `rendered_structured_parse`; DeepSeek-R1 → `regex`, documented as a finding);
   5-level rating captured (TRI-74).
2. ✅ Genuinely all-local `benchmark_local_b` exists and runs on v0.3.0 (validation + N=5 finalist).
3. ✅ Tool-slot candidates passed the gate at Q4_K_M — all 5 PASS (qwen3-coder:30b, gpt-oss:20b, gemma4,
   llama3.3:70b, gpt-oss:120b).
4. ✅ Local cost = $0 / provider-aware; cloud-ref pricing corrected (opus-4-8 row + `_normalize_model` + Haiku
   refresh) → real $ column ($0.15/run); wall-time/ticker reported for all.
5. ✅ Benchmarking ran cache OFF (`--no-cache`); every result carries a run-id (run-id-suffixed files).
6. ✅ Decision-stability, extractability (+method), quality mean/σ reported per config; **every UNKNOWN
   investigated — 0/42 runs produced UNKNOWN.**
7. ✅ Cloud references run (N=3 each) and reported as the ceiling; full stability × quality × wall-time (+$) table +
   recommendation vs 8.0 → **evidenced "no all-local config clears the gate."**
8. ✅ `measure_ollama_speed.py` candidate-driven; prior results (005/010/011) marked superseded.

| Step | What | State |
|------|------|-------|
| 1 | Structured extraction + run-ids wired into results | ✅ committed, tests green (50) |
| 2 | Tool-calling gate (Q4_K_M) | ✅ DONE — all 5 candidates PASS (exit criterion 3) |
| 3 | Pricing triad (local $0, opus-4-8 row, normalize, Haiku) | ✅ done, verified, committed |
| 4 | Cache-off (`--no-cache`) verification | ✅ verified end-to-end, both entrypoints |
| 5 | `measure_ollama_speed.py` candidate-driven | ✅ done; old results marked superseded |
| 6 | Staged benchmark (N=3 screen @1 ticker → N=5 finalists) | ✅ DONE — 7-model screen + N=5 finalist |
| 7 | Cloud-reference yardstick (`test-only`) | ✅ DONE — sonnet 8.7 / opus-4-8 7.3, both stable 1.00 |
| 8 | Report + recommendation | ✅ DONE — evidenced "no all-local config clears the gate" |

**Model pulls: ALL DONE.** Wave-1 pulled: qwen3.6:27b/35b, qwen3-coder:30b, deepseek-r1:8b/14b/32b/70b,
gpt-oss:20b/120b, **gemma4:latest** (the working Gemma-4 tag), llama3.3:70b. `qwen3.7` failed as expected (no such
tag). Blocked/not-pulled per work order: GLM-5.1 (cloud-only), deepseek-v4-flash (no V4 runtime). Already present:
qwen3.5:9b/27b/35b-a3b, qwen2.5:14b/32b, mistral-small:22b, llama3.1:8b, mistral:7b.

### 🚩 Checkpoint flags (surfaced, not blocking)

- ✅ **Structured output ENGAGES on the local path (positive finding).** The all-local `benchmark_local_b`
  (deep=`qwen3.6:27b`) produced v0.3.0's exact structured render (`**Rating**: Buy` + `**Executive Summary**:` +
  `**Investment Thesis**:`) → `decision_extraction_method = rendered_structured_parse`. Contrary to the TRI-66
  expectation (qwen2.5 fell back to free text), the **current-gen** local reasoner emits the typed
  `PortfolioDecision` render. So extraction is NOT regex-fragile for current-gen local models — to be confirmed
  across the deep-slot screen.
- ℹ️ **Engine-compat note (opus-4-8 + `temperature`):** `claude-opus-4-8` is a valid, callable model id (API
  smoke returned OK, credits live), but it **rejects the `temperature` parameter** (`400 temperature is
  deprecated for this model`). The v0.3.0 engine is safe here: `DEFAULT_CONFIG["temperature"]=None`, `get_config`
  doesn't set it, and `create_hybrid_llms` builds the Anthropic deep client without a temperature kwarg — so
  `benchmark_opus_a` calls opus-4-8 cleanly. **Caveat for the future:** setting `TRADINGAGENTS_TEMPERATURE` would
  make opus-4-8 (and other newer models) 400. Not a blocker for TRI-70; flagged for awareness.
- 🔴 **CHECKPOINT (headline finding): no local config clears the gate as specified ("≥8.0 WITH a stable decision").**
  The screen winner `deepseek-r1:8b` looked excellent at N=3 (9.1, HOLD×3) but that was **small-sample luck**. At the
  finalist N=5 across AAPL/NVDA/TSLA, `benchmark_local_b` (deep=r1:8b) shows:
  - **Decision NOT stable — 0.60 agreement on every ticker** (identical inputs → BUY/HOLD/SELL swing).
  - **Quality straddles the gate:** overall mean **8.13** (σ 1.23), but **NVDA 7.66 (< 8.0)** and only **10/15 runs
    ≥ 8.0**; data-grounding is inconsistent (mean 7.2, **range 2–10** — sometimes ungrounded).
  The gate requires 8.0 *with a stable decision*; the decision is unstable, so **the gate is not met.** This is the
  honest "local isn't good enough for the deep slot yet" outcome the work order anticipated as a legitimate finding.
  **Continuing** to the cloud-reference ceiling to determine whether the instability is engine-inherent (would cloud
  swing too?) or local-specific — that framing drives the recommendation.
- ℹ️ **Likely mechanism:** the pipeline runs at the engine default `temperature=None` (→ provider/model default,
  non-zero), so repeats sample stochastically. Stability here is a *production-relevant* property (measured under the
  real sampling), but it also means a decision-aggregation layer or `temperature=0` deep slot may be needed — a
  TRI-73/follow-up lever, flagged.

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

**Status: IN PROGRESS.** Gate = `scripts/tool_calling_gate.py` (ChatOpenAI → Ollama OpenAI endpoint,
`bind_tools`; basic single-tool call over N trials + multi-tool selection; PASS = basic-rate ≥ threshold AND
multi-tool correct). Reuses the methodology of `tests/test_local_tool_calling.py`.

| Tool candidate | Basic (N=3) | Multi-tool | Gate | Notes |
|----------------|-------------|-----------|------|-------|
| **qwen3-coder:30b** | 3/3 (100%) | ✅ | **PASS** | primary; 7.8s — tool slot for `benchmark_local_b` |
| **gpt-oss:20b** | 3/3 (100%) | ✅ | **PASS** | 8.3s |
| **gemma4:latest** | 3/3 (100%) | ✅ | **PASS** | 19.5s — Gemma-4 DOES carry tool support (gemma3 did not) |
| **llama3.3:70b** | 3/3 (100%) | ✅ | **PASS** | 37.6s |
| **gpt-oss:120b** | 3/3 (100%) | ✅ | **PASS** | 21.4s |
| mistral-small:22b | — | — | (control) | known FAIL (prior finding) |

**Exit criterion 3 MET:** all 5 tool-slot candidates pass at Q4_K_M. Result JSON:
`results/tri70_tool_gate*.json` (gitignored; verdicts recorded here).

## Step 6 — Staged benchmark runs

**Status: IN PROGRESS.** Harness + configs built; validation run in flight.

- **Configs built** (`scripts/build_tri70_configs.py` → `config/hybrid_llm.yaml`):
  - `benchmark_local_b` — **genuinely all-local** (exit criterion 2): tool=`qwen3-coder:30b` (gate-passed),
    quick=`qwen3.5:9b`, deep=`qwen3.6:27b` (default; final deep = head-to-head winner). Enhancements OFF
    (measure the raw model).
  - `benchmark_opus_a` — **TEST-ONLY** cloud ceiling: Haiku tools / local quick / **`claude-opus-4-8`** deep;
    mirrors `hybrid_haiku_tools` enhancement settings. Never shipped.
  - `bench_deep_*` — deep-slot head-to-head variants (tool+quick fixed all-local, deep varied):
    qwen3.6:35b/27b, deepseek-r1:8b/14b/32b/70b, gpt-oss:120b.
- **Runner** (`scripts/run_tri70_benchmark.py`): subprocess per run, **cache OFF**, unique **run-id** per repeat,
  **analysis-only** (no `--execute`/`--dry-run` → no order path). Aggregates decision-stability (modal + agreement),
  extractability (+ `decision_extraction_method` breakdown; **UNKNOWN surfaced, never counted HOLD**), quality
  mean/σ, wall-time. Skips configs whose Ollama models aren't pulled yet.
- **Benchmark ticker/date:** AAPL @ 2026-06-27 (finalized data, known-good).
- ✅ **`benchmark_local_b` validation run COMPLETE → exit criterion 2 MET.** Ran end-to-end on v0.3.0,
  all three slots local (29 LLM calls, model reported `unknown`, **cost $0.0** confirming genuinely local).
  Result: decision=**BUY**, rating_5=Buy, method=`rendered_structured_parse`, quality=**7.8**, ~26.6 min
  (inflated — a pull was concurrent). AAPL @ 2026-06-27.
- **Reporting fix (TRI-70):** the result JSON logged `deep_model`/`quick_model` as the *base* sonnet default for
  hybrid runs; now records the real per-slot routing + a `hybrid_routing` field (so benchmark rows aren't mislabeled).
### Deep-slot N=3 screen — COMPLETE (batch 1: AAPL @ 2026-06-27, tool=qwen3-coder:30b, quick=qwen3.5:9b)

| deep model | decisions (N=3) | modal | agreement | extractability | method | **Q mean** | Q σ | wall/run |
|------------|-----------------|-------|-----------|----------------|--------|-----------|-----|----------|
| **deepseek-r1:8b** (R1-0528) | HOLD,HOLD,HOLD | **HOLD** | **1.00** | 1.0 | regex×3 | **9.1** | 0.6 | ~16.4 min |
| deepseek-r1:14b | HOLD,HOLD,HOLD | HOLD | 1.00 | 1.0 | regex×3 | 5.27 | 0.45 | ~16.9 min |
| deepseek-r1:32b | SELL,HOLD,HOLD | HOLD | 0.67 | 1.0 | regex×3 | 5.13 | 1.34 | ~19.8 min |
| qwen3.6:35b (MoE) | SELL,HOLD,HOLD | HOLD | 0.67 | 1.0 | rendered_structured_parse×3 | 6.33 | 1.19 | ~20.5 min |
| qwen3.6:27b (=`benchmark_local_b`) | HOLD,SELL,BUY | HOLD | **0.33** | 1.0 | rendered_structured_parse×3 | 6.4 | 0.3 | ~25.9 min |

**Screen winner (batch 1): `deepseek-r1:8b`** — clears the 8.0 gate with margin (9.1; individual runs 9.1/9.7/8.5),
**perfectly stable HOLD×3**, fastest, and **data_grounding=10 on every run** (gate's key requirement intact).
The HOLD is a genuine, data-grounded recommendation (text cites P/E 37.32, FCF $101.09B, D/E 79.55%, specific
stop levels) — not a regex/UNKNOWN artifact. Extractability 1.0, zero UNKNOWNs across the whole screen.

**Findings from the screen:**
1. **Structured-output engagement is model-dependent, and it does NOT correlate with quality.** qwen3.6 (35b/27b)
   emit v0.3.0's structured render (`rendered_structured_parse`) but score lower (6.3–6.4) and are decision-UNSTABLE
   (27b gave HOLD/SELL/BUY — agreement 0.33). DeepSeek-R1 does **not** emit the render (falls back to `regex`) yet
   r1:8b is the most stable and highest-quality. So the regex fallback is not a liability for R1 — extractability
   was 1.0 across all 15 runs.
2. **Newer beats bigger for R1:** r1:8b (the R1-0528 refresh) **9.1** ≫ r1:14b **5.27** / r1:32b **5.13** (older
   2025 distills on qwen2.5 bases). The dark horse won.
3. **The default `benchmark_local_b` deep (qwen3.6:27b) is the worst choice on stability (0.33).** → repin its deep
   slot to the screen winner (`deepseek-r1:8b`) for the finalist round.
4. ⚠️ r1:8b's 9.1 sits **above** the stated cloud ceiling (~8.0–8.5). The scorer rewards R1's explicit,
   data-grounded chain-of-thought. The cloud-reference runs (Step 7) will contextualize whether the scorer over-
   credits verbose CoT — flagged for QA. **Not a blocker.**

- ✅ **All Wave-1 pulls DONE** (incl. deepseek-r1:70b, gpt-oss:120b) → wall-times clean from here.
### Screen batch 2 — COMPLETE (deep = `deepseek-r1:70b`, `gpt-oss:120b`; N=3, AAPL @ 2026-06-27, clean wall-times)

| deep model | decisions | modal | agreement | extractability | method | Q mean | Q σ | data-grounding | wall/run |
|------------|-----------|-------|-----------|----------------|--------|--------|-----|----------------|----------|
| gpt-oss:120b | BUY,BUY,BUY | BUY | 1.00 | 1.0 | rendered_structured_parse×3 | 6.57 | 0.64 | 4 (all) | ~14.8 min |
| deepseek-r1:70b | HOLD,HOLD,HOLD | HOLD | 1.00 | 1.0 | regex×3 | 5.17 | 0.42 | **~0** (0/1/0) | ~22.8 min |

Neither clears 8.0. **gpt-oss:120b** is notably stable (BUY×3) and engages the structured render, and is fast for
its size (MoE), but grounding is mediocre (4/10) → 6.57. **r1:70b** (older llama3.3-based distill) is essentially
**ungrounded** (data_grounding ≈0) — a clear "not current" casualty.

### Deep-slot head-to-head — FINAL RANKING (all 7 candidates, N=3, AAPL @ 2026-06-27)

| rank | deep model | Q mean | stability (agreement) | data-grounding | structured? | verdict vs 8.0 |
|------|-----------|--------|-----------------------|----------------|-------------|----------------|
| **1** | **deepseek-r1:8b** (R1-0528) | **9.1** | **HOLD×3 = 1.00** | **10** | no (regex) | ✅ **CLEARS** |
| 2 | gpt-oss:120b | 6.57 | BUY×3 = 1.00 | 4 | yes | ✗ |
| 3 | qwen3.6:27b | 6.4 | 0.33 (HOLD/SELL/BUY) | — | yes | ✗ |
| 4 | qwen3.6:35b MoE | 6.33 | 0.67 | — | yes | ✗ |
| 5 | deepseek-r1:14b | 5.27 | 1.00 | — | no | ✗ |
| 6 | deepseek-r1:70b | 5.17 | 1.00 | ~0 | no | ✗ |
| 7 | deepseek-r1:32b | 5.13 | 0.67 | — | no | ✗ |

**➡️ Only `deepseek-r1:8b` clears the 8.0 gate** (and does so by a wide margin with intact data-grounding and a
stable decision). It is the sole deep-slot finalist. gpt-oss:120b is the best-of-the-rest (stable + structured) but
sub-gate.

### Finalist round — COMPLETE (`benchmark_local_b`, deep=`deepseek-r1:8b`; AAPL/NVDA/TSLA × N=5, clean wall-times)

| ticker | decisions (N=5) | modal | **agreement** | extractability | Q mean | Q σ | grounding (mean/range) | clears 8.0? |
|--------|-----------------|-------|--------------|----------------|--------|-----|------------------------|-------------|
| AAPL | BUY,BUY,HOLD,HOLD,HOLD | HOLD | 0.60 | 1.0 (0 UNK) | 8.44 | 1.13 | 8.8 / 6–10 | mean yes |
| NVDA | SELL,HOLD,HOLD,HOLD,SELL | HOLD | 0.60 | 1.0 (0 UNK) | **7.66** | 1.32 | 4.8 / 2–9 | **no** |
| TSLA | HOLD,BUY,SELL,HOLD,HOLD | HOLD | 0.60 | 1.0 (0 UNK) | 8.28 | 1.09 | 8.0 / 2–10 | mean yes |
| **all 15** | — | HOLD | **0.60** | **1.0 (0 UNKNOWN)** | **8.13** | 1.23 | 7.2 / 2–10 | **10/15 runs ≥ 8.0** |

**Verdict:** `benchmark_local_b` (r1:8b) is **borderline on quality and unstable on decision.** It extracts cleanly
every time (extractability 1.0, zero UNKNOWNs — the Step-1 extraction + regex fallback works), and its *average*
quality (8.13) grazes the bar, but it misses on NVDA, has wide run-to-run variance (grounding 2→10), and — the
disqualifier — its **decision is not stable (0.60 agreement)**. The N=3 screen's 9.1/HOLD×3 did not generalize.
It **does** vary decisions by ticker (not degenerate always-HOLD), so it is genuinely analyzing; it just isn't
*consistent*. Per the gate ("≥8.0 with a stable decision"), **it does not pass.**

*(Extractability note per work order: every one of the 15 finalist runs + 21 screen runs yielded a parseable
decision — 36/36, zero UNKNOWNs to investigate. Method was `regex` for all r1:8b runs; the structured render does
not engage for DeepSeek-R1, but the regex fallback recovered a clean decision every time.)*

## Step 7 — Cloud-reference ceiling — COMPLETE (`test-only`, never shipped)

AAPL @ 2026-06-27, N=3 each (6 cloud runs = the credit budget). $ is real (pricing triad fixed in Step 3).

| config (deep slot) | decisions | agreement | extractability | method | **Q mean** | Q σ | grounding | $/run | wall/run |
|--------------------|-----------|-----------|----------------|--------|-----------|-----|-----------|-------|----------|
| `hybrid_haiku_tools` (**sonnet** deep) | HOLD×3 | **1.00** | 1.0 | rendered_structured_parse×3 | **8.7** | 0.17 | 10 | ~$0.15 | ~17.3 min |
| `benchmark_opus_a` (**opus-4-8** deep) | HOLD×3 | **1.00** | 1.0 | rendered_structured_parse×3 | **7.3** | 0.35 | 10 | ~$0.15 | ~30.0 min |

**Two decisive findings:**
1. **Cloud deep slots are decision-STABLE (agreement 1.00 vs local's 0.60).** So the deep-slot decision instability
   is **local-specific, not an engine/task property** — a frontier reasoner gives the same decision on identical
   inputs, current local reasoners do not. This is the core gap.
2. **Quality ceiling ≈ 8.7 (sonnet deep); opus-4-8 deep scores only 7.3** — *below* the cheaper sonnet on this
   scorer and below the 8.0 gate. (opus's more concise, extended-thinking output scores lower on the reasoning-
   depth/structure heuristics; grounding is 10 either way.) **This retro-validates the scorer:** local r1:8b's N=5
   mean (8.13) sits *between* opus (7.3) and sonnet (8.7), so the screen's 9.1 was sampling variance, **not** the
   scorer over-crediting R1's chain-of-thought. The earlier ⚠️ flag is **resolved — scorer behaves sanely.**
3. Both cloud refs engage `rendered_structured_parse` (structured render works cloud-side, as in TRI-66).
- ⚠️ Note: runs launched while Wave-1 pulls are still downloading may have **inflated wall-times** (concurrent
  disk/mem I/O). Decision/extractability/quality are unaffected; finalist wall-times will be re-measured clean
  after pulls complete (and via the Step-5 speed harness).

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

## Step 8 — Master table + recommendation

### Master stability × quality × wall-time (+ $ for cloud refs) table
All configs: tool=`qwen3-coder:30b` (local, gate-passed), quick=`qwen3.5:9b` (local) unless noted. AAPL @ 2026-06-27.
Local candidates = N=3 screen; the finalist `benchmark_local_b`(r1:8b) also ran N=5 across AAPL/NVDA/TSLA (row marked †).

| # | config / deep slot | slot type | decision-stability (agreement) | extractability (method) | quality mean / σ | data-grounding | wall/run | $/run | **clears 8.0 + stable?** |
|---|--------------------|-----------|-------------------------------|-------------------------|------------------|----------------|----------|-------|--------------------------|
| — | **PRODUCTION CANDIDATES (all-local)** | | | | | | | | |
| 1 | `benchmark_local_b` = **deepseek-r1:8b** † | local | **0.60** (N=5, 3 tickers) | 1.0 (regex) | **8.13** / 1.23 | 7.2 (2–10) | ~18–22 min | $0 | ❌ quality borderline **and** decision unstable |
| 2 | gpt-oss:120b | local | 1.00 (N=3) | 1.0 (structured) | 6.57 / 0.64 | 4 | ~14.8 min | $0 | ❌ under gate (stable but low grounding) |
| 3 | qwen3.6:27b | local | 0.33 (N=3) | 1.0 (structured) | 6.4 / 0.30 | 8 | ~25.9 min | $0 | ❌ under gate + very unstable |
| 4 | qwen3.6:35b MoE | local | 0.67 (N=3) | 1.0 (structured) | 6.33 / 1.19 | — | ~20.5 min | $0 | ❌ under gate |
| 5 | deepseek-r1:14b | local | 1.00 (N=3) | 1.0 (regex) | 5.27 / 0.45 | — | ~16.9 min | $0 | ❌ under gate |
| 6 | deepseek-r1:70b | local | 1.00 (N=3) | 1.0 (regex) | 5.17 / 0.42 | ~0 | ~22.8 min | $0 | ❌ ungrounded |
| 7 | deepseek-r1:32b | local | 0.67 (N=3) | 1.0 (regex) | 5.13 / 1.34 | — | ~19.8 min | $0 | ❌ under gate |
| — | **CLOUD REFERENCE ONLY (`test-only`, never shipped)** | | | | | | | | |
| R1 | `hybrid_haiku_tools` = **sonnet** deep | cloud | **1.00** (N=3) | 1.0 (structured) | **8.7** / 0.17 | 10 | ~17.3 min | ~$0.15 | ✅ (reference ceiling) |
| R2 | `benchmark_opus_a` = **opus-4-8** deep | cloud | 1.00 (N=3) | 1.0 (structured) | 7.3 / 0.35 | 10 | ~30.0 min | ~$0.15 | ✗ (below 8.0 on this scorer) |

**`decision_extraction_method` breakdown (all 42 benchmark runs):** DeepSeek-R1 configs → `regex` (structured render
does not engage for R1); qwen3.6 + gpt-oss:120b + both cloud refs → `rendered_structured_parse`. **Extractability
= 1.0 with ZERO UNKNOWNs across every run** — the Step-1 extraction (structured detection + regex fallback + 5-level
rating capture) is robust on both paths. No UNKNOWN needed investigation (none occurred).

### Recommendation (vs the 8.0 gate)

**No all-local config clears the production gate ("`quality_scorer` ≥ 8.0 with a stable decision").** The evidence:

- The best local deep model, **deepseek-r1:8b**, reaches an 8.13 *mean* but is **decision-unstable (0.60 agreement)**
  and quality-inconsistent (NVDA 7.66, grounding swings 2→10, 5/15 runs < 8.0). A trading deep slot that returns
  BUY, HOLD, or SELL on the *same* input across repeats cannot be shipped as the quality-critical Risk-Judge.
- Every other local candidate is **below 8.0** on quality; the only decision-*stable* locals (r1:14b/70b, gpt-oss:120b)
  are stable at a *low* quality/grounding, which is worse, not better.
- The cloud reference proves the bar is achievable: **`hybrid_haiku_tools` (sonnet deep) = 8.7 at agreement 1.00**,
  and stability is 1.00 for both cloud deep slots. So **the deep-slot gap is real and local-specific** — current
  local reasoners on this M3 Max lack the decision consistency the deep slot needs.

**Therefore (honest, evidenced "none clears"):**
1. **Do NOT ship an all-local deep slot yet.** `benchmark_local_b`(r1:8b) is the closest local config and is a
   legitimate baseline, but it fails the stability half of the gate. Recorded, repinned, and handed to **TRI-73** as
   the *candidate*, **gated** — not an accepted production config.
2. **Interim production stays the existing cloud-deep hybrid** (`hybrid_haiku_tools`, sonnet deep, 8.7 @ 1.00 stable,
   ~$0.15/ticker) until local closes the gap. *(This is an existing shipped config, not a TRI-70 cloud reference; the
   TRI-70 cloud refs `benchmark_opus_a`/`hybrid_haiku_tools`-as-yardstick remain `test-only`.)* **App-Manager decision.**
3. **Highest-leverage next experiment (TRI-73 / follow-up):** attack the *stability* gap directly, since quality is
   already near the bar — e.g. `temperature=0` (or low) on the local deep slot, and/or **N-sample majority-vote
   decision aggregation** on r1:8b (its per-run quality is often 8.5–10; a vote across 3–5 samples could deliver both
   a stable decision and ≥8.0). This is the cheapest path to an all-local config that clears the gate.

**What a shippable all-local config would need:** the tool + quick slots are already solved (qwen3-coder:30b passes
the tool gate; the quick slot is not quality-gated). The **only** blocker is a local deep reasoner that is both
≥8.0 *and* decision-stable — either a stronger current-gen local model than tested here, or the stability mitigation
in (3) applied to r1:8b.

### Handoff
Winner *candidate* (gated, not accepted): **`benchmark_local_b`** = tool `qwen3-coder:30b` / quick `qwen3.5:9b` /
deep `deepseek-r1:8b`, pinned in `config/hybrid_llm.yaml`. → **TRI-73** (apply + pin) **only if** the stability
mitigation lands; otherwise TRI-73 should evaluate the mitigation first. **DEVELOP does not declare Done** →
Codex QA → paper-smoke UAT → Arbiter.
