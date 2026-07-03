# TRI-70 — DEVELOP Work Order: Step-0 benchmark (local models on the v0.3.0 engine) · v3

**Issue:** TRI-70 (High) · **Stage:** DEVELOP → QA → UAT → Arbiter · **Depends on:** TRI-66 (Done) · **Feeds:** TRI-69
**Owner:** Engine App Manager · **Report to:** `docs/TASK_TRI-70_REPORT.md`. **Paper/`--dry-run` only.**

> **v3 (after 2nd DEVELOP + QA review, verified against the tree):** removed the stale "OPEN gate" language (gate is resolved at 8.0); fixed the **cloud-pricing triad** (the `_normalize_model` "opus" → retired `opus-4-5` bug means adding a row isn't enough — normalizer + Haiku row must be fixed too); settled the Qwen tag; **GLM-5.1 marked cloud-only/blocked** (no local Ollama tag); cut the Qwen-3.7 rumor; **Gemma → current Gemma-4 27B, tool-gate-required** (gemma3 lacks the `tools` tag); added `decision_extraction_method` as a scored field; corrected the **disk (~250–270 GB) and time (2–4 days)** budgets; reorganized candidates by slot.

---

## Quality gate = **8.0 / 10** (production, local); cloud = benchmark reference only (Jeff, 2026-07-02)
The old **≥9.0** came from the *pre-upgrade* engine (Task 010: 9.1–9.4). On **v0.3.0** the observed cloud ceiling is only ~**8.0–8.5**, so 9.0 fails everything by construction. **Production pass mark: `quality_scorer` ≥ 8.0/10** (data-grounding not degraded; decision stable). A local config clearing 8.0 with a stable decision is "good enough to ship."
**Cloud = MEASUREMENT yardstick only (Jeff):** run the cloud reference configs — with **Opus 4.8** as the chosen expensive reference — to (a) confirm the scorer gives right results and (b) show how close local gets. **Cloud is NEVER a production/shipped model — too expensive; benchmark-only, flagged `test-only`.** *(Supersedes operating-prompt §6.7's old-engine 9.0.)*

## Objective
On the upgraded v0.3.0 engine, benchmark **local** candidates across the 3 slots; produce a **decision-stability × quality × wall-time (+ $ for cloud refs)** table; **recommend a production config** — and honestly answer whether a cheap, all-local config clears 8.0 with a stable decision, or the quality-critical deep slot needs more than local gives (the cloud reference bounds that).

## Model policy (§6.7)
**Outcome over brand** (Jeff: local models don't have to be Chinese — just good). Production = a **local** model; **quality gate = ≥8.0** (data-grounding intact; deep **Risk-Judge** slot is quality-critical); **speed measured, not gated.** Organize by **slot + cost**: cheapest-viable per slot first, escalate only on a miss; then **pin** the winner (TRI-65).

## Candidate models — by slot, cheapest-viable first (brand-agnostic; verify every tag at pull time)
All local; must fit **M3 Max / 128 GB**. **Tool-slot quant floor `Q4_K_M`** (Q3/Q2 degrade tool-calling first). Reliable local tool-callers: Qwen, DeepSeek, Llama.

**Deep / Risk-Judge slot (quality-critical — where local has failed):**
- **Qwen 3.6** — `qwen3.6:35b` (the MoE, ~3B active — fastest/best bang) and dense `qwen3.6:27b`. *(Current-gen; the primary hope.)*
- **gpt-oss** `gpt-oss:120b` (~65 GB) — strong reasoning, structured output.
- **DeepSeek-R1 distills** `deepseek-r1:14b/32b/70b` — **NB: 2025-era distills on qwen2.5-14b/32b & llama3.3-70b bases** (the same qwen2.5 gen whose staleness motivated this) — worth testing (R1 CoT may still lift reasoning), **not "current."** Plus `deepseek-r1:8b` (the one distill with the newer **R1-0528** refresh, ~5.2 GB) as a dark horse.

**Tool slot (needs reliable function-calling — every current config clouds this; all must pass the tool-calling gate):**
- `qwen3-coder:30b` (~18–19 GB, tool-built) — primary.
- `gpt-oss:20b`/`:120b` (tool use + structured output).
- `llama3.3:70b` (~43 GB; carries the Ollama `tools` tag; ~97% well-formed calls).
- **Gemma-4 27B** — verify the current Ollama tag at pull (**not** `gemma3:27b`, the 2025 predecessor, which lacks the `tools` tag). Include **only if it passes the tool gate.**

**Quick slot (fast, cheap):** `qwen3.6:35b` (MoE), `gpt-oss:20b`, or the already-pulled `qwen3.5:9b`.

**Escalation tier (only if the cheap tier misses 8.0):** larger sizes of the above.

**Blocked — not locally runnable today (re-check later, do NOT pull):**
- **GLM-5.1** — Ollama exposes only `glm-5.1:cloud` (no local weights); the flagship is ~756B. No confirmed 128 GB-fit local tag — **dropped from local candidates** unless a small local tag verifies.
- **`deepseek-v4-flash`** — real model, fits memory, but **no stable Ollama/llama.cpp support for the V4 architecture** (cloud-only tag).

**Excluded (don't fit 128 GB / not needed):** Kimi K2.6/2.7, GLM-5.2, DeepSeek-V4-Pro, MiniMax M3.

**Cloud reference (BENCHMARK-ONLY — never production, flagged `test-only`):** `benchmark_opus_a` (tools Haiku, quick local, deep **`claude-opus-4-8`** — the chosen expensive reference) + the existing `hybrid_haiku_tools`. Run only as the yardstick.

## Configs to build (TRI-70's scope, gated)
1. **`benchmark_local_b`** — the prize: **genuinely all-local, no cloud in any slot** (tool = `qwen3-coder:30b` or another local model that passes the tool gate; quick = a fast local; deep = the best local reasoner from the head-to-head). Never existed before.
2. **Per-candidate deep-slot variants** — each reasoner (Qwen3.6-MoE, DeepSeek-R1 8b/14b/32b/70b, gpt-oss:120b) measured head-to-head in the Risk-Judge role, tool+quick fixed.
3. **Cloud reference configs (`test-only`):** `benchmark_opus_a` + `hybrid_haiku_tools` — ~3 repeats each as the yardstick. **Must never be selected as the shipped config.**

## Steps
1. **Structured decision extraction + prove it engages + record the method.** v0.3.0's PM makes a typed `PortfolioDecision` via `with_structured_output` but **renders it to a string and stores only that** (`portfolio_manager.py`). So **persist a normalized structured decision field into app state / the result JSON**, keep regex as fallback, and **record a `decision_extraction_method ∈ {structured, rendered_structured_parse, regex, unknown}`** per run — so QA can prove whether Ollama actually used structured output (local models often fall back to free text; if so, the regex still governs). **Also record the raw 5-level rating** (free data for TRI-74).
2. **Pull + tool-calling gate.** Pull confirmed candidates at **Q4_K_M**. Any tool-slot model passes the tool-calling check first; failures are disqualified from the tool slot (mistral-small, and possibly Gemma, fail this).
3. **Cost accounting — fix the pricing triad.** (a) Local/Ollama pricing → **$0 / provider-aware** (neutralize `hybrid_graph.py:114`'s Sonnet fallback). (b) For the cloud references to have a *real* $ column: **add a `claude-opus-4-8` row (`$5/$25`)**, **refresh the Haiku row `$0.80/$4.00` → `$1.00/$5.00`**, **and fix `_normalize_model` so "opus" → `claude-opus-4-8`** (today it maps to the retired `claude-opus-4-5-20251101` at $15/$75 — the added row is never reached without this). Report **wall-time/ticker** for all configs; **$ only for the cloud refs**.
4. **CACHE OFF for benchmarking.** `make_cached_analyst`/`DataCache` inject cached analyst reports with no LLM call → repeat-run stability would measure the cache. Use **`--no-cache`** (`use_cache=False`, present in both entrypoints), or split into frozen-input PM tests + separate no-cache tool-slot tests.
5. **Update the speed harness.** `scripts/measure_ollama_speed.py:18` is hardcoded to the qwen2.5/3.5 list — make it candidate-driven before using it as evidence.
6. **Repeat-run harness — staged.** **Screen deep-slot variants at 1 ticker × N=3** → run **finalists at the small fixed watchlist × N=5** (halves the initial sweep). Result files carry a **run-id** (TRI-79). Per config report: **decision-stability** (agreement across repeats), **extractability** (fraction yielding a parseable decision; **investigate every UNKNOWN** — never count it as HOLD) with the `decision_extraction_method` breakdown, **quality mean/σ**, **wall-time/ticker**.
7. **Decision (≥8.0 gate).** Among configs clearing 8.0 (deep slot especially, data-grounding intact) with a **stable decision**, pick the cheapest/fastest. **Report the cloud reference results alongside as the ceiling.** If no local config clears 8.0, say so plainly with evidence — a real finding, not a failure.
8. **Report + recommend** → `docs/TASK_TRI-70_REPORT.md`; hand the winner to **TRI-73** (apply + pin).

## Exit criteria
1. Structured decision **persisted to state/results AND verified to engage on Ollama**, with `decision_extraction_method` recorded per run; raw 5-level rating captured (TRI-74). If structured output does not engage locally, it's documented as a finding.
2. A genuinely **all-local `benchmark_local_b`** exists and runs on v0.3.0 (`--dry-run`).
3. Tool-slot candidates passed the tool-calling gate at **Q4_K_M**.
4. **Local cost = $0 / provider-aware; cloud-reference pricing corrected** (opus-4-8 row + `_normalize_model` fix + Haiku refresh) so the yardstick's $ column is real; wall-time/ticker reported for all.
5. Benchmarking ran with **cache OFF** (or the documented split); results carry run-ids.
6. **Decision-stability, extractability (+ method), quality mean/σ** reported per config; every UNKNOWN investigated.
7. **Cloud reference configs run (~3 repeats each) and reported as the ceiling.** A **stability × quality × wall-time (+ $ for cloud refs)** table, and a **recommended production config vs the 8.0 gate** — or an explicit, evidenced "no local config clears 8.0."
8. `measure_ollama_speed.py` updated to the candidate set; prior results (Tasks 005/010/011) marked superseded.

## Prerequisites
- **Disk:** Wave-1 pulls are large — deep + tool + quick candidates (r1:8b/14b/32b/70b, qwen3.6:27b/35b, qwen3-coder:30b, gpt-oss:20b/120b, gemma-4:27b, llama3.3:70b) total **~250–270 GB new**. **Check free disk before pulling.**
- Ollama running; models at Q4_K_M; repo venv; Anthropic paper credits (for the ~6 cloud-reference runs only).
- **Time: 2–4 days serial.** Local runs are ~17–24 min each on one Ollama; the staged sweep across ~8 deep-slot candidates + configs is multi-day — **do NOT interrupt long Ollama runs.** Use the N=3 screen → N=5 finalists staging.

## Guardrails & gate
- Paper/`--dry-run` only; `--execute` forbidden. **Production candidates strictly local.** Cloud (Opus 4.8 / Haiku) runs **only as a benchmark reference/ceiling** — flagged `test-only`, never shipped (too expensive — Jeff).
- Verify every Ollama tag on pull. DEVELOP → QA (Codex) → paper-smoke UAT → Arbiter. App Manager writes the acceptance check; does not declare Done.
