# TRI-70 — DEVELOP Kickoff (Claude Code)

You are **DEVELOP** (Claude Code) executing **TRI-70** on the MacBook Pro M3 Max / 128 GB. TRI-66 is **Done** — the engine is on TradingAgents **v0.3.0** with zero-mod restored. This is the Step-0 benchmark that answers the MVP's core question: **can a cheap, all-local config clear the quality bar with a stable decision, or does the deep slot need more than local can give?**

## Canonical spec
Read first: **`docs/TASK_TRI-70_BENCHMARK.md` (v3)** — full steps + 8 exit criteria. QA brief: `docs/CODEX_TRIFECTA_ENGINE_HANDOFF.md`.

## Mission
Benchmark **local** models across the 3 slots on v0.3.0; build a genuinely **all-local** config; produce a **decision-stability × quality × wall-time** table; **recommend a production config** that clears the gate — or state plainly, with evidence, that none does. Then it goes to QA → UAT → Arbiter. **You do not declare Done.**

## The gate (fixed)
**Production pass mark: `quality_scorer` ≥ 8.0 / 10** (data-grounding not degraded; decision stable). *Local models only compete for production.* **Cloud (Opus 4.8 / Haiku) is a benchmark reference/ceiling ONLY — flagged `test-only`, never shipped (too expensive).**

## Setup
- Branch off current `main` (confirm TRI-32's push landed; the TRI-70 docs must be on your branch).
- **Check free disk first** — Wave-1 pulls total **~250–270 GB** new.
- Ollama running; pull candidates at **`Q4_K_M`**; Anthropic paper credits (only ~6 cloud-reference runs).
- **Budget 2–4 days serial** (~17–24 min/local run, one Ollama). **Do NOT interrupt a long Ollama run** — that's the #1 false-negative failure mode. Use the staging below.

## Verified facts & gotchas — build on these, don't rediscover
- **Cache OFF for benchmarking.** `make_cached_analyst`/`DataCache` inject cached analyst reports with NO LLM call — repeat-run numbers would measure the cache, not the model. Use **`--no-cache`** (`use_cache=False`, present in both entrypoints).
- **Pricing triad (or the cost column lies):** (a) local/Ollama pricing → **$0/provider-aware** (kill `hybrid_graph.py:114`'s Sonnet fallback); (b) add a **`claude-opus-4-8` row (`$5/$25`)**; (c) refresh Haiku **`$0.80/$4.00` → `$1.00/$5.00`**; (d) **fix `_normalize_model` so "opus" → `claude-opus-4-8`** — today it maps to the retired `claude-opus-4-5-20251101` ($15/$75), so the new row is never reached without this.
- **Structured decision extraction (v0.3.0's PM renders the typed `PortfolioDecision` to a string and stores only that):** persist a **normalized structured decision field** into state/results, keep regex as fallback, record **`decision_extraction_method ∈ {structured, rendered_structured_parse, regex, unknown}`** per run, and **verify structured output actually engages on the Ollama path** (local models often fall back to free text). Also record the raw **5-level** rating (data for TRI-74).
- **Speed harness is stale:** `scripts/measure_ollama_speed.py:18` hardcodes the qwen2.5/3.5 list — make it candidate-driven before using it as evidence.
- **Model availability (verify every tag at pull):** deep = `qwen3.6:35b`(MoE)/`:27b`, `gpt-oss:120b`, `deepseek-r1:8b`(0528)/`14b/32b/70b` (**NB: 14b/32b/70b are 2025-era distills on qwen2.5/llama3.3 bases — not "current"**). Tool = `qwen3-coder:30b`, `gpt-oss`, `llama3.3:70b`, **Gemma-4 27B** (verify current tag — *not* `gemma3`, which lacks the `tools` tag; include only if it passes the tool gate). **Blocked — do NOT pull:** GLM-5.1 (`glm-5.1:cloud` only, no local weights) and `deepseek-v4-flash` (no Ollama V4-arch runtime yet).

## Sequence (Steps 1–8 in the work order)
Structured-extraction-first (+ verify it engages) → pull + **tool-calling gate** → fix pricing/local-$0 → **cache-off** repeat harness → update the speed script → **staged runs: deep-slot screen at 1 ticker × N=3 → finalists at the small fixed watchlist × N=5** (run-ids so nothing clobbers) → report stability/extractability/quality/wall-time, **investigate every UNKNOWN** → run the ~3 cloud-reference repeats as the ceiling → recommend the winning local config vs the 8.0 gate (or the honest "none clears 8.0").

## Guardrails
- Paper/`--dry-run` only; `--execute` forbidden. Cloud configs are `test-only` and must **never** be selected as the shipped config.
- **Checkpoint — surface to the App Manager before committing around it:** if **no local config clears 8.0**, if **structured output doesn't engage on Ollama** (so extraction stays regex-fragile for the measured models), or any material v0.3.0 behavior you hit. These are findings, not failures — report them.

## Deliverable → `docs/TASK_TRI-70_REPORT.md`
The full **stability × quality × wall-time (+ $ for cloud refs)** table, `decision_extraction_method` breakdown, the recommended production config (or the evidenced "none clears 8.0"), and the cloud reference numbers as the ceiling. Then: Codex QA → paper-smoke UAT → Arbiter. The winner hands to **TRI-73** (apply + pin).
