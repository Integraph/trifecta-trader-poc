# TRI-70 — Session Handoff (DEVELOP continuation)

**For:** the Claude Code terminal session taking over TRI-70 DEVELOP.
**Read first:** `docs/TRI-70_DEVELOP_KICKOFF.md` (mission + gotchas) and `docs/TASK_TRI-70_BENCHMARK.md` (v3 work order — canonical steps + 8 exit criteria). This file only adds the **session state** those docs don't have.

## Where the previous session left off (2026-07-02)

**Git state — done:**
- TRI-66 gate artifacts + TRI-70 docs committed; `main` fast-forwarded to the TRI-66 tip and **pushed** (commit `710d01d`).
- You are on branch **`jeff/tri-70-step-0-re-benchmark-current-local-models-on-the-m3-max`** (created off that main). Engine verified: vendor at v0.3.0 `85946c2`, zero-mod.

**Step 1 (structured extraction) — code written, NOT yet verified or committed:**
- `tests/test_signal_processing.py`: new class `TestExtractDecisionDetailed` (8 tests) — was confirmed RED before implementation.
- `src/signal_processing.py`: implementation applied — hoisted `PM_RATING_MAP`, added `_PM_RENDER_RATING`/`_PM_RENDER_HEADERS`/`_FENCED_CODE`, refactored `extract_decision` → `_regex_decision_detailed(text) -> (decision, token)`, added `extract_decision_detailed(text) -> {decision, rating_5, method}` and the backcompat `extract_decision` wrapper.
- Design note: `method=rendered_structured_parse` is detected by the **exact v0.3.0 `render_pm_decision()` template fingerprint** — a line-anchored `**Rating**: <5level>` PLUS `**Executive Summary**:`/`**Investment Thesis**:` headers (fence-stripped first). This deterministically recovers the typed rating with **zero vendor modification**. Free-text labels → `regex`; nothing → `unknown`.
- **YOUR FIRST ACTION:** run `python -m pytest tests/test_signal_processing.py -q` — expect ~57 passed. Fix anything red, then finish Step 1:
  1. Wire `extract_decision_detailed` into `src/run_analysis.py` (~line 296): add `pm_rating_5` + `decision_extraction_method` to the result JSON (keep the existing `decision` field/printing).
  2. Land TRI-79 run-ids here: add a `run_id` field to the result JSON and (env-gated, e.g. `TRIFECTA_RUN_ID_FILES=1`) also write a run-id-suffixed copy of the result file so benchmark repeats never clobber.
  3. Commit Step 1 (tests + implementation together).

**Model pulls — restart them; the previous background pull died with the session:**
`ollama pull` is idempotent (skips completed layers). Run in this order (small/critical first), logging failures without stopping:
```
qwen3.6:27b ; qwen3-coder:30b ; deepseek-r1:8b ; deepseek-r1:14b ;
qwen3.6:35b (if FAIL → qwen3.6:35b-a3b — the library-confirmed tag) ;
gpt-oss:20b ; deepseek-r1:32b ;
gemma4:27b (tag TBD — try gemma-4:27b, gemma4; log-only if all fail) ;
qwen3.7 (expected FAIL — 10-second registry check, log it) ;
llama3.3:70b ; deepseek-r1:70b ; gpt-oss:120b
```
Disk was verified: **859 GB free** (need ~250–270 GB). Already pulled: qwen3.5:9b/27b/35b-a3b, qwen2.5:14b/32b, mistral-small:22b, llama3.1:8b, mistral:7b.

**Then continue with the work order Steps 2–8 exactly as written** (tool-calling gate → pricing triad → cache-off → speed-script refresh → staged runs N=3 screen @1 ticker → N=5 finalists → cloud-reference yardstick ~3 repeats → report + recommendation).

## Session learnings not in the docs (don't rediscover)
- The structured render **did engage on the cloud path** in the TRI-66 smokes (`**Rating**: Overweight` + `**Executive Summary**:` = the template) and **fell back to free text on local qwen2.5** — so expect `decision_extraction_method` to split cloud-vs-local exactly as the work order suspects. That's the finding to quantify, not a bug.
- v0.3.0 quality scores run lower than the old engine: cloud 8.0–8.5, old-local 3.6–5.1. The 8.0 gate is calibrated to this; don't "fix" the scorer.
- Local pipeline runs are ~17–24 min; **never interrupt a slow Ollama run** (caused a false-negative in TRI-66 QA round 1).
- Result files without run-ids **clobber on re-run** (TRI-79) — copy evidence aside until your run-id change lands.
- Full pytest baseline = **8 known failures** (5× accuracy stale-date TRI-71, 1× port-8420 env, 2× mistral-small tool-calling) with `--ignore` on `test_reasoning_comparison.py`, `test_prompt_engineering.py`, `test_alpaca_connection.py` (those three make real paid/live calls — never run them casually).

## Guardrails (unchanged)
Paper/`--dry-run` only; `--execute` forbidden. Cloud configs `test-only`, never the shipped recommendation. Surface (don't code around): no local config clears 8.0; structured output not engaging on Ollama; any material v0.3.0 behavior. Report → `docs/TASK_TRI-70_REPORT.md`. You do not declare Done — QA (Codex) → UAT → Arbiter follow.
