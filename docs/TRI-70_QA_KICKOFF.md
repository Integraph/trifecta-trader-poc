# TRI-70 — QA Kickoff (Codex) — SCOPED, not a full re-audit · v3

You are **QA (Codex)** for **TRI-70** (the local-model benchmark). The two Dev reviews agreed and the report's arithmetic/run-accounting was verified in review — so this is **scoped, not ceremonial**. Do the checks below in order, then verdict. Independence: form your own findings before re-reading `docs/TASK_TRI-70_REPORT.md`. Write no committed app code. Output → `docs/TASK_TRI-70_CODEX_REVIEW.md`.

**Under review:** branch `jeff/tri-70-step-0-re-benchmark-current-local-models-on-the-m3-max` (commits through `8d1d6df`).
**Prereqs:** run from the **repo root in the repo venv** (or set `PYTHONPATH=".:vendor/TradingAgents"` — the top-level harness imports repo modules); Ollama with **all three** `benchmark_local_b` models pulled — `qwen3-coder:30b`, `qwen3.5:9b`, `deepseek-r1:8b`; Anthropic paper credits only if you re-run cloud refs. The harness (`scripts/run_tri70_benchmark.py`) is **analysis-only** (no `--execute`), forces `--no-cache`, and writes run-id-suffixed outputs — keep it that way.

## Checks (in this order)

- **1 — Artifact inspection (do first).** Confirm the finalist result JSONs do **not** preserve the raw analyst inputs — they save only `final_trade_decision_text` + `trader_investment_plan` (`src/run_analysis.py:380`), no `market_report`/`news_report`/`sentiment_report` or fetched payloads. **Record as a finding:** input-drift-vs-model-noise is **not fully resolvable from the current artifacts**, and the definitive answer requires the **input-snapshot + point-in-time mode** the edge-test plan (TRI-69) builds. Do not claim the files prove input identity.

- **2 — Reproduce the 0.60.** Nobody has re-executed the pipeline. Exact command:
  ```
  PYTHONPATH=".:vendor/TradingAgents" python scripts/run_tri70_benchmark.py --configs benchmark_local_b --tickers AAPL --n 5 --date 2026-06-27 --tag qa-repro-$(date +%Y%m%d%H%M%S) --output results/tri70_qa_repro_agg.json
  ```
  *(The `PYTHONPATH` prefix is required — without it the top-level script fails `ModuleNotFoundError: No module named 'src'`. The timestamped tag avoids clobbering a prior repro.)*
  Report the agreement. (Note: this re-run still hits live sources — so a ~0.60 here is **not** attributable to model noise until Check 3.)

- **3 — Localize the instability (weak output-layer proxy — state the limit).** Across the repro repeats, diff `trader_investment_plan` / `final_trade_decision_text`. **These are LLM *outputs*, not source inputs** (Trader generates the plan; PM generates the decision text — `vendor/.../trader/trader.py:51`, `vendor/.../managers/portfolio_manager.py:66`). So this proxy **cannot distinguish raw input drift from upstream LLM sampling variance** — it only shows **whether the instability appears before or at the PM decision layer.** If the upstream narratives already differ run-to-run, variance is upstream (inputs and/or analyst sampling); if the trader plan is stable but the final label flips, it's the **PM / risk-debate layer or unsnapshotted intermediate context** — the PM also consumes `risk_debate_state["history"]`, which is **not saved** (`portfolio_manager.py:55`), so this is *not* proof of pure PM sampling. Report it as *localization*, not attribution.

- **4 — Characterize the thin claims with a method.** Cloud stability `1.00` and each tool-gate pass rest on **N=3**. Compute a **95% Wilson interval** for each (e.g. `3/3` has a wide lower bound → explicitly **downgrade** the "stable" claim to what the interval supports). "Cloud stable, local not" is directionally supported (0.60 @ N=15 vs 1.00 @ N=3 + tighter σ) but thin — flag any claim the prose over-states.

- **5 — False-green pass.** Run the suite exactly as the baseline was measured:
  ```
  pytest --ignore=tests/test_reasoning_comparison.py --ignore=tests/test_prompt_engineering.py --ignore=tests/test_alpaca_connection.py -q
  ```
  The **failure set must be exactly these 8** (passed count may differ with TRI-70's new tests): `test_accuracy_reporter.py::TestSummary::{test_aggregates_by_decision, test_best_and_worst_signals, test_counts_complete_outcomes, test_direction_accuracy_aggregation, test_quality_tier_breakdown}`, `test_admin_scheduler.py::TestSchedulerHistory::test_history_returns_runs`, `test_local_tool_calling.py::test_tool_calling_{basic,multi_tool}[mistral-small:22b]`. **Any failure outside these 8 = CHANGES-REQUESTED.** Confirm the new configs / structured-extraction / pricing fixes (opus-4-8 row + `_normalize_model` + Haiku refresh) are real, tested, and don't perturb the config-roster tests.

## Rules
- **Safety:** any re-run uses the analysis-only harness (no `--execute`) and keeps run-id-suffixed outputs.
- **Escalate, don't fix silently.** `file:line` + severity. Verdict: **APPROVED** / **CHANGES-REQUESTED** → `docs/TASK_TRI-70_CODEX_REVIEW.md`.
- You do **not** declare Done. On APPROVED → paper-smoke UAT → Arbiter. The *headline* (no all-local config clears the gate) is a **finding, not a defect** — validate the evidence, not the conclusion's palatability.
