# TRI-70 Codex QA Review

Date: 2026-07-04
Branch under review: `jeff/tri-70-step-0-re-benchmark-current-local-models-on-the-m3-max`
Commit under review: `8d1d6df`

Verdict: CHANGES-REQUESTED

## Prompt Review

`docs/TRI-70_QA_KICKOFF.md` is good enough to execute as scoped QA. It correctly prevents the two bad shortcuts: claiming raw input identity from saved finalist JSONs, and treating output-layer diffs as root-cause attribution.

Minor prompt nit: `docs/TRI-70_QA_KICKOFF.md:1` still says `v3` while this is being reviewed as v4. Not blocking, but sloppy.

## Findings

### [P1] Report still overclaims input identity and local-specific root cause

Refs:
- `docs/TASK_TRI-70_REPORT.md:73`
- `docs/TASK_TRI-70_REPORT.md:235`
- `docs/TASK_TRI-70_REPORT.md:236`
- `docs/TASK_TRI-70_REPORT.md:237`
- `src/run_analysis.py:380`
- `src/run_analysis.py:381`

The finalist artifacts support the headline that `benchmark_local_b` was decision-unstable in the saved runs. They do not support the stronger claim that the runs had "identical inputs" or that the instability is proven "local-specific, not an engine/task property."

I inspected all 15 finalist JSON files:

- `results/*/analysis_2026-06-27_benchmark_local_b_finalist-*.json`
- Saved fields include `final_trade_decision_text` and `trader_investment_plan`.
- Saved fields do not include `market_report`, `news_report`, `sentiment_report`, `fundamentals_report`, raw fetched payloads, or risk-debate history.

Existing finalist outputs do show instability before or at the PM decision layer. For AAPL, the five saved runs are BUY, BUY, HOLD, HOLD, HOLD, and the saved trader plans differ in length/content across all repeats. NVDA and TSLA show the same kind of run-to-run output variance. That is useful localization, not source-input proof.

Required change: soften the report language. Replace claims like "identical inputs -> BUY/HOLD/SELL swing" and "local-specific, not an engine/task property" with language that says the saved artifacts prove output instability under the same ticker/date benchmark setup, while definitive input-drift-vs-model-noise attribution requires TRI-69's input snapshots and point-in-time mode.

This does not require structured input snapshots inside TRI-70. It does require not claiming evidence the artifacts do not contain.

### [P2] Independent N=5 repro was not completed from the prescribed command

Refs:
- `docs/TRI-70_QA_KICKOFF.md:14`
- `vendor/TradingAgents/tradingagents/default_config.py:74`
- `scripts/run_tri70_benchmark.py:64`
- `scripts/run_tri70_benchmark.py:75`

The exact repro command failed in this Codex sandbox: all 5 subprocesses errored because TradingAgents tried to write the memory log under `~/.tradingagents`, which is outside the writable workspace here. The aggregate written to `results/tri70_qa_repro_agg.json` has `runs_ok=0`, `errors=5`.

I reran with a workspace memory log:

```bash
TRADINGAGENTS_MEMORY_LOG_PATH="$PWD/results/tri70_qa_memory/trading_memory_$TS.md" \
PYTHONPATH=".:vendor/TradingAgents" \
python scripts/run_tri70_benchmark.py \
  --configs benchmark_local_b \
  --tickers AAPL \
  --n 5 \
  --date 2026-06-27 \
  --tag qa-repro-localmem-$TS \
  --output results/tri70_qa_repro_agg_localmem.json
```

That completed only repeat 1 before runtime became impractical in this environment: BUY, quality 8.5, wall time 1029.2s. Repeat 2 exceeded the review window and I interrupted it. So I did not independently reproduce the reported 0.60 agreement.

Required change: add the workspace-safe `TRADINGAGENTS_MEMORY_LOG_PATH` guidance to the QA/runbook path, or provide a completed independent QA repro artifact before Arbiter/UAT treats the 0.60 rerun as independently verified.

### [P2] Cloud stability language is too strong for N=3

Refs:
- `docs/TASK_TRI-70_REPORT.md:231`
- `docs/TASK_TRI-70_REPORT.md:232`
- `docs/TASK_TRI-70_REPORT.md:234`
- `docs/TASK_TRI-70_REPORT.md:235`

The cloud reference runs are useful smoke evidence, but `3/3` has a wide 95% Wilson interval: approximately `[0.438, 1.000]`. Calling the finding "decisive" and "decision-STABLE" overstates what N=3 can prove, especially combined with the missing raw-input snapshots above.

The right claim is: cloud references were stable in the observed N=3 smoke runs, while local finalist artifacts were unstable at N=15. That directionally supports the recommendation, but it is not a reliability proof.

Tool gates with `3/3` should be read the same way: screen PASS evidence, not high-confidence production reliability.

## Checks Performed

- Artifact inspection: confirmed 15 finalist JSONs do not preserve raw analyst/source inputs. They save final PM text and trader plan only.
- Output-layer localization: existing finalist artifacts show trader-plan and PM-decision text differences across repeats. That localizes variance upstream of or at PM, but cannot distinguish raw source drift from upstream LLM sampling.
- Repro attempt: exact command produced 5/5 errors from sandbox memory-log permissions. Workspace-memory retry produced 1 successful repeat, then was interrupted during repeat 2 for runtime.
- Wilson intervals:
  - `3/3`: point estimate 1.00, 95% Wilson about `[0.438, 1.000]`
  - `9/15`: point estimate 0.60, 95% Wilson about `[0.357, 0.802]`
  - `3/5`: point estimate 0.60, 95% Wilson about `[0.231, 0.882]`
- Config/pricing spot checks:
  - `benchmark_local_b` is all-local: `ollama/qwen3-coder:30b`, `ollama/qwen3.5:9b`, `ollama/deepseek-r1:8b`.
  - `benchmark_opus_a` routes Haiku tools, local quick, and `anthropic/claude-opus-4-8` deep.
  - `_normalize_model("anthropic/claude-opus-4-8")` resolves to `claude-opus-4-8`.
  - `MODEL_PRICING["claude-opus-4-8"]` is `{input: 5.0, output: 25.0}`.
- Pytest baseline:
  - Command reached 100% and printed exactly the expected 8 failures.
  - Summary: `8 failed, 621 passed, 2 skipped, 190 warnings`.
  - Failure set matched the expected five `test_accuracy_reporter.py::TestSummary` failures, `test_admin_scheduler.py::TestSchedulerHistory::test_history_returns_runs`, and the two `mistral-small:22b` tool-calling failures.
  - After the final pytest summary, leftover background analysis work kept printing; I interrupted the process after the summary was captured.

## Bottom Line

Do not block on the fact that all-local failed the gate. That finding is credible from the saved outputs.

Do block on the report claiming more than the artifacts prove. TRI-70 can close after the report downgrades "identical inputs/local-specific/decision-stable" language to the evidence actually available, and after the independent repro story is either completed or explicitly marked as not completed in this sandbox.
