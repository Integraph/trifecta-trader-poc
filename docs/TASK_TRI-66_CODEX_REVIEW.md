# TASK TRI-66 - Codex QA Review

**Date:** 2026-07-01
**Reviewer:** Codex QA
**Branch reviewed:** `jeff/tri-66-upgrade-vendored-tradingagents-v020-v030`
**Verdict:** CHANGES-REQUESTED

## Findings

### P1 - Decision extraction is not actually header-anchored and can overwrite a valid PM decision

`src/signal_processing.py:47-52` claims to anchor the v0.3.0 Portfolio Manager matcher to decision header labels, but the regex has no start-of-line/header boundary:

```python
pm_pattern = (
    r'\*{0,2}(?:Recommendation|Rating|Action|Final\s+Transaction\s+Proposal)'
    r'\*{0,2}[:\s]*\*{0,2}'
    r'(Buy|Overweight|Hold|Underweight|Sell)\*{0,2}'
)
```

Because method 0 takes the last match across the entire document, reasoning prose or quoted prior output can override the actual final decision. Repro:

```text
valid_rating_then_prose_rating: SELL
valid_recommendation_then_prose_action: SELL
valid_action_then_prose_final: SELL
no_decision_prose_label: SELL
```

Those came from:

```python
from src.signal_processing import extract_decision

extract_decision("**Rating**: Hold\n\nRationale: prior note wrote Rating: Underweight.")
extract_decision("**Recommendation**: Buy\n\nRationale: prior playbook said Action: Sell.")
extract_decision("**Action**: Hold\n\nRationale: stale memo labeled Final Transaction Proposal: Sell.")
extract_decision("The analyst wrote Rating: Underweight in the bearish argument; we have not decided yet.")
```

This directly violates `docs/TRI-66_QA_KICKOFF.md:19` and contradicts `docs/TASK_TRI-66_REPORT.md:51`, which says the matcher is header-anchored and prose-safe. This is a measurement-integrity blocker: a benchmark or paper signal can be silently converted to the wrong BUY/SELL decision.

Requested fix:
- Anchor PM labels to actual decision lines, e.g. start of line with optional markdown heading/bold prefix.
- Preserve the local qwen formats (`Action`, `Final Transaction Proposal`) but only when they appear as decision headers.
- Add adversarial tests where a valid final PM decision is followed by prose/quoted prior labels.
- Prefer structured `PortfolioDecision` extraction when available in TRI-70, but TRI-66 still needs the regex fallback to be safe.

### P2 - I could not independently reproduce the two smoke gates to completion

The developer report records both dry-run smokes as passed:

- `hybrid_haiku_tools` -> BUY, quality 8.5, elapsed ~1001s.
- `hybrid_aggressive_qwen` -> SELL, quality 5.1, elapsed ~1402s.

My independent attempts did not reach a final decision before I interrupted them after prolonged no-output/no-new-artifact windows:

- `python -m src.run_batch --tickers AAPL --hybrid hybrid_haiku_tools --dry-run`
  - Reached graph execution and wrote fresh AAPL analyst caches.
  - Interrupt trace showed it waiting inside `vendor/TradingAgents/tradingagents/agents/risk_mgmt/conservative_debator.py:39`, on an Ollama/openai-compatible `qwen2.5:14b` chat completion.
- `python -m src.run_batch --tickers AAPL --hybrid hybrid_aggressive_qwen --dry-run`
  - Interrupt trace showed it waiting inside `vendor/TradingAgents/tradingagents/agents/researchers/bull_researcher.py:47`, also on the Ollama/openai-compatible path.

This may be runtime latency rather than a source regression; the report's elapsed times are long enough that patience is a factor. Still, Codex QA did not independently observe a parsed BUY/HOLD/SELL decision, so I cannot approve exit criteria 5/6 from my own run. After the P1 fix, rerun both smokes and preserve the exact final decision/quality/cost output.

## Verified Clean

- Linear is connected. TRI-66 is in `Trifecta -> Trifecta Trader - Engine`, status `In Review`; no child issues were returned. TRI-70 is the linked benchmark/config follow-up and remains Backlog.
- Submodule is at `85946c2f60768ab2dae23a5a36cd927662feef94`; `v0.3.0^{}` resolves to the same commit; `git -C vendor/TradingAgents diff --stat v0.3.0` is empty.
- Required deps are installed in this environment:
  - `langgraph==1.0.10`
  - `langchain-core==1.2.16`
  - `langgraph-checkpoint==4.1.1`
  - `langgraph-checkpoint-sqlite==3.1.0`
  - `yfinance==1.5.1`
  - `stockstats==0.6.8`
- `import langgraph.checkpoint.sqlite`, `import stockstats`, `import yfinance`, and `import src.run_analysis` all succeed.
- No config/model-string diff in `config/hybrid_llm.yaml`, `src/hybrid_llm.py`, or `admin-ui/src/components/config/ConfigPage.tsx`.
- `pytest --ignore=tests/test_alpaca_connection.py --ignore=tests/test_prompt_engineering.py --ignore=tests/test_structured_output.py -q` matched the documented baseline:
  - `8 failed, 596 passed, 3 skipped`
  - Failures were the expected 5 accuracy reporter stale-date tests, 1 scheduler/port-8420 test, and 2 mistral tool-calling tests.
- `pytest tests/test_signal_processing.py -q` passes: `28 passed`.
- `pytest tests/test_pipeline.py -q` passes: `5 passed`.
- Risk Judge remap is structurally correct:
  - `src/hybrid_graph.py` uses `create_portfolio_manager(self.reasoning_deep_llm)`.
  - Final node is `"Portfolio Manager"`.
  - Vendor `TradingAgentsGraph.propagate()` injects `past_context` from `TradingMemoryLog`; `HybridTradingGraph.propagate()` delegates to it.
- Offline memory is file-based in `vendor/TradingAgents/tradingagents/agents/utils/memory.py`; I found no embeddings/API dependency in the active memory path.

## Notes

- Current HEAD includes an extra docs-only TRI-32 commit (`PROJECT_BRIEF.md`, `docs/TRI-66_QA_KICKOFF.md`) after the two TRI-66 implementation commits. I did not treat that as app-code scope creep.
- The developer report's "header-anchored" statement is the main discrepancy. The tests prove rating words without labels are ignored, but they do not cover label-shaped prose or stale quoted decisions, which is the real failure mode.

---

# TRI-66 - Codex QA Review Round 2

**Date:** 2026-07-02
**Commits reviewed:** `2a57026` plus `2cffe43`
**Verdict:** CHANGES-REQUESTED

## Round-2 Finding

### P1 - PM matcher still lets quoted or fenced stale decisions override the real decision

The four round-1 repro cases are fixed. These now return the intended values:

```text
r1_valid_rating_then_prose_rating: HOLD
r1_valid_recommendation_then_prose_action: BUY
r1_valid_action_then_prose_final: HOLD
r1_no_decision_prose_label: UNKNOWN
```

The new tests in `tests/test_signal_processing.py:95-144` are real line-anchoring tests, not tautological. They cover mid-line prose labels, looping decision headers, markdown heading/list prefixes, qualified labels, and unlisted-qualifier prose.

The P1 is still not closed, though. `src/signal_processing.py:61-68` allows `>` in the line prefix and does not ignore fenced code blocks:

```python
pm_pattern = (
    r'^[\s#>*-]{0,12}'
    r'(?:(?:Investment|Final|Overall|Portfolio|Trading|Recommended)\s+)?'
    r'(?:Recommendation|Rating|Action|Transaction\s+Proposal)'
    r'\*{0,2}\s*:\s*\*{0,2}'
    r'(Buy|Overweight|Hold|Underweight|Sell)\b'
)
```

Because the last matching line wins, a stale quoted/code-block decision below the real decision still overrides it:

```text
blockquote_stale_below_real_should_keep_real: actual=SELL expected=HOLD pass=False
fenced_code_stale_below_real_should_keep_real: actual=SELL expected=HOLD pass=False
literal_label_start_residual: actual=SELL expected=HOLD pass=False
```

Repro inputs:

```python
extract_decision("**Rating**: Hold\n\n> Rating: Underweight\n> from yesterday\n\nRationale rejects it.")
extract_decision("**Rating**: Hold\n\n```\nRating: Underweight\n```\n\nRationale rejects it.")
extract_decision("**Rating**: Hold\n\nRecommendation: Sell was the prior memo, not the current call.")
```

The first two are not acceptable residuals. A blockquote or fenced code block is exactly how stale prior output is likely to be embedded, so counting those as current PM decision lines can still silently flip BUY/HOLD/SELL. This is the same measurement-integrity class as round 1.

Requested fix:
- Do not allow blockquote prefixes (`>`) as live decision headers.
- Strip or ignore fenced code blocks before applying the PM decision-line matcher.
- Add tests where blockquoted and fenced stale decisions appear both above and below the real decision.
- Keep "last genuine decision line wins" for repeated live PM output.

Judgment call: structured `PortfolioDecision` extraction does not need to be forced into TRI-66. A bounded regex fallback is acceptable for this upgrade if it ignores obvious quoted/code stale text. The documented residual of a prose line literally beginning `Recommendation: Sell` is tolerable as a known ceiling until TRI-70, but the current implementation goes beyond that ceiling.

## Round-2 Verification

- `pytest tests/test_signal_processing.py -q` -> `37 passed`.
- `pytest --ignore=tests/test_alpaca_connection.py --ignore=tests/test_prompt_engineering.py --ignore=tests/test_structured_output.py -q` -> `8 failed, 605 passed, 3 skipped`; failures are the known baseline set:
  - 5x `tests/test_accuracy_reporter.py::TestSummary`
  - 1x `tests/test_admin_scheduler.py::TestSchedulerHistory::test_history_returns_runs`
  - 2x `tests/test_local_tool_calling.py` for `mistral-small:22b`
- Submodule still zero-mod: `vendor/TradingAgents` HEAD is `85946c2f60768ab2dae23a5a36cd927662feef94`, matching `v0.3.0^{}`; diff vs tag is empty.
- Deps/import gate still clean:
  - `langgraph==1.0.10`
  - `langchain-core==1.2.16`
  - `langgraph-checkpoint==4.1.1`
  - `langgraph-checkpoint-sqlite==3.1.0`
  - `yfinance==1.5.1`
  - `stockstats==0.6.8`
  - `import src.run_analysis` succeeds.
- No config/model-string diff in `config/hybrid_llm.yaml`, `src/hybrid_llm.py`, or `admin-ui/src/components/config/ConfigPage.tsx`.
- Real PM styles still parse:
  - `**Rating**: Overweight` -> `BUY`
  - `**Action**: Underweight` -> `SELL`
  - `**Investment Recommendation: Underweight**` -> `SELL`
  - legacy `FINAL TRANSACTION PROPOSAL: **HOLD**` -> `HOLD`
  - `an underweighted position might be prudent` -> `UNKNOWN`

## Scope Notes

- I did not rerun full smokes in round 2. The requested code-level P1 is still open, and re-running smokes would also clobber result files per the already-known result-evidence issue.
- Linear confirms TRI-78 exists in Backlog and owns run-to-run signal non-determinism for TRI-69/TRI-70 benchmark design. That variance is out of TRI-66 scope as long as TRI-66 uses smokes only to prove "pipeline runs and returns a valid decision," not to claim stable edge or signal quality.
