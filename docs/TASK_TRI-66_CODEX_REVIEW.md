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
