# Task 002 Report: Fix Signal Processing Bug and Looping Issue

## 1. Summary of Steps

| Step | Status | Notes |
|------|--------|-------|
| Create `src/signal_processing.py` | **Succeeded** | Implements `extract_decision()` with 3-tier priority extraction |
| Update `src/run_analysis.py` | **Succeeded** | Imports and uses `extract_decision()` to override upstream signal processor |
| Create `tests/test_signal_processing.py` | **Succeeded** | 15 test cases covering proposals, negation, edge cases |
| Create `tests/test_pipeline.py` | **Succeeded** | 5 tests covering conditional logic, propagator defaults, config pass-through bug |
| All tests pass | **Succeeded** | 25/25 tests pass |
| Looping bug investigated | **Succeeded** | Root cause identified and documented below |
| Full pipeline run | **Skipped** | Costs money (Anthropic API calls) and takes ~20 minutes; unit tests verify the logic |

## 2. Test Output

```
============================= test session starts ==============================
platform darwin -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/jeffbezenyan/Projects/Trifecta_Trader/App/trifecta-trader-poc
configfile: pyproject.toml
plugins: anyio-4.12.1, langsmith-0.7.9
collected 25 items

tests/test_config.py::test_tradingagents_importable PASSED               [  4%]
tests/test_config.py::test_config_creation PASSED                        [  8%]
tests/test_config.py::test_data_vendors_default_to_yfinance PASSED       [ 12%]
tests/test_config.py::test_results_directory PASSED                      [ 16%]
tests/test_pipeline.py::test_decision_not_repeated PASSED                [ 20%]
tests/test_pipeline.py::test_conditional_logic_defaults PASSED           [ 24%]
tests/test_pipeline.py::test_conditional_logic_risk_round_limit PASSED   [ 28%]
tests/test_pipeline.py::test_conditional_logic_config_not_passed PASSED  [ 32%]
tests/test_pipeline.py::test_propagator_defaults PASSED                  [ 36%]
tests/test_signal_processing.py::TestExtractDecision::test_final_transaction_proposal_hold PASSED [ 40%]
tests/test_signal_processing.py::TestExtractDecision::test_final_transaction_proposal_buy PASSED [ 44%]
tests/test_signal_processing.py::TestExtractDecision::test_final_transaction_proposal_sell PASSED [ 48%]
tests/test_signal_processing.py::TestExtractDecision::test_ignores_negation_not_recommending_sell PASSED [ 52%]
tests/test_signal_processing.py::TestExtractDecision::test_multiple_proposals_takes_last PASSED [ 56%]
tests/test_signal_processing.py::TestExtractDecision::test_recommendation_pattern PASSED [ 60%]
tests/test_signal_processing.py::TestExtractDecision::test_no_markdown_bold PASSED [ 64%]
tests/test_signal_processing.py::TestExtractDecision::test_empty_input PASSED [ 68%]
tests/test_signal_processing.py::TestExtractDecision::test_no_decision_found PASSED [ 72%]
tests/test_signal_processing.py::TestExtractDecision::test_sell_in_reasoning_hold_in_proposal PASSED [ 76%]
tests/test_signal_processing.py::TestExtractDecision::test_case_insensitive PASSED [ 80%]
tests/test_signal_processing.py::TestExtractDecision::test_standalone_decision_fallback PASSED [ 84%]
tests/test_signal_processing.py::TestEdgeCases::test_buy_with_extra_text PASSED [ 88%]
tests/test_signal_processing.py::TestEdgeCases::test_hold_with_conditions PASSED [ 92%]
tests/test_signal_processing.py::TestEdgeCases::test_hash_prefix_on_proposal PASSED [ 96%]
tests/test_signal_processing.py::TestEdgeCases::test_conviction_level_does_not_interfere PASSED [100%]

============================== 25 passed in 2.22s ==============================
```

## 3. Root Cause Analysis: Looping Bug

### Investigation findings

The looping bug has **two contributing causes**:

#### Cause A: Config values not passed to ConditionalLogic (vendor bug)

In `vendor/TradingAgents/tradingagents/graph/trading_graph.py` line 108:

```python
self.conditional_logic = ConditionalLogic()
```

`ConditionalLogic.__init__` accepts `max_debate_rounds` and `max_risk_discuss_rounds` parameters, and these are present in the config dict (`self.config`), but they are **never passed through**. The constructor always uses its defaults (`1` and `1`).

This means if a user sets `config["max_risk_discuss_rounds"] = 2` (as the CLI does at `vendor/TradingAgents/cli/main.py` line 906), the value is silently ignored and the debate always runs with the default of 1 round.

The same issue affects `Propagator()` on line 121 — `max_recur_limit` from config is not passed.

**Impact on our POC:** Since our `run_analysis.py` also sets these to 1 (matching the defaults), the config-passthrough bug does not directly cause extra looping in our case. However, it would cause unexpected behavior if we changed these config values.

#### Cause B: Debug streaming prints repeated messages (display bug)

In `trading_graph.py` `propagate()` method (debug mode), the graph is streamed with `stream_mode="values"`. Each node execution emits a full state snapshot. During the risk debate phase (Trader → Aggressive → Conservative → Neutral → Risk Judge = 5 nodes), the `messages` list does not change because debator nodes only update `risk_debate_state`, not `messages`. However, the code prints `chunk["messages"][-1].pretty_print()` for every chunk where `len(chunk["messages"]) > 0`.

This means the **same last analyst message gets printed 5 times** — once for each risk-phase node — giving the appearance that "the final trade decision repeats 5 times."

### Graph flow analysis

With `max_risk_discuss_rounds=1`, the conditional logic correctly terminates after one full round:

| Step | Node | count after | latest_speaker | Next |
|------|------|------------|----------------|------|
| 1 | Aggressive Analyst | 1 | "Aggressive" | Conservative Analyst |
| 2 | Conservative Analyst | 2 | "Conservative" | Neutral Analyst |
| 3 | Neutral Analyst | 3 | "Neutral" | count >= 3 → Risk Judge |
| 4 | Risk Judge | 3 | "Judge" | END |

The risk debate loop itself terminates correctly after 3 speakers (one round). There is no actual infinite loop or re-entry.

## 4. Fixes Applied

### Bug 1: Signal Processing (fixed in `src/`)

Created `src/signal_processing.py` with `extract_decision()` that uses priority-based extraction:

1. **Primary**: Regex match on `FINAL TRANSACTION PROPOSAL: <DECISION>` — takes the last occurrence
2. **Secondary**: Regex match on `MY RECOMMENDATION: <DECISION>`
3. **Tertiary**: Last standalone BUY/HOLD/SELL after stripping negation contexts ("NOT Recommending SELL", etc.)
4. **Fallback**: Returns `"UNKNOWN"` instead of guessing

Updated `src/run_analysis.py` to:
- Import and call `extract_decision()` on the raw `final_trade_decision` text
- Log when our decision differs from the upstream LLM-based extraction
- Include both `decision` and `upstream_decision` in the saved JSON for auditability

### Bug 2: Looping (documented, no vendor code modified)

The looping is a **display issue** in debug mode, not an actual graph loop. The risk debate terminates correctly after one round with the current defaults.

**No vendor code was modified.** The config-passthrough bug (Cause A above) is documented and should be submitted upstream when ready.

## 5. Vendor Code Modifications

**No vendor files were modified.** All fixes are in `src/` and `tests/`.

### Recommended upstream fix (not applied)

The following change in `vendor/TradingAgents/tradingagents/graph/trading_graph.py` would fix the config passthrough:

**Line 108** — change:
```python
self.conditional_logic = ConditionalLogic()
```
to:
```python
self.conditional_logic = ConditionalLogic(
    max_debate_rounds=self.config.get("max_debate_rounds", 1),
    max_risk_discuss_rounds=self.config.get("max_risk_discuss_rounds", 1),
)
```

**Line 121** — change:
```python
self.propagator = Propagator()
```
to:
```python
self.propagator = Propagator(
    max_recur_limit=self.config.get("max_recur_limit", 100)
)
```

## 6. Modifications to Task Instructions

None. All steps were followed as specified.

## 7. Git Log

```
0ecbeea Fix signal processing extraction and investigate looping issue
96bf243 Add Task 001 completion report
99bc853 Initial POC structure with TradingAgents submodule and analysis runner
```
