# Task 009 Report: Fix Parameter Extraction & Dual-Source Parsing

**Date:** 2026-03-02  
**Duration:** ~35 minutes (including 32-minute live pipeline run)  
**Status:** ✅ Complete

---

## 1. Step Summary

| Step | Description | Status |
|------|-------------|--------|
| 1 | Expand `_extract_from_structured_block()` with multi-format regex | ✅ Done |
| 2 | Add `extract_trade_params_dual()` for Risk Judge + Trader fallback | ✅ Done |
| 3 | Wire dual-source into `run_analysis.py`, capture `trader_investment_plan` | ✅ Done |
| 4 | Add `deduplicate_repeated_blocks()` to `signal_processing.py` | ✅ Done |
| 5 | Add tests: multi-format block, dual-source, live data, dedup | ✅ Done |
| 6 | AAPL dry-run with `--dry-run` flag | ✅ Done (HOLD, 32 min) |
| 7 | Full test suite | ✅ 127 passed, 8 skipped, 2 pre-existing failures |
| 8 | Commit + report | ✅ Done (commit 888ee0f) |

---

## 2. Regex Expansion

### Problem
`_extract_from_structured_block()` only matched `## EXECUTION PARAMETERS` (h2 format), but the Qwen Trader produces `3. **Execution Parameters:**` (numbered bold list item). The Risk Judge (Claude) often doesn't produce any structured block at all.

### Solution: 5-pattern fallback chain

```python
block_patterns = [
    # h2: ## EXECUTION PARAMETERS
    r'##\s*EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
    # Bold: **Execution Parameters:** or **Execution Parameters**
    r'\*\*Execution Parameters[*:]*\s*\n(.*?)(?:\n##|\n\*\*[A-Z]|\Z)',
    # Numbered bold: 3. **Execution Parameters:**
    r'\d+\.\s*\*\*Execution Parameters[*:]*\s*\n(.*?)(?:\n##|\n\d+\.|\Z)',
    # h3: ### Execution Parameters
    r'###\s*Execution Parameters\s*\n(.*?)(?:\n##|\Z)',
    # Plain uppercase: EXECUTION PARAMETERS
    r'EXECUTION PARAMETERS\s*\n(.*?)(?:\n##|\Z)',
]
```

**Key fix:** Changed `\*?\*?:?` (broken for `:**`) to `[*:]*` which matches any sequence of `*` and `:` characters — correctly handles `:**`, `**`, `:` variants.

---

## 3. Dual-Source Extraction Results

### `extract_trade_params_dual()` Strategy

1. Parse `final_trade_decision_text` (Risk Judge) using full structured + regex fallbacks
2. If `stop_loss` or `price_target` still missing → parse `trader_investment_plan` (Trader)
3. Fill only the missing fields from Trader (Judge always wins on values it has)
4. Recalculate risk metrics with merged values

### Real-data test results

Using the actual AAPL Trader output from the March 2026 pipeline run:

```
test_trader_output_extraction:
  ✅ Trader extraction: SL=$256.5, Target=$248.0

test_dual_extraction_makes_actionable:
  ✅ Stop-loss:   $256.5
  ✅ Target:      $248.0
  ✅ Position:    2.0%
  ✅ R/R ratio:   2.1066...
  ✅ Actionable:  True
```

**Before Task 009:** 0/1 live runs were actionable (stop-loss missing from Risk Judge output).  
**After Task 009:** Dual-source ensures stop-loss is recovered from Trader output → actionable rate improves to 1/1 for BUY/SELL decisions.

---

## 4. Pipeline Dry-Run Results (AAPL, 2026-03-02)

```
Ticker:    AAPL
Date:      2026-03-02
Mode:      HYBRID (hybrid_qwen_enhanced)
Deep LLM:  claude-sonnet-4-5-20250929
Quick LLM: qwen2.5:14b (via Ollama)

[signal_processing] Corrected decision: upstream='SELL' -> ours='HOLD'

DECISION: HOLD
Quality Score: 9.1/10
  Reasoning depth:     7/10
  Data grounding:      10/10
  Risk awareness:      10/10
  Decision consistent: Yes
  Elapsed time:        1918.8s

TRADE PARAMETERS
  Decision:    HOLD
  Stop-loss:   $N/A
  Target:      $N/A
  Position:    3.0%
  R/R Ratio:   N/A
  Actionable:  False

  [DRY RUN — no order submitted]
```

**Note:** HOLD decisions are correctly not actionable (no trade placed). The position_pct of 3.0% was successfully extracted from the Trader's output. The `trader_investment_plan` field is now captured in the result JSON (length: 1,773 chars), containing Trader's execution parameters including Stop-Loss: $275 and Price Target: $215.

---

## 5. Deduplication Utility

Added `deduplicate_repeated_blocks()` to `src/signal_processing.py`.

Strategy: any line ≥ `min_block_length` chars whose first `min_block_length` characters have already been seen is considered a duplicate and dropped. Short separator lines always pass through.

Test results:
```
TestDeduplicateRepeatedBlocks::test_deduplicate_removes_repeated_blocks   PASSED
TestDeduplicateRepeatedBlocks::test_deduplicate_preserves_unique_content  PASSED
TestDeduplicateRepeatedBlocks::test_short_text_returned_unchanged         PASSED
TestDeduplicateRepeatedBlocks::test_empty_text_returned_unchanged         PASSED
TestDeduplicateRepeatedBlocks::test_none_text_returned_unchanged          PASSED
```

Note: This is a best-effort utility. It handles single-line block repetitions (most common case when `max_recur_limit` triggers the loop). The real fix for agent looping would be in vendor graph conditional logic, which we cannot modify per task rules.

---

## 6. Full Test Output

```
2 failed, 127 passed, 8 skipped in 109.37s (1:49)

FAILED tests/test_local_tool_calling.py::test_tool_calling_basic[mistral-small:22b]
FAILED tests/test_local_tool_calling.py::test_tool_calling_multi_tool[mistral-small:22b]
```

**Both failures are pre-existing** (documented since Task 007): `mistral-small:22b` outputs JSON as text content rather than structured `tool_calls`. This is a known model limitation unrelated to Task 009.

New tests added (all passing):

| File | Tests | Result |
|------|-------|--------|
| `tests/test_trade_params.py` | +3 (multi-format block class) | ✅ 24/24 |
| `tests/test_dual_source.py` | 6 new | ✅ 6/6 |
| `tests/test_dual_extraction_live.py` | 2 new | ✅ 2/2 |
| `tests/test_signal_processing.py` | +5 (dedup class) | ✅ 18/18 |

---

## 7. Git Log

```
888ee0f Task 009: fix parameter extraction & add dual-source parsing
8b3012e Add Task 008 completion report
5d07f82 Add structured execution output and fix deep LLM parameter extraction
58afbda Add Task 007 completion report
105c70c Add trade execution layer with position management and Alpaca paper trading
87ba240 Add prompt engineering for local model quality and multi-ticker validation
e31a40a Add local model scaling experiment with quality comparison
fc741b8 Add live hybrid validation, comparison tooling, and pipeline results
```

---

## 8. Vendor Code Modifications

**None.** All changes are in `src/` and `tests/` only.

---

## 9. Actionable Rate Comparison

| Run | Task | Decision | Stop-Loss Source | Actionable |
|-----|------|----------|-----------------|------------|
| AAPL (Mar 1, Task 008) | 008 | SELL | Risk Judge (missing) | ❌ No |
| AAPL (Mar 2, Task 009) | 009 | HOLD | N/A (HOLD = not traded) | N/A |

**Theoretical improvement (validated by unit tests):** When a BUY/SELL decision is produced, dual-source extraction now recovers stop-loss from the Trader's `**Execution Parameters:**` block even when the Risk Judge omits it. The `test_dual_extraction_makes_actionable` and `test_dual_extraction_with_real_data` tests confirm this with actual Trader text from the March pipeline run.

**Next BUY/SELL run:** Expected actionable rate to be 1/1 (vs. 0/1 before this fix).

---

## 10. Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `src/execution/trade_params.py` | Modified | 5-pattern block regex + `extract_trade_params_dual()` |
| `src/run_analysis.py` | Modified | Capture `trader_investment_plan`, use dual extraction |
| `src/signal_processing.py` | Modified | Add `deduplicate_repeated_blocks()` |
| `tests/test_trade_params.py` | Modified | `TestMultiFormatStructuredBlock` class (3 tests) |
| `tests/test_dual_source.py` | Created | 6 dual-source extraction tests |
| `tests/test_dual_extraction_live.py` | Created | 2 real-data validation tests |
| `tests/test_signal_processing.py` | Modified | `TestDeduplicateRepeatedBlocks` class (5 tests) |
| `docs/TASK_009_REPORT.md` | Created | This report |
