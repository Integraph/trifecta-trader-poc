# Task 004 Report: Live Hybrid Validation & Comparison

## 1. Summary of Steps

| Step | Status | Notes |
|------|--------|-------|
| Step 1: Reasoning comparison | **Succeeded** | Both local models produce parseable decisions |
| Step 2: Enhance run_analysis.py | **Succeeded** | Added timing, quality scoring, enhanced JSON output |
| Step 3: All-cloud baseline run | **Succeeded** | After retry (initial run hit Anthropic billing limit) |
| Step 4: Hybrid Qwen run | **Succeeded** | Completed successfully on first attempt |
| Step 5: Create compare_runs.py | **Succeeded** | Comparison report generated |
| Step 6: Create test_compare_runs.py | **Succeeded** | 3 tests, all pass |
| Step 7: All tests pass | **Succeeded** | 43 passed |
| No vendor code modified | **Confirmed** | All changes in src/ and tests/ |

## 2. Reasoning Comparison Results (Step 1)

Isolated reasoning tests comparing models on bull/bear/trader prompts:

### Bull Case Quality

| Model | Words | Has Numbers | Quality |
|-------|-------|------------|---------|
| qwen2.5:14b | 511 | Yes | Detailed, well-structured with headers, specific data references |
| mistral-small:22b | 347 | Yes | Concise, organized, covers key points |

### Trader Decision Format

| Model | Decision | Format |
|-------|----------|--------|
| qwen2.5:14b | BUY | Used "Final Transaction Proposal: **BUY**" (slightly non-standard but parseable) |
| mistral-small:22b | HOLD | Used "FINAL TRANSACTION PROPOSAL: HOLD" (perfect format) |

**Quality Gate: PASSED** — Both models produce parseable decisions.

## 3. Full Pipeline Results — Side-by-Side Comparison

| Metric | all_cloud | hybrid_qwen |
|--------|-----------|-------------|
| **Decision** | SELL | HOLD |
| **Composite Score** | 10.0/10 | 8.6/10 |
| **Reasoning Depth** | 10/10 | 8/10 |
| **Data Grounding** | 10/10 | 10/10 |
| **Risk Awareness** | 10/10 | 6/10 |
| **Decision Consistent** | Yes | Yes |
| **Has Stop-Loss** | Yes | No |
| **Has Price Target** | Yes | Yes |
| **Has Position Sizing** | Yes | Yes |
| **Output Length** | 18,204 chars | 8,644 chars |
| **Elapsed Time** | 1,230.1s (~20.5 min) | 1,242.5s (~20.7 min) |
| **Signal Corrected** | No | No |

### Analysis of Differences

**Decision Divergence:** The all-cloud run decided SELL while hybrid_qwen decided HOLD. Both decisions are defensible given the same market data. The difference stems from the debater/researcher agents:

- **All-cloud (SELL):** Claude-powered debaters produced more aggressive arguments. The Conservative analyst's arguments were particularly detailed, and the Risk Manager synthesized them into a high-conviction SELL with extensive position management (covered calls, bear put spreads, phased exit strategy).

- **Hybrid Qwen (HOLD):** Qwen-powered debaters produced solid but shorter arguments. The Risk Manager (still Claude) received less detailed debate material and concluded with a more cautious HOLD recommendation with conditional buy triggers.

**Quality Gap:** The 1.4-point quality gap (10.0 vs 8.6) is primarily driven by:
1. **Reasoning depth** (10 vs 8): All-cloud output was 2.1x longer (18K vs 8.6K chars)
2. **Risk awareness** (10 vs 6): All-cloud included explicit stop-loss levels; hybrid mentioned triggers but the scorer didn't detect a stop-loss pattern

**Data Grounding:** Both scored 10/10 — the local model grounds arguments in specific numbers equally well.

**Timing:** Nearly identical (~20.5 min each). The hybrid run is not faster because the analyst agents (the longest-running phase with tool calls and multiple API roundtrips) still use Claude in both configs.

## 4. Comparison Report Output

```
================================================================================
HYBRID LLM QUALITY COMPARISON
================================================================================

Config                    Decision Consistent  Depth  Data   Risk   Score  Cost
--------------------------------------------------------------------------------
all_cloud                 SELL     yes         10     10     10     10.0   $0.0000
hybrid_qwen              HOLD     yes         8      10     6      8.6    $0.0000
--------------------------------------------------------------------------------

Best quality:  all_cloud (score: 10.0)
Lowest cost:   all_cloud ($0.0000)
```

(Cost estimation not yet implemented — both show $0.0000)

## 5. Log Snippets

### All-Cloud Run (first and last lines)

**Header:**
```
============================================================
Trifecta Trader POC - Analysis Run
============================================================
Ticker:    AAPL
Date:      2026-02-27
Mode:      HYBRID (all_cloud)
Deep LLM:  claude-sonnet-4-5-20250929
Quick LLM: claude-sonnet-4-5-20250929
============================================================

Hybrid routing: {'tool_calling': 'anthropic/claude-sonnet-4-5-20250929', 'reasoning_quick': 'anthropic/claude-sonnet-4-5-20250929', 'reasoning_deep': 'anthropic/claude-sonnet-4-5-20250929'}
```

**Footer:**
```
============================================================
DECISION: SELL

Quality Score: 10.0/10
  Reasoning depth:     10/10
  Data grounding:      10/10
  Risk awareness:      10/10
  Decision consistent: Yes
  Elapsed time:        1230.1s

Results saved to: results/AAPL/analysis_2026-02-27_all_cloud.json
============================================================
```

### Hybrid Qwen Run (first and last lines)

**Header:**
```
============================================================
Trifecta Trader POC - Analysis Run
============================================================
Ticker:    AAPL
Date:      2026-02-27
Mode:      HYBRID (hybrid_qwen)
Deep LLM:  claude-sonnet-4-5-20250929
Quick LLM: claude-sonnet-4-5-20250929
============================================================

Hybrid routing: {'tool_calling': 'anthropic/claude-sonnet-4-5-20250929', 'reasoning_quick': 'ollama/qwen2.5:14b', 'reasoning_deep': 'anthropic/claude-sonnet-4-5-20250929'}
```

**Footer:**
```
============================================================
DECISION: HOLD

Quality Score: 8.6/10
  Reasoning depth:     8/10
  Data grounding:      10/10
  Risk awareness:      6/10
  Decision consistent: Yes
  Elapsed time:        1242.5s

Results saved to: results/AAPL/analysis_2026-02-27_hybrid_qwen.json
============================================================
```

## 6. Errors Encountered and Resolution

### Anthropic Billing Error (Step 3, first attempt)
- **Error:** `anthropic.BadRequestError: Error code: 400 - 'Your credit balance is too low to access the Anthropic API'`
- **When:** ~13 minutes into all-cloud run, at the "Research Manager" step
- **Resolution:** User reset billing options; pipeline retried and completed successfully

No other errors encountered. Both pipeline runs completed cleanly on their successful attempts.

## 7. Git Log

```
412b670 Add live hybrid validation, comparison tooling, and pipeline results
4333351 Add hybrid LLM routing and quality comparison framework
a3f3269 Fix signal processing extraction and investigate looping issue
96bf243 Add Task 001 completion report
99bc853 Initial POC structure with TradingAgents submodule and analysis runner
```

## 8. Recommendation

### Which hybrid config should become default?

**`hybrid_qwen` is viable but not yet recommended as default.** Here's why:

**In favor of hybrid_qwen:**
- Successfully completed the full pipeline end-to-end
- Both decisions (HOLD) and signal extraction worked correctly
- Data grounding is identical to all-cloud (10/10)
- Estimated ~40-50% reduction in Anthropic API costs (6 of 12 agents run locally)
- No crashes, no format issues, no tool calling problems

**Against making it default now:**
- Quality gap is real: 8.6 vs 10.0 composite score
- The local model produces shorter, less detailed debate arguments (8.6K vs 18.2K chars)
- This flows through to the Risk Manager (Claude) which had less material to synthesize
- Different final decision (HOLD vs SELL) — while both are defensible, the divergence means the local model meaningfully changes pipeline behavior
- No stop-loss detection in the hybrid output (risk awareness 6 vs 10)

**Recommended next steps:**
1. Run 3-5 more hybrid_qwen analyses on different tickers/dates to see if the quality gap is consistent
2. Try `hybrid_mistral` to compare — mistral-small:22b may produce higher quality reasoning even though it can't do tool calling
3. Consider a `hybrid_qwen_deep` config where qwen handles debaters but Claude handles researchers AND judges (researchers produce the initial reports that feed the debate, so quality there matters)
4. Implement cost tracking to quantify the actual dollar savings per run

### Are we ready for Phase B (Alpaca paper trading)?

**Not yet.** Before integrating with Alpaca:
1. Need confidence that the hybrid config produces reliable decisions (more runs needed)
2. Need the position sizing / stop-loss output format to be consistent (the hybrid run missed explicit stop-loss format)
3. The signal processor correctly extracts decisions from both configs — that's validated
4. The infrastructure (HybridTradingGraph, quality scorer, comparison tools) is solid and ready

**Estimated readiness:** After 5-10 more validation runs across different market conditions and tickers, with consistent quality scores above 8.0, we can proceed to Phase B.
