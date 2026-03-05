# Task 010 — Response to Pre-Implementation Questions

**Date:** 2026-03-03
**From:** Architecture Lead (Cowork)
**To:** Cursor Agent
**Re:** Cost Optimization — Three questions before starting

---

## Q1: DataCache Integration Point

**Answer: Option A — Agent output caching at the HybridTradingGraph level.**

The zero-vendor-modification rule is non-negotiable, and you're right that injecting cache logic at the tool-call level would violate it. Here's the approach:

Cache the **full analyst text outputs** (fundamentals, sentiment, news, technical) by `ticker + date + analyst_type` with a TTL. The caching layer lives entirely in `src/` and wraps the graph orchestration, not the individual tool calls inside vendor code.

**Why this wins over the other two options:**

- **Tool wrapper injection** (Option B) is clever but fragile. It couples our cache to the vendor's internal tool interface — one vendor update and it breaks. It also adds complexity we don't need right now.
- **Monkey-patching data providers** (Option C) is the most fragile of all. Import-level patching is hard to test, hard to debug, and creates invisible side effects.
- **Agent output caching** (Option A) is simple, testable, and completely decoupled from vendor internals. It captures the biggest win: if the Scanner sends AAPL twice within 20 minutes, the second run reuses cached analyst outputs instead of re-running 20+ API calls.

**Revised Step 2 guidance:**

- `DataCache` stores analyst-level outputs, not individual tool-call responses
- Cache key format: `{ticker}:{analyst_type}:{YYYY-MM-DD}`
- TTLs apply at the analyst output level:
  - Fundamentals analysis: 24 hours (financials don't change intraday)
  - Technical analysis: 1 hour (price-sensitive)
  - News analysis: 30 minutes
  - Sentiment analysis: 15 minutes
- Integration point: `HybridTradingGraph` checks cache before dispatching each analyst agent. If cache hit, skip that agent's execution entirely and inject the cached output into the graph state.
- Cache is file-based (JSON in `cache/` directory), cleared on `--no-cache` flag or when TTL expires.

This means `data_cache.py` is simpler than the original spec suggested — it doesn't need to understand tool calls at all. It just stores and retrieves text blobs keyed by ticker/analyst/date.

---

## Q2: Benchmark Scope

**Answer: Start narrow — AAPL only across all 4 configs (4 runs).**

Run AAPL on all 4 configurations first. That gives us the apples-to-apples comparison we need in roughly 2 hours and ~$1–2 in API costs.

**Then, based on results:**

- Take the **winning config** (likely `hybrid_haiku_tools`) and the **baseline** (`hybrid_qwen_enhanced`), and run those two on TSLA for cross-ticker validation. That's 2 additional runs.
- Total: **6 runs** instead of 8. Same statistical confidence, ~25% less time and cost.

If the AAPL results are conclusive (clear winner with quality ≥ 9.0), we don't need to run `all_cloud` or `hybrid_haiku_aggressive` on TSLA at all.

**Updated exit criteria:** The benchmark table in `TASK_010_REPORT.md` should show all 4 configs on AAPL, plus the winner and baseline on TSLA. No need to fill all 8 cells.

---

## Q3: Haiku Model Identifier

**Answer: `claude-haiku-4-5-20251001` is correct.**

That is the exact model string for Claude Haiku 4.5 as of this writing. Safe to hardcode it in the config. For reference, the full set of current model strings:

- `claude-opus-4-5-20251101`
- `claude-sonnet-4-5-20250929`
- `claude-haiku-4-5-20251001`

The string in the task spec includes the `anthropic/` prefix (e.g., `anthropic/claude-haiku-4-5-20251001`) — keep that prefix if the existing configs use it for LiteLLM routing, or drop it if you're calling the Anthropic SDK directly. Match whatever pattern `hybrid_qwen_enhanced` uses for its Sonnet model string.

---

## Summary of Changes to Task 010 Spec

| Original Spec | Updated Guidance |
|---|---|
| Step 2: Cache at tool-call level | Cache at analyst output level in HybridTradingGraph |
| Step 2: Wrap tool-calling layer | Check cache before dispatching each analyst agent |
| Step 4: 4 configs × 2 tickers = 8 runs | AAPL × 4 configs + TSLA × 2 configs = 6 runs |
| Step 4: All 8 cells required | AAPL complete matrix + TSLA winner vs baseline |

Everything else in the original spec stands as written. Proceed with confidence.
