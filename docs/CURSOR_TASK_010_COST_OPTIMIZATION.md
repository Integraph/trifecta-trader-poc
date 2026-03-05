# CURSOR TASK 010: Cost Optimization — Haiku Tier, Aggressive Local Routing & Response Caching

## Context

The hybrid LLM architecture achieves 35-40% cost savings over all-cloud, but the original goal was 80%. The gap exists because the heaviest token consumers — tool-calling agents (4 analysts making 20+ API calls per run) — remain on Claude Sonnet 4.5. This task introduces three optimization layers to close that gap.

**Current cost structure (hybrid_qwen_enhanced):**
- Tool-calling agents: Claude Sonnet 4.5 (most expensive, highest token volume)
- Reasoning quick: Ollama qwen2.5:14b (free)
- Reasoning deep (Risk Judge): Claude Sonnet 4.5 (expensive but low volume)

**Target: 60-70% cost savings** while maintaining quality ≥ 9.0/10.

---

## Step 1: Add Claude Haiku Tier for Tool-Calling Agents

**File:** `src/hybrid_llm.py`

Add a new model tier to `HybridLLMConfig`:

```python
# New field
tool_calling_model: str  # e.g., "anthropic/claude-haiku-4-5-20251001"
```

The tool-calling agents (Fundamentals Analyst, Sentiment Analyst, News Analyst, Technical Analyst) use `bind_tools()` to fetch data. They don't need deep reasoning — they need reliable function calling. Claude Haiku 4.5 handles tool calling well at ~1/10th the cost of Sonnet.

**Create two new configs:**

```python
"hybrid_haiku_tools": HybridLLMConfig(
    tool_calling="anthropic/claude-haiku-4-5-20251001",
    reasoning_quick="ollama/qwen2.5:14b",
    reasoning_deep="anthropic/claude-sonnet-4-5-20250929",
    enhance=True,
    enhance_style="financial_analysis",
    enhance_deep=True,
    enhance_deep_style="execution_params_only",
)

"hybrid_haiku_aggressive": HybridLLMConfig(
    tool_calling="anthropic/claude-haiku-4-5-20251001",
    reasoning_quick="ollama/qwen2.5:14b",
    reasoning_deep="ollama/qwen2.5:14b",  # Risk Judge also local
    enhance=True,
    enhance_style="financial_analysis",
    enhance_deep=True,
    enhance_deep_style="execution_params_only",
)
```

**Validation:** Run AAPL and TSLA with `hybrid_haiku_tools`. Compare quality scores against `hybrid_qwen_enhanced` baseline. Tool-calling quality should be nearly identical since Haiku handles function calling well.

---

## Step 2: Add Response Caching for Market Data

**New file:** `src/data_cache.py`

The tool-calling agents fetch the same market data (financial statements, price history, news) for the same ticker within a single analysis run. Multiple agents requesting overlapping data creates redundant API calls.

Implement a simple TTL-based cache:

```python
import hashlib
import json
import time
from pathlib import Path

class DataCache:
    """TTL-based file cache for market data API responses."""

    def __init__(self, cache_dir="cache", default_ttl=3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl  # 1 hour default

    def get(self, key: str) -> dict | None:
        """Return cached response if valid, None if expired/missing."""
        ...

    def set(self, key: str, data: dict, ttl: int = None):
        """Store response with TTL."""
        ...

    def key_for(self, tool_name: str, **params) -> str:
        """Generate cache key from tool name and parameters."""
        ...
```

**TTL guidelines:**
- Financial statements: 24 hours (don't change intraday)
- Price history (daily): 4 hours
- Current price: 5 minutes
- News: 30 minutes
- Sentiment data: 15 minutes

**Integration:** Wrap the tool-calling layer so that when an analyst agent calls a data-fetching tool, the cache is checked first. This is NOT LLM response caching — it's raw data caching that reduces redundant API calls to financial data providers.

**Important:** Cache should be per-ticker and cleared at the start of each new analysis run for a different ticker.

---

## Step 3: Add Token Usage Tracking and Cost Breakdown

**File:** `src/run_analysis.py` (update existing)

Currently `estimated_cost_usd` is tracked in `quality_scorer.py` but not broken down by agent or model tier. Add detailed cost tracking:

```python
cost_breakdown = {
    "total_usd": 0.0,
    "by_model": {
        "claude-sonnet-4-5": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "claude-haiku-4-5": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "ollama/qwen2.5:14b": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    },
    "by_agent": {
        "fundamentals_analyst": {"model": "...", "tokens": 0, "cost_usd": 0.0},
        "sentiment_analyst": {"model": "...", "tokens": 0, "cost_usd": 0.0},
        # ... etc
    },
    "cache_hits": 0,
    "cache_misses": 0,
    "savings_vs_all_cloud": "XX%",
}
```

Add this to the result JSON and print a summary after each run:

```
============================================================
COST BREAKDOWN
============================================================
  Claude Sonnet 4.5:  $0.12 (Risk Judge only)
  Claude Haiku 4.5:   $0.03 (4 tool-calling analysts)
  Ollama (local):     $0.00 (reasoning agents)
  Cache hits:         14/38 tool calls (37% saved)
  Total:              $0.15
  vs All-Cloud:       82% savings
============================================================
```

This gives visibility into where tokens are being spent and validates the optimization is working.

---

## Step 4: Benchmark All Configurations

Run a comparison matrix with AAPL and TSLA across these configs:

| Config | Tool-Calling | Reasoning | Deep | Expected Cost |
|--------|-------------|-----------|------|---------------|
| `all_cloud` (baseline) | Sonnet | Sonnet | Sonnet | $$$$ |
| `hybrid_qwen_enhanced` (current) | Sonnet | Qwen 14B | Sonnet | $$$ |
| `hybrid_haiku_tools` (new) | Haiku | Qwen 14B | Sonnet | $$ |
| `hybrid_haiku_aggressive` (new) | Haiku | Qwen 14B | Qwen 14B | $ |

For each run, record: quality score, decision, cost breakdown, elapsed time, and extraction success.

**Save results** to `results/<TICKER>/` with config name in the filename (this pattern is already established).

**Quality threshold:** Any config scoring below 9.0/10 on either ticker should be flagged but not discarded — document the quality delta for decision-making.

---

## Step 5: Update CLI and Defaults

**File:** `src/run_analysis.py`

- Add the new configs to the `--hybrid` CLI argument choices
- If `hybrid_haiku_tools` passes quality validation, make it the new default (replacing `hybrid_qwen_enhanced`)
- Add a `--cost-breakdown` flag that prints the detailed cost report (enabled by default, can be suppressed with `--no-cost-breakdown`)

---

## Step 6: Tests

**New test file:** `tests/test_cost_optimization.py`

- Test that `hybrid_haiku_tools` config is properly constructed
- Test that `hybrid_haiku_aggressive` config is properly constructed
- Test DataCache TTL expiry logic (mock time)
- Test DataCache key generation consistency
- Test cost breakdown calculation with mock token counts
- Test that cache is cleared between different tickers

**Run full test suite** and confirm no regressions: expect 127+ passed (current baseline).

---

## Step 7: Document Results

**New file:** `docs/TASK_010_REPORT.md`

Include:
- Quality comparison table across all 4 configs
- Cost comparison table with actual dollar amounts
- Decision on new default config
- Any quality degradation observed and mitigation
- Cache hit rates and their impact
- Recommendation for production config

---

## Exit Criteria

- [ ] `hybrid_haiku_tools` and `hybrid_haiku_aggressive` configs exist and work
- [ ] DataCache reduces redundant API calls (measurable cache hit rate)
- [ ] Cost breakdown prints after each run showing per-model and per-agent costs
- [ ] Benchmark results for all 4 configs on AAPL and TSLA documented
- [ ] Quality ≥ 9.0/10 maintained on the recommended config
- [ ] Cost savings ≥ 60% vs all-cloud on the recommended config
- [ ] All existing tests pass, new tests added
- [ ] TASK_010_REPORT.md written with full comparison data

---

## Important Notes

- **Zero vendor modifications.** All changes go in `src/`. Do not modify anything in `vendor/TradingAgents/`.
- **Do not change the existing `hybrid_qwen_enhanced` config.** It's the proven baseline. Add new configs alongside it.
- **Haiku model string:** `claude-haiku-4-5-20251001` — verify this is the correct model identifier in the Anthropic API.
- **Cache directory:** Use `cache/` at repo root, add to `.gitignore`.
- **Anthropic API pricing (as of early 2026):** Verify current Haiku vs Sonnet pricing and use actual rates in the cost calculation.
