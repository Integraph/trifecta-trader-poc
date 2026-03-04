# CURSOR TASK 011: Qwen 3.5 Local Model Benchmark

## Context

Qwen 3.5 was released on February 16, 2026 — a generational leap over our current Qwen 2.5:14b. The new lineup includes models that could significantly improve reasoning quality at the local inference tier while maintaining or increasing throughput.

**Current local model:** `ollama/qwen2.5:14b` — 14B dense, ~35 t/s on M3 Max 128GB, 9.1-9.4/10 quality in the `hybrid_haiku_tools` pipeline.

**Qwen 3.5 candidates to test:**

| Model | Architecture | Active Params | Total Params | Expected Speed | Why Test It |
|---|---|---|---|---|---|
| `qwen3.5:27b` | Dense | 27B | 27B | ~15-20 t/s | Highest reasoning density. Scores 72.4 on SWE-bench (matches GPT-5 mini). Major quality upgrade potential. |
| `qwen3.5:35b-a3b` | MoE | 3B | 35B | ~35-50 t/s | Only 3B active per token = fast. 35B total knowledge base. Speed of a 3B with the memory of a 35B. |
| `qwen3.5:9b` | Dense | 9B | 9B | ~50-60 t/s | Smallest in new lineup. Benchmarks show it beating much larger previous-gen models. Fastest option. |

**Goal:** Determine if any Qwen 3.5 model can replace Qwen 2.5:14b in our pipeline with equal or better quality, and characterize the speed/quality tradeoffs.

---

## Prerequisites

Before starting any benchmark runs:

1. **Update Ollama** to v0.17.5 or later (GGUF support for Qwen 3.5 was fixed on March 2, 2026)
2. **Pull the models:**
   ```bash
   ollama pull qwen3.5:27b
   ollama pull qwen3.5:35b-a3b
   ollama pull qwen3.5:9b
   ```
3. **Verify each model loads and responds:**
   ```bash
   ollama run qwen3.5:27b "What is the P/E ratio of a stock trading at $50 with EPS of $2.50?"
   ollama run qwen3.5:35b-a3b "What is the P/E ratio of a stock trading at $50 with EPS of $2.50?"
   ollama run qwen3.5:9b "What is the P/E ratio of a stock trading at $50 with EPS of $2.50?"
   ```
   Expected answer: 20. If any model fails to load or gives nonsensical output, skip it and document the issue.

4. **Clear the analyst cache** before starting benchmarks:
   ```bash
   rm -rf cache/
   ```

---

## Step 1: Add Qwen 3.5 Configs

**File:** `src/hybrid_llm.py`

Add three new configs that mirror `hybrid_haiku_tools` (our current best) but swap the local reasoning model:

```python
"hybrid_haiku_qwen35_27b": HybridLLMConfig(
    tool_calling="anthropic/claude-haiku-4-5-20251001",
    reasoning_quick="ollama/qwen3.5:27b",
    reasoning_deep="anthropic/claude-sonnet-4-5-20250929",
    enhance=True,
    enhance_style="financial_analysis",
    enhance_deep=True,
    enhance_deep_style="execution_params_only",
)

"hybrid_haiku_qwen35_35b": HybridLLMConfig(
    tool_calling="anthropic/claude-haiku-4-5-20251001",
    reasoning_quick="ollama/qwen3.5:35b-a3b",
    reasoning_deep="anthropic/claude-sonnet-4-5-20250929",
    enhance=True,
    enhance_style="financial_analysis",
    enhance_deep=True,
    enhance_deep_style="execution_params_only",
)

"hybrid_haiku_qwen35_9b": HybridLLMConfig(
    tool_calling="anthropic/claude-haiku-4-5-20251001",
    reasoning_quick="ollama/qwen3.5:9b",
    reasoning_deep="anthropic/claude-sonnet-4-5-20250929",
    enhance=True,
    enhance_style="financial_analysis",
    enhance_deep=True,
    enhance_deep_style="execution_params_only",
)
```

Also add these to the `--hybrid` CLI argument choices in `src/run_analysis.py`.

**Important:** Do NOT modify or remove the existing `hybrid_haiku_tools` config. It is the current production baseline.

---

## Step 2: Measure Inference Speed

Before running full pipeline benchmarks, measure raw inference speed for each model on a representative financial reasoning prompt. This isolates model speed from network/API latency.

Create a simple speed test script at `scripts/measure_ollama_speed.py`:

```python
"""Measure tokens/second for each Ollama model on a financial reasoning prompt."""

import time
import requests

MODELS = [
    "qwen2.5:14b",       # Current baseline
    "qwen3.5:27b",       # Dense 27B
    "qwen3.5:35b-a3b",   # MoE 35B
    "qwen3.5:9b",        # Dense 9B
]

PROMPT = """Analyze the following financial data and provide a risk assessment:
- Revenue: $4.2B (up 12% YoY)
- Net Income: $380M (down 5% YoY)
- Debt/Equity: 1.8 (industry avg: 1.2)
- Free Cash Flow: $520M
- P/E Ratio: 28.5 (industry avg: 22)
- Short Interest: 8.2%

Provide a detailed risk assessment covering: financial health, valuation risk,
market sentiment, and an overall risk rating from 1-10."""

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    start = time.time()
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": PROMPT,
        "stream": False,
    })
    elapsed = time.time() - start

    data = response.json()
    total_tokens = data.get("eval_count", 0)
    speed = total_tokens / elapsed if elapsed > 0 else 0

    print(f"Tokens generated: {total_tokens}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Speed: {speed:.1f} t/s")
    print(f"Response quality (first 200 chars): {data.get('response', '')[:200]}")
```

Run it and record the results:
```bash
python scripts/measure_ollama_speed.py
```

---

## Step 3: Run Pipeline Benchmarks

Use AAPL as the benchmark ticker, consistent with Task 010. Clear cache between each run to get fresh (uncached) results.

**Run order:**

1. `hybrid_haiku_tools` (baseline — Qwen 2.5:14b) — fresh run, no cache
2. Clear cache
3. `hybrid_haiku_qwen35_27b` — fresh run, no cache
4. Clear cache
5. `hybrid_haiku_qwen35_35b` — fresh run, no cache
6. Clear cache
7. `hybrid_haiku_qwen35_9b` — fresh run, no cache

```bash
# Baseline
rm -rf cache/
python src/run_analysis.py AAPL --hybrid hybrid_haiku_tools

# Qwen 3.5 27B
rm -rf cache/
python src/run_analysis.py AAPL --hybrid hybrid_haiku_qwen35_27b

# Qwen 3.5 35B-A3B (MoE)
rm -rf cache/
python src/run_analysis.py AAPL --hybrid hybrid_haiku_qwen35_35b

# Qwen 3.5 9B
rm -rf cache/
python src/run_analysis.py AAPL --hybrid hybrid_haiku_qwen35_9b
```

**For each run, record:**
- Decision (BUY/SELL/HOLD)
- Quality score
- Cost breakdown (from Task 010 output)
- Total elapsed time
- Extraction success (Stop-loss, Target, Actionable)
- Any errors or anomalies

**If a Qwen 3.5 model clearly wins** (quality ≥ 9.0 AND faster than baseline), run it on TSLA for cross-ticker validation. Otherwise, skip the TSLA run.

---

## Step 4: Compatibility Check

Qwen 3.5 has a different tokenizer and architecture than Qwen 2.5. Verify:

1. **Ollama model loads without errors** — check `ollama ps` shows the model running
2. **LiteLLM routing works** — the `ollama/qwen3.5:*` model strings route correctly through LiteLLM
3. **Enhanced prompts work** — the `financial_analysis` enhancement style produces valid output (no garbled tokens, no truncation)
4. **Token counting works** — the `TokenUsageCallback` from Task 010 captures tokens correctly for the new models

Document any compatibility issues. If a model fails to work with our pipeline, document the error and skip the benchmark for that model. Do not spend time debugging Ollama/LiteLLM integration issues — document them and move on.

---

## Step 5: Tests

**File:** `tests/test_cost_optimization.py` (extend existing)

Add tests:
- Verify all three new Qwen 3.5 configs are properly constructed
- Verify model strings are valid format
- Verify the configs use the same tool-calling and deep reasoning models as `hybrid_haiku_tools`

**Run full test suite** and confirm no regressions.

---

## Step 6: Document Results

**New file:** `docs/TASK_011_REPORT.md`

Include:

### Speed Comparison Table
| Model | Raw Speed (t/s) | Pipeline Time (AAPL) |
|---|---|---|

### Quality Comparison Table
| Config | Decision | Quality | Cost | Time | Notes |
|---|---|---|---|---|---|

### Recommendation
One of:
- **Upgrade to [model]** — if a Qwen 3.5 model matches or beats the baseline on quality AND speed
- **Stay on Qwen 2.5:14b** — if no Qwen 3.5 model meets both criteria
- **Conditional upgrade** — if a model improves quality significantly but sacrifices speed (or vice versa), present the tradeoff for user decision

### Known Issues
Document any Ollama compatibility problems, token counting gaps, or other anomalies.

---

## Exit Criteria

- [ ] Ollama updated to v0.17.5+
- [ ] All three Qwen 3.5 models pulled and verified
- [ ] Raw inference speed measured for all 4 models (baseline + 3 new)
- [ ] Pipeline benchmark on AAPL for all 4 configs (fresh, no cache)
- [ ] Cross-ticker TSLA validation for winner (if one emerges)
- [ ] All existing tests pass, new tests added
- [ ] TASK_011_REPORT.md written with full comparison and recommendation
- [ ] No vendor modifications

---

## Important Notes

- **Zero vendor modifications.** All changes go in `src/` and `scripts/`. Do not modify anything in `vendor/TradingAgents/`.
- **Do not change `hybrid_haiku_tools`.** It is the current production config and the benchmark baseline. Add new configs alongside it.
- **Clear cache between benchmark runs.** Each run must be fresh to get comparable results. Use `rm -rf cache/` before each run.
- **If Ollama has issues with Qwen 3.5**, document the problem and skip that model. GGUF support was just fixed (v0.17.5, March 2) so there may be edge cases. Do not spend more than 30 minutes troubleshooting any single model's Ollama integration.
- **Watch for the 27B model's memory usage.** At Q4_K_M quantization, Qwen 3.5:27b may use ~18-20GB. The M3 Max 128GB has plenty of headroom, but verify Ollama doesn't swap to disk.
- **The MoE model (35B-A3B) may behave differently** from dense models in our pipeline. If it produces responses with unusual formatting or token patterns, note this in the report.
