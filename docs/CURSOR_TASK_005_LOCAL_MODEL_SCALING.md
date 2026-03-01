# Cursor Task 005: Local Model Scaling Experiment

## Objective
Systematically test increasingly powerful local Ollama models to find the best quality/performance tradeoff on the user's MacBook Pro M3 Max (128GB RAM). The goal is to close the quality gap between all-cloud and hybrid configurations before committing to full pipeline runs.

## Context
- Task 004 showed a quality gap: all-cloud scored 10.0 vs hybrid_qwen (14B) at 8.6
- The gap is driven by reasoning depth (18K vs 8.6K chars) and risk awareness formatting
- The M3 Max with 128GB RAM is underutilized — Qwen 14B only uses ~9GB
- We need to find the local model that produces Claude-quality debate material
- This task uses ONLY the isolated reasoning tests (a few cents per run, not full pipeline)

## Important Rules
- **DO NOT modify files in `vendor/TradingAgents/` directly**
- **DO NOT run the full pipeline** — use isolated reasoning tests only
- All new code goes in `src/` and `tests/`
- Pull models one at a time to avoid filling disk unnecessarily
- Record inference speed (tokens/sec) for each model — speed matters for a 12-agent pipeline

---

## Step 1: Pull and Benchmark Larger Models

### Goal
Download progressively larger models and measure raw inference speed on the M3 Max.

### Models to Test (in order of increasing size)

| Model | Size (Q4) | Expected Speed | Notes |
|-------|-----------|---------------|-------|
| `qwen2.5:14b` | ~9 GB | ~35-45 t/s | Already installed — baseline |
| `qwen2.5:32b` | ~20 GB | ~25-35 t/s | Sweet spot candidate |
| `qwen2.5:72b` | ~43 GB | ~10-18 t/s | High quality, may be slower |
| `llama3.3:70b` | ~43 GB | ~10-18 t/s | Meta's latest, strong reasoning |
| `command-r:35b` | ~20 GB | ~25-35 t/s | Cohere's model, good at structured output |

**Note:** The 72B/70B models will be ~43GB each. With 128GB RAM, the machine can run them but only one at a time. Don't pull both 70B+ models simultaneously — test one, then swap.

### Commands

Pull models one at a time, benchmark each immediately, then move on:

```bash
# Model 1: qwen2.5:32b (the prime candidate)
ollama pull qwen2.5:32b

# Quick benchmark: measure tokens/sec
echo "Explain the bull case for Apple stock in exactly 200 words, citing specific financial metrics." | \
    ollama run qwen2.5:32b --verbose 2>&1 | tail -5

# Model 2: qwen2.5:72b (if 32b quality isn't sufficient)
ollama pull qwen2.5:72b

echo "Explain the bull case for Apple stock in exactly 200 words, citing specific financial metrics." | \
    ollama run qwen2.5:72b --verbose 2>&1 | tail -5
```

**Important:** If 32B closes the quality gap (Step 2 will determine this), you can skip pulling the larger models. Don't waste bandwidth/disk.

### What to Record

For each model, capture:
- Download size
- `ollama run <model> --verbose` output showing eval rate (tokens/sec)
- Any warnings about memory pressure

---

## Step 2: Isolated Reasoning Quality Test

### Goal
Run the same bull/bear/trader prompts through each model and compare output quality. This reuses the test infrastructure from Task 003.

### Implementation

Update `tests/test_reasoning_comparison.py` to add the new models. Add these to `_create_model_configs()`:

```python
def _create_model_configs():
    """Define models to compare."""
    configs = []

    if _ollama_available():
        # Existing models
        configs.append({
            "name": "qwen2.5:14b",
            "provider": "ollama",
            "model": "qwen2.5:14b",
        })
        configs.append({
            "name": "mistral-small:22b",
            "provider": "ollama",
            "model": "mistral-small:22b",
        })

        # New models — only include if actually pulled
        for model_name in ["qwen2.5:32b", "qwen2.5:72b", "llama3.3:70b", "command-r:35b"]:
            if _model_available(model_name):
                configs.append({
                    "name": model_name,
                    "provider": "ollama",
                    "model": model_name,
                })

    if os.environ.get("ANTHROPIC_API_KEY"):
        configs.append({
            "name": "claude-sonnet-4.5",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        })

    return configs


def _model_available(model_name: str) -> bool:
    """Check if a specific Ollama model is pulled."""
    try:
        import urllib.request
        import json
        response = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        data = json.loads(response.read())
        available = [m["name"] for m in data.get("models", [])]
        # Handle tag matching: "qwen2.5:32b" should match "qwen2.5:32b" or "qwen2.5:32b-instruct-..."
        return any(model_name in m for m in available)
    except Exception:
        return False
```

### Add Timing and Detail Metrics

Add a new test function that captures more detailed quality metrics for comparison:

```python
@skip_unless_comparison
def test_detailed_quality_comparison(comparison_results):
    """Detailed quality comparison with scoring across all models."""
    from src.quality_scorer import score_pipeline_output

    print("\n" + "=" * 90)
    print("DETAILED QUALITY COMPARISON — ALL MODELS")
    print("=" * 90)

    header = f"{'Model':<22} {'Role':<8} {'Words':<7} {'Numbers':<8} {'Decision':<10} {'StopLoss':<9} {'Target':<8}"
    print(header)
    print("-" * 90)

    for model_name, responses in comparison_results.items():
        for prompt_name in ["bull", "bear", "trader"]:
            response = responses.get(prompt_name, "NOT RUN")
            if response.startswith("ERROR"):
                print(f"{model_name:<22} {prompt_name:<8} ERROR: {response[:50]}")
                continue

            word_count = len(response.split())
            import re
            numbers = re.findall(r'\$[\d,.]+|\d+\.?\d*%', response)
            has_stop = bool(re.search(r'stop.?loss', response, re.IGNORECASE))
            has_target = bool(re.search(r'(?:price\s+)?target|upside|downside', response, re.IGNORECASE))

            decision = ""
            if prompt_name == "trader":
                from src.signal_processing import extract_decision
                decision = extract_decision(response)

            print(
                f"{model_name:<22} {prompt_name:<8} {word_count:<7} "
                f"{len(numbers):<8} {decision:<10} "
                f"{'Yes' if has_stop else 'No':<9} {'Yes' if has_target else 'No':<8}"
            )

    print("-" * 90)

    # Full trader output comparison
    print("\n" + "=" * 90)
    print("TRADER OUTPUT — FULL TEXT COMPARISON")
    print("=" * 90)

    for model_name, responses in comparison_results.items():
        trader_response = responses.get("trader", "NOT RUN")
        if not trader_response.startswith("ERROR"):
            print(f"\n{'='*40} {model_name} {'='*40}")
            print(f"Word count: {len(trader_response.split())}")
            print(f"Char count: {len(trader_response)}")
            # Print last 500 chars (where the decision and risk params should be)
            print(f"\n--- Last 500 chars ---")
            print(trader_response[-500:])
            print()
```

### Add Timing to Each Model Call

Update `_invoke_model()` to return timing:

```python
def _invoke_model(provider: str, model: str, prompt: str) -> tuple:
    """Invoke a model and return (response_text, elapsed_seconds)."""
    import time

    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    start = time.time()
    result = llm.invoke(prompt)
    elapsed = time.time() - start

    return result.content, elapsed
```

Update the `comparison_results` fixture accordingly:

```python
@pytest.fixture(scope="module")
def comparison_results(request):
    """Run all comparisons and collect results."""
    if not request.config.getoption("--run-comparison"):
        pytest.skip("Requires --run-comparison flag")

    configs = _create_model_configs()
    if not configs:
        pytest.skip("No models available for comparison")

    results = {}

    for config in configs:
        name = config["name"]
        results[name] = {}

        for prompt_name, prompt in [
            ("bull", BULL_PROMPT),
            ("bear", BEAR_PROMPT),
            ("trader", TRADER_PROMPT),
        ]:
            try:
                response, elapsed = _invoke_model(config["provider"], config["model"], prompt)
                results[name][prompt_name] = response
                results[name][f"{prompt_name}_elapsed"] = elapsed
                print(f"  {name} / {prompt_name}: {elapsed:.1f}s, {len(response.split())} words")
            except Exception as e:
                results[name][prompt_name] = f"ERROR: {e}"
                results[name][f"{prompt_name}_elapsed"] = 0

    return results
```

Update existing tests to use the new tuple return if needed, or keep `_invoke_model` backward compatible by checking the callers. The simplest approach: have `_invoke_model` return just the string (as before) and track timing separately inside the fixture.

### Run

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

# Run comparison with all available local models + Claude
pytest tests/test_reasoning_comparison.py -v -s --run-comparison 2>&1 | tee results/model_scaling_comparison.txt
```

### What to Record

For each model, capture in the report:
- Word count for bull, bear, and trader prompts
- Number of data points cited
- Whether trader output includes stop-loss, price target, position sizing
- Whether the decision is parseable
- Time per prompt
- Overall quality impression vs Claude

---

## Step 3: Tool Calling Test for New Models

### Goal
Test whether larger models handle tool calling (needed if we want to try `hybrid_aggressive_*` configs where local models run analysts too).

### Implementation

Update `tests/test_local_tool_calling.py` to include new models:

```python
OLLAMA_MODELS = [
    "qwen2.5:14b",
    "mistral-small:22b",
]

# Dynamically add larger models if available
import urllib.request, json
try:
    response = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    data = json.loads(response.read())
    available = [m["name"] for m in data.get("models", [])]
    for model in ["qwen2.5:32b", "qwen2.5:72b", "llama3.3:70b", "command-r:35b"]:
        if any(model in m for m in available):
            OLLAMA_MODELS.append(model)
except Exception:
    pass
```

### Run

```bash
pytest tests/test_local_tool_calling.py -v -s 2>&1 | tee results/tool_calling_scaling.txt
```

---

## Step 4: Update Hybrid Configs for Best Model

### Goal
Once Step 2 identifies the best local model, add new hybrid configs that use it.

### Implementation

Update `src/hybrid_llm.py` to add configs for the winning model. For example, if `qwen2.5:32b` wins:

```python
# Add to CONFIGS dict:

    "hybrid_qwen32": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:32b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    "hybrid_aggressive_qwen32": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:32b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="qwen2.5:32b",
    ),
```

Also update `src/run_analysis.py` to include the new config names in the `--hybrid` choices list.

Update `tests/test_hybrid_llm.py` to test the new configs exist.

### Naming Convention
Use the pattern `hybrid_<model_family><size>` for configs:
- `hybrid_qwen32` — Qwen 2.5 32B for quick reasoning
- `hybrid_qwen72` — Qwen 2.5 72B for quick reasoning
- `hybrid_llama70` — Llama 3.3 70B for quick reasoning

---

## Step 5: Scaling Summary Report

### Goal
Create a comprehensive comparison that shows quality and speed for each model size.

### Implementation

Create `src/scaling_report.py`:

```python
"""
Generate a scaling report comparing local model sizes.

Usage:
    python -m src.scaling_report results/model_scaling_comparison.txt
"""

import sys
import re
from pathlib import Path


def parse_comparison_output(filepath: str) -> dict:
    """Parse the test output to extract metrics per model."""
    content = Path(filepath).read_text()
    # Parse the detailed comparison table and timing data
    # This will depend on the exact output format from the tests
    # Return structured data for reporting
    return {"raw": content}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.scaling_report <comparison_output.txt>")
        sys.exit(1)

    data = parse_comparison_output(sys.argv[1])

    print("=" * 80)
    print("LOCAL MODEL SCALING REPORT")
    print("=" * 80)
    print()
    print("This report summarizes quality and performance across local model sizes,")
    print("compared against Claude Sonnet 4.5 as the cloud baseline.")
    print()
    print("See results/model_scaling_comparison.txt for full test output.")
    print()

    # Print the raw data for now — the test output contains the table
    print(data["raw"])


if __name__ == "__main__":
    main()
```

The real value here is the test output itself. The `test_detailed_quality_comparison` function produces the comparison table we need.

---

## Step 6: Final Verification

### Run All Tests

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
pytest tests/ -v --ignore=tests/test_reasoning_comparison.py 2>&1 | tee results/task_005_unit_tests.txt
```

(We ignore the reasoning comparison tests in the general run since they require `--run-comparison` and make API calls.)

### Commit

```bash
git add .
git commit -m "Add local model scaling experiment with quality comparison"
git push
```

---

## Execution Strategy (IMPORTANT)

This task is designed to be run **incrementally**. Don't pull all models at once.

**Round 1: qwen2.5:32b**
1. Pull `qwen2.5:32b`
2. Run reasoning comparison (Step 2)
3. Run tool calling test (Step 3)
4. Check results: does 32B close the quality gap significantly?

**If 32B is close enough to Claude (quality score ≥ 9.0 on trader prompt):**
- Add hybrid configs for 32B (Step 4)
- Skip larger models
- Write report

**If 32B still has a meaningful gap (quality score < 9.0):**
- Pull `qwen2.5:72b` OR `llama3.3:70b` (pick one based on 32B results)
- Re-run reasoning comparison
- If 70B+ closes the gap, add hybrid configs
- If 70B+ still doesn't close the gap, document the ceiling

**Round 2 (only if needed): 70B+ class**
- Only pull one 70B model at a time (~43GB each)
- Test, evaluate, then decide whether to try the other
- Record inference speed — if it's below 8 t/s, flag it as impractical for a 12-agent pipeline

---

## Verification Checklist

- [ ] At least `qwen2.5:32b` pulled and benchmarked
- [ ] Reasoning comparison run with all available models + Claude baseline
- [ ] Tool calling test run for new models
- [ ] Quality comparison table generated (word count, data grounding, decision format, timing)
- [ ] Full trader output compared across models (last 500 chars showing decision and risk params)
- [ ] New hybrid configs added for best-performing local model
- [ ] `run_analysis.py` updated with new config choices
- [ ] All unit tests pass
- [ ] Changes committed and pushed

---

## Report

After completing all steps, create `docs/TASK_005_REPORT.md` containing:

1. **Model benchmark table:**
   | Model | Size | Speed (t/s) | Tool Calling | Bull Words | Bear Words | Trader Words | Decision | Stop-Loss | Target | Sizing |

2. **Quality ranking** — which local model comes closest to Claude quality?

3. **Speed vs quality tradeoff** — is the 70B model worth the speed hit?

4. **Recommended hybrid config** — which model should we use for quick reasoning agents?

5. **Ceiling assessment** — can ANY local model match Claude quality, or is there a permanent gap?

6. **Full test output** — the detailed comparison table and trader output excerpts

7. **Git log** — `git log --oneline`

8. **Next steps** — ready for a full pipeline run with the best local model?
