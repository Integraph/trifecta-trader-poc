# Cursor Task 004: Live Hybrid Validation & Comparison

## Objective
Validate the hybrid pipeline end-to-end by running the isolated reasoning comparison tests, then executing two full pipeline runs (all-cloud baseline vs hybrid_qwen), scoring both, and producing a comparison report.

## Context
- Task 003 built all the hybrid infrastructure: `HybridLLMConfig`, `HybridGraphSetup`, `HybridTradingGraph`, quality scorer, and isolated reasoning tests
- **Key finding from Task 003:** `qwen2.5:14b` handles tool calling AND reasoning; `mistral-small:22b` is reasoning-only
- The recommended first hybrid config is `hybrid_qwen`: Claude for analysts (tool calling) and judges (deep reasoning), Qwen 2.5 14B for debaters/researchers/trader (quick reasoning)
- 48 tests currently pass (44 passed + 2 expected failures + 2 skips)
- The pipeline has been run before with all-Claude and produced correct results (HOLD for AAPL with risk parameters)

## Important Rules
- **DO NOT modify files in `vendor/TradingAgents/` directly** — keep upstream clean
- All new code goes in `src/` and `tests/`
- This task **DOES** involve running the full pipeline (2 runs) — budget ~40 minutes total
- Make sure `ANTHROPIC_API_KEY` is set in your environment before running
- Make sure Ollama is running with `qwen2.5:14b` available before running

---

## Step 1: Run Isolated Reasoning Comparison

### Goal
Run the side-by-side reasoning comparison tests from Task 003 to get a quick quality signal before committing to full pipeline runs.

### Commands

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

# Verify Ollama is running and models are available
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; tags=json.load(sys.stdin); [print(m['name']) for m in tags['models']]"

# Verify Anthropic key is set
echo "ANTHROPIC_API_KEY is ${ANTHROPIC_API_KEY:+set}"

# Run the reasoning comparison (costs a few cents for Claude call)
pytest tests/test_reasoning_comparison.py -v -s --run-comparison 2>&1 | tee results/reasoning_comparison_output.txt
```

### What to Record
- Bull case: word count, data references, and overall coherence for each model
- Bear case: same metrics
- Trader decision: does each model produce a parseable FINAL TRANSACTION PROPOSAL?
- Any model that fails to produce a valid decision — flag it

### Quality Gate
If qwen2.5:14b fails the trader decision format test (can't produce a parseable FINAL TRANSACTION PROPOSAL), **STOP HERE** and document the failure. The hybrid pipeline won't work if the local model can't produce properly formatted decisions for the debater/researcher roles.

If it passes, continue to Step 2.

---

## Step 2: Enhance Result Capture

### Goal
Before running full pipelines, improve `run_analysis.py` to capture timing, cost estimation, and the full `final_trade_decision` text for quality scoring.

### Implementation

Update `src/run_analysis.py` to capture additional data:

```python
import time

# In run_analysis(), add timing around the propagate call:
start_time = time.time()
final_state, upstream_decision = ta.propagate(ticker, trade_date)
elapsed_seconds = time.time() - start_time

# Capture the full trade decision text for quality scoring
final_trade_text = final_state.get("final_trade_decision", "")

# Save enhanced results
result = {
    "ticker": ticker,
    "trade_date": trade_date,
    "provider": provider,
    "hybrid_config": hybrid,
    "deep_model": config["deep_think_llm"],
    "quick_model": config["quick_think_llm"],
    "decision": decision,
    "upstream_decision": upstream_decision,
    "final_trade_decision_text": final_trade_text,
    "elapsed_seconds": round(elapsed_seconds, 1),
    "run_timestamp": datetime.now().isoformat(),
}

# Save with hybrid config name in filename for disambiguation
config_label = hybrid if hybrid else provider
result_file = results_dir / f"analysis_{trade_date}_{config_label}.json"
```

Also add a post-run quality score to the output:

```python
from src.quality_scorer import score_pipeline_output

score = score_pipeline_output(
    config_name=config_label,
    ticker=ticker,
    trade_date=trade_date,
    final_trade_decision=final_trade_text,
    extracted_decision=decision,
)

result["quality_score"] = {
    "composite": score.composite_score,
    "reasoning_depth": score.reasoning_depth,
    "data_grounding": score.data_grounding,
    "risk_awareness": score.risk_awareness,
    "decision_consistent": score.decision_consistent,
    "has_stop_loss": score.has_stop_loss,
    "has_price_target": score.has_price_target,
    "has_position_sizing": score.has_position_sizing,
}

print(f"\nQuality Score: {score.composite_score:.1f}/10")
print(f"  Reasoning depth:   {score.reasoning_depth}/10")
print(f"  Data grounding:    {score.data_grounding}/10")
print(f"  Risk awareness:    {score.risk_awareness}/10")
print(f"  Decision consistent: {'Yes' if score.decision_consistent else 'No'}")
print(f"  Elapsed time:      {elapsed_seconds:.1f}s")
```

### Tests

Verify existing tests still pass after the changes:

```bash
pytest tests/test_config.py tests/test_signal_processing.py tests/test_quality_scorer.py tests/test_hybrid_llm.py -v
```

---

## Step 3: Run All-Cloud Baseline

### Goal
Run the full pipeline with `all_cloud` config to establish a quality/cost baseline.

### Command

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

python -m src.run_analysis \
    --ticker AAPL \
    --date 2026-02-27 \
    --hybrid all_cloud \
    --no-debug \
    2>&1 | tee results/AAPL/run_all_cloud.log
```

### What to Record
- Decision (BUY/HOLD/SELL)
- Quality score (composite + breakdown)
- Elapsed time in seconds
- Whether the decision was corrected by our signal processor
- Full log saved to `results/AAPL/run_all_cloud.log`

### Troubleshooting
- If you get `ANTHROPIC_API_KEY` errors, run `export ANTHROPIC_API_KEY=<key>` or check `.env`
- If you get Redis errors, run `brew services start redis`
- If the pipeline crashes, capture the full traceback and save it to the report

---

## Step 4: Run Hybrid Qwen

### Goal
Run the same analysis with `hybrid_qwen` config — Claude for analysts and judges, Qwen 2.5 14B for all quick reasoning agents.

### Command

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

python -m src.run_analysis \
    --ticker AAPL \
    --date 2026-02-27 \
    --hybrid hybrid_qwen \
    --no-debug \
    2>&1 | tee results/AAPL/run_hybrid_qwen.log
```

### What to Record
- Same metrics as Step 3
- Any errors related to Ollama or the hybrid routing
- Whether the local model produced coherent debate arguments
- Full log saved to `results/AAPL/run_hybrid_qwen.log`

### Troubleshooting
- If Ollama models fail mid-pipeline, check `ollama ps` to see if the model is loaded
- If you get OOM or slowness, check `ollama run qwen2.5:14b` works independently first
- If HybridTradingGraph crashes, capture the traceback — likely a graph wiring issue in `hybrid_graph.py`

---

## Step 5: Generate Comparison Report

### Goal
Load both result JSONs and produce a side-by-side comparison using the quality scorer.

### Implementation

Create `src/compare_runs.py`:

```python
"""
Compare results from multiple pipeline runs.

Usage:
    python -m src.compare_runs results/AAPL/analysis_2026-02-27_all_cloud.json results/AAPL/analysis_2026-02-27_hybrid_qwen.json
"""

import json
import sys
from pathlib import Path
from src.quality_scorer import QualityScore, compare_scores


def load_result(filepath: str) -> dict:
    """Load a pipeline result JSON."""
    with open(filepath) as f:
        return json.load(f)


def result_to_score(result: dict) -> QualityScore:
    """Convert a pipeline result dict to a QualityScore."""
    qs = result.get("quality_score", {})

    return QualityScore(
        config_name=result.get("hybrid_config") or result.get("provider", "unknown"),
        ticker=result.get("ticker", ""),
        trade_date=result.get("trade_date", ""),
        decision=result.get("decision", "UNKNOWN"),
        decision_consistent=qs.get("decision_consistent", True),
        reasoning_depth=qs.get("reasoning_depth", 0),
        data_grounding=qs.get("data_grounding", 0),
        risk_awareness=qs.get("risk_awareness", 0),
        estimated_cost_usd=0.0,  # TODO: estimate from token counts
        full_output_length=len(result.get("final_trade_decision_text", "")),
        has_stop_loss=qs.get("has_stop_loss", False),
        has_price_target=qs.get("has_price_target", False),
        has_position_sizing=qs.get("has_position_sizing", False),
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.compare_runs <result1.json> [result2.json ...]")
        sys.exit(1)

    scores = []
    for filepath in sys.argv[1:]:
        result = load_result(filepath)
        score = result_to_score(result)
        scores.append(score)

        # Print individual run summary
        elapsed = result.get("elapsed_seconds", "N/A")
        print(f"\n{'='*60}")
        print(f"Run: {score.config_name}")
        print(f"  Decision: {score.decision}")
        print(f"  Elapsed: {elapsed}s")
        print(f"  Output length: {score.full_output_length} chars")
        print(f"  Quality score: {score.composite_score:.1f}/10")
        print(f"{'='*60}")

    if len(scores) > 1:
        print("\n")
        print(compare_scores(scores))

        # Save comparison report
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / "comparison_report.txt"
        with open(report_file, "w") as f:
            f.write(compare_scores(scores))
        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
```

### Run

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc

python -m src.compare_runs \
    results/AAPL/analysis_2026-02-27_all_cloud.json \
    results/AAPL/analysis_2026-02-27_hybrid_qwen.json
```

---

## Step 6: Tests for compare_runs

Create `tests/test_compare_runs.py`:

```python
"""Tests for the run comparison utility."""

import json
import tempfile
from pathlib import Path
from src.compare_runs import load_result, result_to_score


class TestCompareRuns:

    def _make_result(self, config_name="test", decision="HOLD", elapsed=120.0):
        return {
            "ticker": "AAPL",
            "trade_date": "2026-02-27",
            "provider": "anthropic",
            "hybrid_config": config_name,
            "decision": decision,
            "upstream_decision": decision,
            "final_trade_decision_text": (
                f"FINAL TRANSACTION PROPOSAL: **{decision}**\n"
                "Stop-loss: $258\nTarget: $295\n"
                "Position sizing: 5% of portfolio\n"
                "Revenue growth of 8.2% and P/E of 33x."
            ),
            "elapsed_seconds": elapsed,
            "quality_score": {
                "composite": 7.5,
                "reasoning_depth": 3,
                "data_grounding": 4,
                "risk_awareness": 10,
                "decision_consistent": True,
                "has_stop_loss": True,
                "has_price_target": True,
                "has_position_sizing": True,
            },
        }

    def test_load_result(self, tmp_path):
        result = self._make_result()
        filepath = tmp_path / "test_result.json"
        with open(filepath, "w") as f:
            json.dump(result, f)

        loaded = load_result(str(filepath))
        assert loaded["ticker"] == "AAPL"
        assert loaded["decision"] == "HOLD"

    def test_result_to_score(self):
        result = self._make_result("all_cloud", "HOLD")
        score = result_to_score(result)
        assert score.config_name == "all_cloud"
        assert score.decision == "HOLD"
        assert score.decision_consistent is True
        assert score.has_stop_loss is True
        assert score.composite_score > 0

    def test_result_to_score_missing_quality(self):
        """Handle result without quality_score block."""
        result = {
            "ticker": "AAPL",
            "trade_date": "2026-02-27",
            "provider": "anthropic",
            "hybrid_config": None,
            "decision": "BUY",
            "final_trade_decision_text": "",
        }
        score = result_to_score(result)
        assert score.config_name == "anthropic"
        assert score.decision == "BUY"
        assert score.reasoning_depth == 0
```

### Run

```bash
pytest tests/test_compare_runs.py -v
```

---

## Step 7: Final Verification

### Run All Tests

```bash
cd ~/Projects/Trifecta_Trader/App/trifecta-trader-poc
pytest tests/ -v 2>&1 | tee results/task_004_all_tests.txt
```

### Commit

```bash
git add .
git commit -m "Add live hybrid validation, comparison tooling, and pipeline results"
git push
```

---

## Verification Checklist

- [ ] Step 1: Reasoning comparison run completed — qwen2.5:14b produces valid decisions
- [ ] Step 2: `run_analysis.py` updated with timing, quality scoring, enhanced result JSON
- [ ] Step 3: All-cloud baseline run completed — result JSON saved
- [ ] Step 4: Hybrid Qwen run completed — result JSON saved
- [ ] Step 5: `src/compare_runs.py` created — comparison report generated
- [ ] Step 6: `tests/test_compare_runs.py` created and passing
- [ ] Step 7: All tests pass, changes committed and pushed
- [ ] No vendor code modified

---

## Report

After completing all steps, create `docs/TASK_004_REPORT.md` containing:

1. Which steps succeeded and which had issues
2. **Reasoning comparison results** — quality of each model for bull/bear/trader prompts
3. **Full pipeline results** — side-by-side comparison table:
   - Config name
   - Final decision (BUY/HOLD/SELL)
   - Quality score (composite + each dimension)
   - Elapsed time
   - Whether signal processor corrected the decision
4. **Comparison report** — output of `compare_runs.py`
5. Full log snippets from both pipeline runs (first 50 and last 50 lines of each)
6. Any errors encountered and how they were resolved
7. `git log --oneline` showing all commits
8. **Recommendation**: which hybrid config should become the default, and whether we're ready for Phase B (Alpaca paper trading integration)
