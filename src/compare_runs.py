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
        estimated_cost_usd=0.0,
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

        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / "comparison_report.txt"
        with open(report_file, "w") as f:
            f.write(compare_scores(scores))
        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
