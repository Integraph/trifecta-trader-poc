"""
Quality scorer for comparing pipeline outputs across LLM configurations.

Evaluates the quality of trading analysis along several dimensions:
- Decision consistency (does the extracted decision match the reasoning?)
- Reasoning depth (length, specificity, use of data)
- Data grounding (does the analysis reference real data points?)
- Risk awareness (are risk parameters specified?)
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QualityScore:
    """Quality assessment of a pipeline run."""
    config_name: str
    ticker: str
    trade_date: str

    decision: str
    decision_consistent: bool

    reasoning_depth: int
    data_grounding: int
    risk_awareness: int

    total_tokens_approx: int = 0
    estimated_cost_usd: float = 0.0

    full_output_length: int = 0
    num_data_points_cited: int = 0
    has_stop_loss: bool = False
    has_price_target: bool = False
    has_position_sizing: bool = False

    @property
    def composite_score(self) -> float:
        """Weighted composite quality score (0-10)."""
        return (
            (3.0 if self.decision_consistent else 0.0) +
            self.reasoning_depth * 0.3 +
            self.data_grounding * 0.2 +
            self.risk_awareness * 0.2
        )


def score_pipeline_output(
    config_name: str,
    ticker: str,
    trade_date: str,
    final_trade_decision: str,
    extracted_decision: str,
    full_output: Optional[str] = None,
) -> QualityScore:
    """Score a pipeline output for quality.

    Args:
        config_name: Name of the hybrid config used
        ticker: Stock ticker
        trade_date: Analysis date
        final_trade_decision: Raw text from the pipeline
        extracted_decision: Decision extracted by our signal processor
        full_output: Full pipeline output text (if available)

    Returns:
        QualityScore with all metrics
    """
    text = final_trade_decision or ""

    from src.signal_processing import extract_decision
    our_decision = extract_decision(text)
    decision_consistent = our_decision == extracted_decision

    word_count = len(text.split())
    if word_count > 500:
        reasoning_depth = min(10, 5 + (word_count - 500) // 200)
    elif word_count > 200:
        reasoning_depth = 3 + (word_count - 200) // 100
    elif word_count > 50:
        reasoning_depth = 1 + word_count // 50
    else:
        reasoning_depth = 0

    numbers = re.findall(r'\$[\d,.]+|\d+\.?\d*%|\d{1,3}(?:,\d{3})+', text)
    num_data_points = len(numbers)
    data_grounding = min(10, num_data_points // 2)

    risk_score = 0
    has_stop_loss = bool(re.search(r'stop.?loss', text, re.IGNORECASE))
    has_price_target = bool(re.search(r'(?:price\s+)?target|upside|downside', text, re.IGNORECASE))
    has_position_sizing = bool(re.search(r'position\s+siz|allocation|portfolio\s+weight', text, re.IGNORECASE))

    if has_stop_loss:
        risk_score += 4
    if has_price_target:
        risk_score += 3
    if has_position_sizing:
        risk_score += 3
    risk_awareness = min(10, risk_score)

    return QualityScore(
        config_name=config_name,
        ticker=ticker,
        trade_date=trade_date,
        decision=extracted_decision,
        decision_consistent=decision_consistent,
        reasoning_depth=reasoning_depth,
        data_grounding=data_grounding,
        risk_awareness=risk_awareness,
        full_output_length=len(text),
        num_data_points_cited=num_data_points,
        has_stop_loss=has_stop_loss,
        has_price_target=has_price_target,
        has_position_sizing=has_position_sizing,
    )


def compare_scores(scores: List[QualityScore]) -> str:
    """Generate a comparison report from multiple pipeline runs.

    Args:
        scores: List of QualityScore from different configurations

    Returns:
        Formatted comparison table as a string
    """
    if not scores:
        return "No scores to compare."

    lines = []
    lines.append("=" * 80)
    lines.append("HYBRID LLM QUALITY COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    header = (
        f"{'Config':<25} {'Decision':<8} {'Consistent':<11} "
        f"{'Depth':<6} {'Data':<6} {'Risk':<6} {'Score':<6} {'Cost':<8}"
    )
    lines.append(header)
    lines.append("-" * 80)

    for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
        consistent_mark = "yes" if s.decision_consistent else "no"
        line = (
            f"{s.config_name:<25} "
            f"{s.decision:<8} "
            f"{consistent_mark:<11} "
            f"{s.reasoning_depth:<6} "
            f"{s.data_grounding:<6} "
            f"{s.risk_awareness:<6} "
            f"{s.composite_score:<6.1f} "
            f"${s.estimated_cost_usd:<7.4f}"
        )
        lines.append(line)

    lines.append("-" * 80)
    lines.append("")

    best = max(scores, key=lambda x: x.composite_score)
    cheapest = min(scores, key=lambda x: x.estimated_cost_usd)
    lines.append(f"Best quality:  {best.config_name} (score: {best.composite_score:.1f})")
    lines.append(f"Lowest cost:   {cheapest.config_name} (${cheapest.estimated_cost_usd:.4f})")

    return "\n".join(lines)
