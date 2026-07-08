"""Consensus decision rules for repeat-run model signals.

The engine's repeat-run benchmarks showed that a modal BUY/HOLD/SELL is not
enough: a 3/5 HOLD still means the model family disagreed. This module turns
repeat runs into either a high-confidence trade decision or an explicit
NO_TRADE abstention.

NO_TRADE is not HOLD. HOLD is a confident portfolio decision. NO_TRADE means
the repeat-run evidence failed the consensus gate:
  * no new entry is authorized,
  * no size increase/decrease is authorized from this signal alone,
  * existing-position handling must be defined by the caller's risk policy
    (for example keep-with-existing-stops, manual review, or flatten).
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

TRADE_DECISIONS = {"BUY", "HOLD", "SELL"}
ABSTAIN_DECISION = "NO_TRADE"
ABSTAIN_SEMANTICS = {
    "new_entries_allowed": False,
    "position_changes_allowed": False,
    "equivalent_to_hold": False,
    "existing_position_policy_required": True,
}


def _quality_of(row: dict[str, Any]) -> float | None:
    quality = row.get("quality_score")
    if isinstance(quality, dict):
        value = quality.get("composite")
    else:
        value = row.get("quality")
    return float(value) if isinstance(value, (int, float)) else None


def consensus_with_abstention(
    rows: list[dict[str, Any]],
    *,
    min_agreement: float = 0.8,
    min_quality: float = 8.0,
    min_runs: int = 3,
) -> dict[str, Any]:
    """Return a trade decision only when repeat runs form a strong consensus.

    Agreement is computed over all supplied rows, so UNKNOWN/errors penalize the
    vote instead of being silently dropped. Quality is averaged over rows with a
    parseable BUY/HOLD/SELL and a numeric composite score.
    """
    if not rows:
        return _abstain("no_runs", [], None, None, min_agreement, min_quality, min_runs)

    decisions = [
        str(row.get("decision", "UNKNOWN")).upper()
        if not row.get("error") and not row.get("_error")
        else "ERROR"
        for row in rows
    ]
    counts = Counter(decisions)
    trade_counts = {d: counts.get(d, 0) for d in sorted(TRADE_DECISIONS)}
    total = len(rows)

    if total < min_runs:
        return _abstain("insufficient_runs", decisions, None, None,
                        min_agreement, min_quality, min_runs, counts)

    max_count = max(trade_counts.values(), default=0)
    modal_trade_decisions = [d for d, count in trade_counts.items() if count == max_count and count > 0]
    if len(modal_trade_decisions) != 1:
        return _abstain("tie_or_no_trade_votes", decisions, None, None,
                        min_agreement, min_quality, min_runs, counts)

    modal = modal_trade_decisions[0]
    agreement = max_count / total
    qualities = [
        q for row in rows
        if str(row.get("decision", "")).upper() in TRADE_DECISIONS
        for q in [_quality_of(row)]
        if q is not None
    ]
    quality_mean = statistics.mean(qualities) if qualities else None
    quality_stdev = statistics.stdev(qualities) if len(qualities) > 1 else 0.0

    if agreement < min_agreement:
        return _abstain("low_agreement", decisions, modal, agreement,
                        min_agreement, min_quality, min_runs, counts,
                        quality_mean, quality_stdev)

    if quality_mean is None:
        return _abstain("missing_quality", decisions, modal, agreement,
                        min_agreement, min_quality, min_runs, counts)

    if quality_mean < min_quality:
        return _abstain("low_quality", decisions, modal, agreement,
                        min_agreement, min_quality, min_runs, counts,
                        quality_mean, quality_stdev)

    return {
        "decision": modal,
        "abstained": False,
        "trade_intent": "TRADE_DECISION",
        "reason": "consensus_pass",
        "runs": total,
        "decisions": decisions,
        "decision_counts": dict(counts),
        "modal_decision": modal,
        "agreement": round(agreement, 3),
        "quality_mean": round(quality_mean, 3),
        "quality_stdev": round(quality_stdev, 3),
        "min_agreement": min_agreement,
        "min_quality": min_quality,
        "min_runs": min_runs,
    }


def _abstain(
    reason: str,
    decisions: list[str],
    modal: str | None,
    agreement: float | None,
    min_agreement: float,
    min_quality: float,
    min_runs: int,
    counts: Counter | None = None,
    quality_mean: float | None = None,
    quality_stdev: float | None = None,
) -> dict[str, Any]:
    return {
        "decision": ABSTAIN_DECISION,
        "abstained": True,
        "trade_intent": "ABSTAIN",
        "no_trade_semantics": dict(ABSTAIN_SEMANTICS),
        "reason": reason,
        "runs": len(decisions),
        "decisions": decisions,
        "decision_counts": dict(counts or Counter(decisions)),
        "modal_decision": modal,
        "agreement": round(agreement, 3) if agreement is not None else None,
        "quality_mean": round(quality_mean, 3) if quality_mean is not None else None,
        "quality_stdev": round(quality_stdev, 3) if quality_stdev is not None else None,
        "min_agreement": min_agreement,
        "min_quality": min_quality,
        "min_runs": min_runs,
    }
