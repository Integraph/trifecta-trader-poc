from src.consensus import ABSTAIN_DECISION, ABSTAIN_SEMANTICS, consensus_with_abstention


def row(decision, quality=8.5, **extra):
    return {
        "decision": decision,
        "quality_score": {"composite": quality},
        **extra,
    }


def test_high_agreement_high_quality_passes():
    out = consensus_with_abstention([
        row("SELL", 8.5),
        row("SELL", 8.3),
        row("SELL", 7.9),
    ])

    assert out["decision"] == "SELL"
    assert out["abstained"] is False
    assert out["trade_intent"] == "TRADE_DECISION"
    assert out["agreement"] == 1.0
    assert out["quality_mean"] == 8.233


def test_weak_three_of_five_majority_abstains():
    out = consensus_with_abstention([
        row("BUY", 8.7),
        row("BUY", 8.2),
        row("HOLD", 8.6),
        row("HOLD", 8.4),
        row("HOLD", 8.3),
    ])

    assert out["decision"] == ABSTAIN_DECISION
    assert out["trade_intent"] == "ABSTAIN"
    assert out["no_trade_semantics"] == ABSTAIN_SEMANTICS
    assert out["reason"] == "low_agreement"
    assert out["modal_decision"] == "HOLD"
    assert out["agreement"] == 0.6


def test_no_trade_semantics_are_not_hold():
    out = consensus_with_abstention([
        row("BUY", 8.7),
        row("HOLD", 8.4),
        row("SELL", 8.5),
    ])

    assert out["decision"] == ABSTAIN_DECISION
    assert out["trade_intent"] == "ABSTAIN"
    assert out["no_trade_semantics"]["equivalent_to_hold"] is False
    assert out["no_trade_semantics"]["new_entries_allowed"] is False
    assert out["no_trade_semantics"]["position_changes_allowed"] is False
    assert out["no_trade_semantics"]["existing_position_policy_required"] is True


def test_low_quality_abstains_even_with_unanimous_vote():
    out = consensus_with_abstention([
        row("HOLD", 7.8),
        row("HOLD", 7.7),
        row("HOLD", 7.6),
    ])

    assert out["decision"] == ABSTAIN_DECISION
    assert out["reason"] == "low_quality"


def test_unknown_rows_penalize_agreement():
    out = consensus_with_abstention([
        row("BUY", 9.0),
        row("BUY", 8.8),
        row("UNKNOWN", None),
    ])

    assert out["decision"] == ABSTAIN_DECISION
    assert out["reason"] == "low_agreement"
    assert out["agreement"] == 0.667


def test_tie_abstains():
    out = consensus_with_abstention([
        row("BUY", 9.0),
        row("SELL", 9.1),
        row("HOLD", 9.2),
    ])

    assert out["decision"] == ABSTAIN_DECISION
    assert out["reason"] == "tie_or_no_trade_votes"


def test_insufficient_runs_abstains():
    out = consensus_with_abstention([
        row("BUY", 9.0),
        row("BUY", 9.1),
    ])

    assert out["decision"] == ABSTAIN_DECISION
    assert out["reason"] == "insufficient_runs"
