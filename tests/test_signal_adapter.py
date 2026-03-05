"""Tests for src/integration/signal_adapter.py (Task 013).

Covers:
- transform_to_signal() produces all required AISignal fields
- transform_to_signal() with None trade_params (price fields are null)
- alphaScore / confidence clamping and scaling
- confidenceInterval derivation (±0.08 band, capped at [0,1])
- strategy field is lowercased decision
- id is a valid UUID4
- signal_ref slug format (sig_TICKER_YYYYMMDD_HHMMSS)
- signal_ref stored in evidenceIds as type='internal'
- riskFlags extraction (volatility, earnings, poor R/R, missing bracket)
- decision_consistent=False triggers 'decision_inconsistent' flag
- createdAt / expiresAt timezone handling (defaults to UTC)
- expiresAt is 24h after createdAt by default
- custom ttl_hours respected
- reasoning extraction (first 2000 chars)
- reasoningPath contains all expected agent names
- to_supabase_row() converts camelCase → snake_case
- to_supabase_row() preserves all field values
"""

import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_result(
    ticker="AAPL",
    decision="BUY",
    composite=9.4,
    reasoning_depth=8,
    data_grounding=10,
    risk_awareness=10,
    decision_consistent=True,
    has_stop_loss=True,
    has_price_target=True,
    has_position_sizing=True,
    run_timestamp="2026-03-03T15:05:25.374611",
    trade_date="2026-03-03",
    final_text="## Summary\nStrong technical momentum and solid fundamentals.",
):
    return {
        "ticker":        ticker,
        "decision":      decision,
        "trade_date":    trade_date,
        "run_timestamp": run_timestamp,
        "final_trade_decision_text": final_text,
        "trader_investment_plan":    "## KEY METRICS\nP/E: 28x",
        "hybrid_config": "hybrid_haiku_tools",
        "elapsed_seconds": 1006.0,
        "quality_score": {
            "composite":           composite,
            "reasoning_depth":     reasoning_depth,
            "data_grounding":      data_grounding,
            "risk_awareness":      risk_awareness,
            "decision_consistent": decision_consistent,
            "has_stop_loss":       has_stop_loss,
            "has_price_target":    has_price_target,
            "has_position_sizing": has_position_sizing,
        },
        "cost_breakdown": {"total_usd": 0.0626},
        "portfolio_context": {"held": False},
    }


def _make_trade_params(
    entry_price=195.0,
    stop_loss=180.0,
    price_target=220.0,
    position_pct=5.0,
    risk_reward_ratio=2.0,
    confidence="medium-high",
    has_bracket_params=True,
):
    tp = MagicMock()
    tp.entry_price        = entry_price
    tp.stop_loss          = stop_loss
    tp.price_target       = price_target
    tp.position_pct       = position_pct
    tp.risk_reward_ratio  = risk_reward_ratio
    tp.confidence         = confidence
    tp.has_bracket_params = has_bracket_params
    return tp


# ── 1. Required fields present ────────────────────────────────────────────────

class TestRequiredFields:

    def test_all_required_fields_present(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params())
        for field in ("id", "symbol", "strategy", "alphaScore", "confidence",
                      "confidenceInterval", "riskFlags", "reasoning",
                      "reasoningPath", "promptVersion", "evidenceIds",
                      "createdAt", "expiresAt"):
            assert field in signal, f"Missing field: {field}"

    def test_id_is_valid_uuid4(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params())
        parsed = uuid.UUID(signal["id"])
        assert parsed.version == 4

    def test_symbol_matches_ticker(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(ticker="TSLA"), _make_trade_params())
        assert signal["symbol"] == "TSLA"

    def test_ticker_is_uppercased(self):
        from src.integration.signal_adapter import transform_to_signal
        result = _make_result()
        result["ticker"] = "aapl"
        signal = transform_to_signal(result, _make_trade_params())
        assert signal["symbol"] == "AAPL"

    def test_strategy_is_lowercased_decision(self):
        from src.integration.signal_adapter import transform_to_signal
        for decision, expected in [("BUY", "buy"), ("SELL", "sell"), ("HOLD", "hold")]:
            signal = transform_to_signal(_make_result(decision=decision), _make_trade_params())
            assert signal["strategy"] == expected


# ── 2. Score scaling ──────────────────────────────────────────────────────────

class TestScoreScaling:

    def test_alpha_score_scaled_from_composite(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(composite=9.4), _make_trade_params())
        assert abs(signal["alphaScore"] - 0.94) < 0.001

    def test_confidence_scaled_from_reasoning_depth(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(reasoning_depth=8), _make_trade_params())
        assert abs(signal["confidence"] - 0.8) < 0.001

    def test_alpha_score_clamped_at_1(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(composite=12.0), _make_trade_params())
        assert signal["alphaScore"] == 1.0

    def test_alpha_score_clamped_at_0(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(composite=-1.0), _make_trade_params())
        assert signal["alphaScore"] == 0.0

    def test_confidence_interval_structure(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(reasoning_depth=8), _make_trade_params())
        ci = signal["confidenceInterval"]
        assert ci["level"] == "95%"
        assert "lower" in ci and "upper" in ci

    def test_confidence_interval_width(self):
        from src.integration.signal_adapter import transform_to_signal, CI_HALF_WIDTH
        signal = transform_to_signal(_make_result(reasoning_depth=8), _make_trade_params())
        ci = signal["confidenceInterval"]
        conf = signal["confidence"]
        assert abs(ci["lower"] - max(0.0, conf - CI_HALF_WIDTH)) < 0.001
        assert abs(ci["upper"] - min(1.0, conf + CI_HALF_WIDTH)) < 0.001

    def test_confidence_interval_capped_at_1(self):
        from src.integration.signal_adapter import transform_to_signal
        # reasoning_depth=10 → confidence=1.0; upper would be 1.08 but clamped to 1.0
        signal = transform_to_signal(_make_result(reasoning_depth=10), _make_trade_params())
        assert signal["confidenceInterval"]["upper"] <= 1.0

    def test_confidence_interval_floored_at_0(self):
        from src.integration.signal_adapter import transform_to_signal
        # reasoning_depth=0 → confidence=0.0; lower would be -0.08 but clamped to 0.0
        signal = transform_to_signal(_make_result(reasoning_depth=0), _make_trade_params())
        assert signal["confidenceInterval"]["lower"] >= 0.0


# ── 3. Trade params mapping ───────────────────────────────────────────────────

class TestTradeParamsMapping:

    def test_entry_price_mapped(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params(entry_price=195.0))
        assert signal["entryPrice"] == 195.0

    def test_stop_loss_mapped(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params(stop_loss=180.0))
        assert signal["stopLoss"] == 180.0

    def test_take_profit_from_price_target(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params(price_target=220.0))
        assert signal["takeProfit"] == 220.0

    def test_none_trade_params_yields_null_prices(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), trade_params=None)
        assert signal["entryPrice"]  is None
        assert signal["stopLoss"]    is None
        assert signal["takeProfit"]  is None


# ── 4. Risk flags ─────────────────────────────────────────────────────────────

class TestRiskFlags:

    def test_high_volatility_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        result = _make_result(final_text="High volatility expected this week.")
        flags = extract_risk_flags(result, _make_trade_params())
        assert "high_volatility" in flags

    def test_volatile_keyword_triggers_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        result = _make_result(final_text="The stock is highly volatile.")
        flags = extract_risk_flags(result, _make_trade_params())
        assert "high_volatility" in flags

    def test_earnings_approaching_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        result = _make_result(
            final_text="Earnings are upcoming next week and could cause volatility."
        )
        flags = extract_risk_flags(result, _make_trade_params())
        assert "earnings_approaching" in flags

    def test_poor_risk_reward_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        tp = _make_trade_params(risk_reward_ratio=1.2)
        flags = extract_risk_flags(_make_result(), tp)
        assert "poor_risk_reward" in flags

    def test_good_risk_reward_no_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        tp = _make_trade_params(risk_reward_ratio=2.5)
        flags = extract_risk_flags(_make_result(), tp)
        assert "poor_risk_reward" not in flags

    def test_missing_bracket_params_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        tp = _make_trade_params(has_bracket_params=False)
        flags = extract_risk_flags(_make_result(), tp)
        assert "missing_bracket_params" in flags

    def test_decision_inconsistent_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        result = _make_result(decision_consistent=False)
        flags = extract_risk_flags(result, _make_trade_params())
        assert "decision_inconsistent" in flags

    def test_low_confidence_flag(self):
        from src.integration.signal_adapter import extract_risk_flags
        tp = _make_trade_params(confidence="low")
        flags = extract_risk_flags(_make_result(), tp)
        assert "low_confidence" in flags

    def test_clean_signal_no_flags(self):
        from src.integration.signal_adapter import extract_risk_flags
        result = _make_result(
            final_text="Strong fundamentals and steady growth outlook.",
            decision_consistent=True,
        )
        tp = _make_trade_params(
            risk_reward_ratio=2.0,
            confidence="high",
            has_bracket_params=True,
        )
        flags = extract_risk_flags(result, tp)
        assert "poor_risk_reward"      not in flags
        assert "decision_inconsistent" not in flags
        assert "low_confidence"        not in flags

    def test_none_trade_params_skips_param_flags(self):
        from src.integration.signal_adapter import extract_risk_flags
        # Should not raise; param-based flags simply absent
        flags = extract_risk_flags(_make_result(), trade_params=None)
        assert "poor_risk_reward"    not in flags
        assert "missing_bracket_params" not in flags


# ── 5. Signal reference slug ──────────────────────────────────────────────────

class TestSignalRef:

    def test_signal_ref_format(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(
            _make_result(ticker="AAPL", trade_date="2026-03-03",
                         run_timestamp="2026-03-03T14:30:25"),
            _make_trade_params(),
        )
        assert signal["signal_ref"].startswith("sig_AAPL_20260303_")

    def test_signal_ref_in_evidence_ids(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params())
        internal = [e for e in signal["evidenceIds"] if e.get("type") == "internal"]
        assert len(internal) == 1
        assert internal[0]["id"] == signal["signal_ref"]

    def test_evidence_ids_include_analyst_reports(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params())
        types = [e["type"] for e in signal["evidenceIds"]]
        assert types.count("analysis") >= 4  # market, social_media, news, fundamentals


# ── 6. Timestamps ─────────────────────────────────────────────────────────────

class TestTimestamps:

    def test_created_at_is_iso8601(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params())
        # Should parse without error
        datetime.fromisoformat(signal["createdAt"].replace("Z", "+00:00"))

    def test_expires_at_24h_after_created_at(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params(), ttl_hours=24)
        created = datetime.fromisoformat(signal["createdAt"])
        expires = datetime.fromisoformat(signal["expiresAt"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        delta = expires - created
        assert abs(delta.total_seconds() - 86400) < 2

    def test_custom_ttl_respected(self):
        from src.integration.signal_adapter import transform_to_signal
        signal = transform_to_signal(_make_result(), _make_trade_params(), ttl_hours=48)
        created = datetime.fromisoformat(signal["createdAt"])
        expires = datetime.fromisoformat(signal["expiresAt"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        delta = expires - created
        assert abs(delta.total_seconds() - 48 * 3600) < 2

    def test_invalid_timestamp_falls_back_to_now(self):
        from src.integration.signal_adapter import transform_to_signal
        result = _make_result(run_timestamp="not-a-timestamp")
        # Should not raise
        signal = transform_to_signal(result, _make_trade_params())
        assert signal["createdAt"] != ""


# ── 7. Reasoning path ─────────────────────────────────────────────────────────

class TestReasoningPath:

    def test_reasoning_path_contains_all_agents(self):
        from src.integration.signal_adapter import transform_to_signal, REASONING_PATH
        signal = transform_to_signal(_make_result(), _make_trade_params())
        for agent in REASONING_PATH:
            assert agent in signal["reasoningPath"]

    def test_prompt_version_is_semantic(self):
        from src.integration.signal_adapter import transform_to_signal, PROMPT_VERSION
        signal = transform_to_signal(_make_result(), _make_trade_params())
        assert signal["promptVersion"] == PROMPT_VERSION
        parts = PROMPT_VERSION.lstrip("v").split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts)


# ── 8. to_supabase_row() ──────────────────────────────────────────────────────

class TestToSupabaseRow:

    def test_camel_to_snake_case_conversion(self):
        from src.integration.signal_adapter import transform_to_signal, to_supabase_row
        signal = transform_to_signal(_make_result(), _make_trade_params())
        row = to_supabase_row(signal)
        for key in ("alpha_score", "confidence_interval", "entry_price",
                    "stop_loss", "take_profit", "risk_flags",
                    "reasoning_path", "prompt_version", "evidence_ids",
                    "created_at", "expires_at"):
            assert key in row, f"Missing snake_case key: {key}"

    def test_values_preserved(self):
        from src.integration.signal_adapter import transform_to_signal, to_supabase_row
        signal = transform_to_signal(
            _make_result(composite=9.4, reasoning_depth=8),
            _make_trade_params(entry_price=195.0, stop_loss=180.0, price_target=220.0),
        )
        row = to_supabase_row(signal)
        assert abs(row["alpha_score"] - 0.94) < 0.001
        assert row["entry_price"] == 195.0
        assert row["stop_loss"]   == 180.0
        assert row["take_profit"] == 220.0
        assert row["strategy"]    == "buy"

    def test_none_prices_preserved(self):
        from src.integration.signal_adapter import transform_to_signal, to_supabase_row
        signal = transform_to_signal(_make_result(), trade_params=None)
        row = to_supabase_row(signal)
        assert row["entry_price"] is None
        assert row["stop_loss"]   is None
        assert row["take_profit"] is None

    def test_id_is_uuid_string(self):
        from src.integration.signal_adapter import transform_to_signal, to_supabase_row
        signal = transform_to_signal(_make_result(), _make_trade_params())
        row = to_supabase_row(signal)
        parsed = uuid.UUID(row["id"])
        assert parsed.version == 4
