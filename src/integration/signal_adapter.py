"""
Signal Adapter — transforms pipeline result dicts into the platform's AISignal schema.

The pipeline produces structured JSON (see run_analysis.py).
The platform UI expects AISignal rows in Supabase (snake_case columns).
This module bridges the gap.

Transformation highlights:
- id: UUID4 (Supabase PK)
- signal_ref: human-readable slug stored in evidenceIds for debugging
- alphaScore: quality_score.composite / 10, clamped to [0, 1]
- confidence: reasoning_depth / 10, clamped to [0, 1]
- confidenceInterval: ±0.08 band around confidence, capped at [0, 1]
- riskFlags: extracted from analysis text and trade param heuristics
- reasoning: first 2000 chars of the final trade decision text
- expiresAt: createdAt + configurable TTL (default 24h)
"""

import re
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Bump this string whenever pipeline prompts materially change.
# The UI displays it for EU AI Act audit compliance.
PROMPT_VERSION = "v1.0.0"

# Standard agent names that run in every pipeline execution.
REASONING_PATH = [
    "market_analyst",
    "social_media_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "bull_researcher",
    "bear_researcher",
    "risk_judge",
]

# Default signal TTL in hours (overridden by config/supabase.yaml)
DEFAULT_TTL_HOURS = 24

# Confidence interval half-width (±0.08 at 95%)
CI_HALF_WIDTH = 0.08


# ── Risk flag extraction ───────────────────────────────────────────────────────

def extract_risk_flags(result: dict, trade_params) -> list:
    """Extract risk flags from analysis text and trade parameters.

    Args:
        result: Pipeline result dict from run_analysis().
        trade_params: TradeParams dataclass instance (may be None).

    Returns:
        List of risk flag strings (may be empty).
    """
    flags = []
    text = (result.get("final_trade_decision_text", "") or "").lower()

    # Volatility
    if "high volatility" in text or "volatile" in text:
        flags.append("high_volatility")

    # Earnings proximity
    if "earnings" in text and any(
        kw in text for kw in ("upcoming", "approaching", "next week", "next quarter")
    ):
        flags.append("earnings_approaching")

    # Macro risk
    if any(kw in text for kw in ("recession", "fed", "interest rate", "inflation risk")):
        flags.append("macro_risk")

    # Sector headwinds
    if any(kw in text for kw in ("sector headwind", "regulatory risk", "antitrust")):
        flags.append("sector_headwinds")

    if trade_params is not None:
        # Low confidence
        if trade_params.confidence and "low" in str(trade_params.confidence).lower():
            flags.append("low_confidence")

        # Poor risk/reward
        if (trade_params.risk_reward_ratio is not None
                and trade_params.risk_reward_ratio < 1.5):
            flags.append("poor_risk_reward")

        # Missing bracket params (no stop-loss or target)
        if not getattr(trade_params, "has_bracket_params", None):
            flags.append("missing_bracket_params")

    # Decision inconsistency
    if not result.get("quality_score", {}).get("decision_consistent", True):
        flags.append("decision_inconsistent")

    return flags


# ── Reasoning extraction ──────────────────────────────────────────────────────

def _extract_reasoning(text: str, max_chars: int = 2000) -> str:
    """Return up to max_chars of the most informative part of the analysis.

    Tries to find a ## Summary or ## RECOMMENDATION section first;
    falls back to the first max_chars of the full text.
    """
    if not text:
        return ""

    # Try to find a summary or recommendation section
    for header in (r"##\s*Summary", r"##\s*RECOMMENDATION", r"##\s*Final", r"## Decision"):
        m = re.search(header, text, re.IGNORECASE)
        if m:
            excerpt = text[m.start():]
            return excerpt[:max_chars].strip()

    return text[:max_chars].strip()


# ── Evidence IDs ──────────────────────────────────────────────────────────────

def _build_evidence_ids(result: dict, signal_ref: str, created_at: str) -> list:
    """Build the evidenceIds array.

    Always includes one 'internal' entry (the signal_ref slug for debugging),
    plus one entry per analyst report in the pipeline.
    """
    ticker     = result.get("ticker", "")
    trade_date = result.get("trade_date", "")
    ts         = created_at

    evidence = [
        # Internal reference — human-readable slug for debugging without schema change
        {"type": "internal", "id": signal_ref, "timestamp": ts},
    ]

    # One entry per analyst report
    for analyst in ("market", "social_media", "news", "fundamentals"):
        evidence.append({
            "type":      "analysis",
            "id":        f"{analyst}_report_{ticker}_{trade_date.replace('-', '')}",
            "timestamp": ts,
        })

    return evidence


# ── Main transform function ───────────────────────────────────────────────────

def transform_to_signal(
    result: dict,
    trade_params=None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict:
    """Transform a pipeline result dict into a platform AISignal dict.

    Args:
        result: Pipeline result dict from run_analysis().
        trade_params: TradeParams dataclass (may be None — price fields will be null).
        ttl_hours: Signal expiry in hours from creation (default 24).

    Returns:
        AISignal dict with both camelCase (for logging) and the full set of
        fields needed by to_supabase_row().
    """
    ticker     = result.get("ticker", "UNKNOWN").upper()
    trade_date = result.get("trade_date", "")
    run_ts_raw = result.get("run_timestamp", "")
    decision   = result.get("decision", "HOLD")

    # ── Timestamps ────────────────────────────────────────────────────────────
    try:
        created_dt = datetime.fromisoformat(run_ts_raw)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        created_dt = datetime.now(timezone.utc)

    created_at = created_dt.isoformat()
    expires_at = (created_dt + timedelta(hours=ttl_hours)).isoformat()

    # ── ID generation ─────────────────────────────────────────────────────────
    time_slug  = created_dt.strftime("%H%M%S")
    date_slug  = trade_date.replace("-", "") if trade_date else created_dt.strftime("%Y%m%d")
    signal_ref = f"sig_{ticker}_{date_slug}_{time_slug}"
    signal_id  = str(uuid.uuid4())

    # ── Scores ────────────────────────────────────────────────────────────────
    qs             = result.get("quality_score", {})
    composite      = qs.get("composite", 0.0) or 0.0
    reasoning_d    = qs.get("reasoning_depth", 0) or 0
    alpha_score    = max(0.0, min(1.0, composite / 10.0))
    confidence     = max(0.0, min(1.0, reasoning_d / 10.0))
    ci_lower       = round(max(0.0, confidence - CI_HALF_WIDTH), 4)
    ci_upper       = round(min(1.0, confidence + CI_HALF_WIDTH), 4)

    # ── Trade params (may be None) ────────────────────────────────────────────
    entry_price = None
    stop_loss   = None
    take_profit = None
    if trade_params is not None:
        entry_price = getattr(trade_params, "entry_price",  None)
        stop_loss   = getattr(trade_params, "stop_loss",    None)
        take_profit = getattr(trade_params, "price_target", None)

    # ── Risk flags ────────────────────────────────────────────────────────────
    risk_flags = extract_risk_flags(result, trade_params)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasoning = _extract_reasoning(result.get("final_trade_decision_text", ""))

    # ── Evidence IDs ─────────────────────────────────────────────────────────
    evidence_ids = _build_evidence_ids(result, signal_ref, created_at)

    signal = {
        # UUID PK for Supabase
        "id":           signal_id,
        # Human-readable slug in evidenceIds (see _build_evidence_ids)
        "signal_ref":   signal_ref,
        # Core fields
        "symbol":       ticker,
        "strategy":     decision.lower(),
        "alphaScore":   round(alpha_score, 4),
        "confidence":   round(confidence, 4),
        "confidenceInterval": {
            "lower": ci_lower,
            "upper": ci_upper,
            "level": "95%",
        },
        "entryPrice":   entry_price,
        "stopLoss":     stop_loss,
        "takeProfit":   take_profit,
        "riskFlags":    risk_flags,
        "reasoning":    reasoning,
        "reasoningPath": REASONING_PATH,
        "promptVersion": PROMPT_VERSION,
        "evidenceIds":  evidence_ids,
        "createdAt":    created_at,
        "expiresAt":    expires_at,
    }

    logger.info(
        "Signal built: %s  %s  alpha=%.2f  conf=%.2f  flags=%s",
        signal_ref, decision, alpha_score, confidence, risk_flags,
    )
    return signal


def to_supabase_row(signal: dict) -> dict:
    """Convert an AISignal dict (camelCase) to a Supabase row (snake_case).

    The Supabase table uses snake_case columns; the frontend API converts
    to camelCase for JavaScript consumers.
    """
    return {
        "id":                  signal["id"],
        "symbol":              signal["symbol"],
        "strategy":            signal["strategy"],
        "alpha_score":         signal["alphaScore"],
        "confidence":          signal["confidence"],
        "confidence_interval": signal["confidenceInterval"],
        "entry_price":         signal["entryPrice"],
        "stop_loss":           signal["stopLoss"],
        "take_profit":         signal["takeProfit"],
        "risk_flags":          signal["riskFlags"],
        "reasoning":           signal["reasoning"],
        "reasoning_path":      signal["reasoningPath"],
        "prompt_version":      signal["promptVersion"],
        "evidence_ids":        signal["evidenceIds"],
        "created_at":          signal["createdAt"],
        "expires_at":          signal["expiresAt"],
    }
