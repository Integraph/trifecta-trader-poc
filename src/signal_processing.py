"""
Improved signal processing for extracting trade decisions.

Fixes the upstream bug where process_signal() picks up BUY/SELL/HOLD
from reasoning context rather than the actual decision line.
"""

import re
from typing import Optional


def extract_decision(full_signal: str) -> str:
    """Extract the trade decision from the full signal text.

    Uses a priority-based extraction:
    1. Look for 'FINAL TRANSACTION PROPOSAL: <DECISION>'
    2. Look for 'MY RECOMMENDATION: <DECISION>'
    3. Look for the last standalone BUY/HOLD/SELL not in a negation context
    4. Return 'UNKNOWN' if no clear decision found

    Args:
        full_signal: The complete text output from the trading pipeline

    Returns:
        One of: 'BUY', 'HOLD', 'SELL', or 'UNKNOWN'
    """
    if not full_signal or not isinstance(full_signal, str):
        return "UNKNOWN"

    # Method 1: Look for FINAL TRANSACTION PROPOSAL line
    # Handles formats like:
    #   FINAL TRANSACTION PROPOSAL: **HOLD**
    #   FINAL TRANSACTION PROPOSAL: HOLD
    #   FINAL TRANSACTION PROPOSAL: **BUY** AAPL
    #   ## FINAL TRANSACTION PROPOSAL: **SELL**
    proposal_pattern = r'FINAL\s+TRANSACTION\s+PROPOSAL[:\s]*\*{0,2}(BUY|HOLD|SELL)\*{0,2}'
    proposals = re.findall(proposal_pattern, full_signal, re.IGNORECASE)

    if proposals:
        return proposals[-1].upper()

    # Method 2: Look for "MY RECOMMENDATION: <DECISION>" pattern
    recommendation_pattern = r'MY\s+RECOMMENDATION[:\s]*\*{0,2}(BUY|HOLD|SELL)\*{0,2}'
    recommendations = re.findall(recommendation_pattern, full_signal, re.IGNORECASE)

    if recommendations:
        return recommendations[-1].upper()

    # Method 3: Look for standalone decision words, excluding negation contexts
    cleaned = full_signal
    negation_patterns = [
        r"(?:NOT|n't|not)\s+(?:recommending|recommend|suggesting|suggest)\s+(?:a\s+)?(?:full\s+)?(BUY|HOLD|SELL)",
        r"(?:NOT|n't|not)\s+(?:a\s+)?(BUY|HOLD|SELL)",
        r"Why\s+(?:I'm\s+)?NOT\s+(?:Recommending\s+)?(BUY|HOLD|SELL)",
        r"(?:rather\s+than|instead\s+of|over)\s+(?:a\s+)?(?:full\s+)?(BUY|HOLD|SELL)",
    ]
    for pattern in negation_patterns:
        cleaned = re.sub(pattern, "[NEGATED]", cleaned, flags=re.IGNORECASE)

    standalone_pattern = r'\b(BUY|HOLD|SELL)\b'
    decisions = re.findall(standalone_pattern, cleaned, re.IGNORECASE)

    if decisions:
        return decisions[-1].upper()

    return "UNKNOWN"
