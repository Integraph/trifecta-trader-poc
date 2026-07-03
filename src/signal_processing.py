"""
Improved signal processing for extracting trade decisions.

Fixes the upstream bug where process_signal() picks up BUY/SELL/HOLD
from reasoning context rather than the actual decision line.
"""

import re
from typing import Optional

# 5-level PortfolioDecision rating -> our 3-level decision (standard mapping:
# Overweight -> BUY = increase exposure; Underweight -> SELL = reduce exposure).
PM_RATING_MAP = {
    "BUY": "BUY", "OVERWEIGHT": "BUY",
    "HOLD": "HOLD",
    "UNDERWEIGHT": "SELL", "SELL": "SELL",
}

# v0.3.0 render_pm_decision() template fingerprint (vendor schemas.py): the
# structured path always renders '**Rating**: <rating>' plus the exact
# '**Executive Summary**:' / '**Investment Thesis**:' headers. Matching this
# template is a deterministic inverse of the vendor render — it recovers the
# typed rating without modifying the vendor (zero-mod).
_PM_RENDER_RATING = re.compile(
    r'^\*\*Rating\*\*:\s*(Buy|Overweight|Hold|Underweight|Sell)\b',
    re.IGNORECASE | re.MULTILINE,
)
_PM_RENDER_HEADERS = re.compile(
    r'^\*\*(?:Executive Summary|Investment Thesis)\*\*:', re.MULTILINE
)
_FENCED_CODE = re.compile(r'```.*?(?:```|\Z)', re.DOTALL)


def _regex_decision_detailed(full_signal: str) -> tuple:
    """Regex-based extraction. Returns (decision, matched_rating_token_or_None).

    Uses a priority-based extraction:
    0. Look for a PM decision-header line
       ('**<Label>**: <Buy|Overweight|Hold|Underweight|Sell>')
    1. Look for 'FINAL TRANSACTION PROPOSAL: <DECISION>'
    2. Look for 'MY RECOMMENDATION: <DECISION>'
    3. Look for the last standalone BUY/HOLD/SELL not in a negation context
    4. ('UNKNOWN', None) if no clear decision found
    """
    if not full_signal or not isinstance(full_signal, str):
        return "UNKNOWN", None

    # Method 0 (TradingAgents v0.3.0): the Portfolio Manager emits a 5-level
    # PortfolioDecision rating (Buy / Overweight / Hold / Underweight / Sell) on a
    # decision header line. Different LLMs label that line differently — the
    # structured render uses "**Recommendation**:", cloud models (Haiku/Sonnet)
    # tend to write "**Rating**:", and local qwen writes "**Action**:" and
    # "**Final Transaction Proposal: <rating>**".
    #
    # The label must START a line (allowing markdown heading/list/bold prefixes)
    # and the colon is mandatory — so label-shaped prose mid-sentence ("a prior
    # note wrote Rating: Underweight") or quoted stale decisions cannot override
    # the real decision header (Codex QA P1, TRI-66). If a model repeats the
    # header while looping, the last decision line wins. Residual ceiling: a
    # prose line that itself BEGINS with "<Label>: <rating>" still matches —
    # regex scraping can't fully disambiguate that; structured PortfolioDecision
    # extraction (TRI-70) is the durable fix.
    #
    # 5-level -> 3-level mapping (standard):
    #   Overweight -> BUY (favorable, increase exposure)
    #   Underweight -> SELL (reduce exposure, take partial profits)
    # Core labels observed so far: Recommendation / Rating / Action /
    # Transaction Proposal — optionally prefixed by ONE qualifier from a tight
    # allowlist (local qwen wrote "**Investment Recommendation: Underweight**"
    # in one run and "**Action**: ..." in the next — the label varies per RUN,
    # not just per model).
    #
    # Quoted/stale decisions must not count (Codex QA round 2): blockquote
    # lines are excluded ('>' is deliberately NOT in the decoration class —
    # v0.3.0 injects prior decisions as past_context, so the PM plausibly
    # quotes an old decision), and fenced code blocks are stripped before
    # matching. An unterminated fence strips to end-of-text: losing a real
    # decision to UNKNOWN (loud, gets investigated) beats silently adopting
    # a stale one.
    pm_text = re.sub(r'```.*?(?:```|\Z)', '', full_signal, flags=re.DOTALL)
    pm_pattern = (
        r'^[\s#*-]{0,12}'
        r'(?:(?:Investment|Final|Overall|Portfolio|Trading|Recommended)\s+)?'
        r'(?:Recommendation|Rating|Action|Transaction\s+Proposal)'
        r'\*{0,2}\s*:\s*\*{0,2}'
        r'(Buy|Overweight|Hold|Underweight|Sell)\b'
    )
    pm_matches = re.findall(pm_pattern, pm_text, re.IGNORECASE | re.MULTILINE)
    if pm_matches:
        token = pm_matches[-1].title()
        return PM_RATING_MAP[token.upper()], token

    # Method 1: Look for FINAL TRANSACTION PROPOSAL line
    # Handles formats like:
    #   FINAL TRANSACTION PROPOSAL: **HOLD**
    #   FINAL TRANSACTION PROPOSAL: HOLD
    #   FINAL TRANSACTION PROPOSAL: **BUY** AAPL
    #   ## FINAL TRANSACTION PROPOSAL: **SELL**
    proposal_pattern = r'FINAL\s+TRANSACTION\s+PROPOSAL[:\s]*\*{0,2}(BUY|HOLD|SELL)\*{0,2}'
    proposals = re.findall(proposal_pattern, full_signal, re.IGNORECASE)

    if proposals:
        word = proposals[-1].upper()
        return word, word.title()

    # Method 2: Look for "MY RECOMMENDATION: <DECISION>" pattern
    recommendation_pattern = r'MY\s+RECOMMENDATION[:\s]*\*{0,2}(BUY|HOLD|SELL)\*{0,2}'
    recommendations = re.findall(recommendation_pattern, full_signal, re.IGNORECASE)

    if recommendations:
        word = recommendations[-1].upper()
        return word, word.title()

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
        word = decisions[-1].upper()
        return word, word.title()

    return "UNKNOWN", None


def extract_decision_detailed(full_signal: str) -> dict:
    """Extract the decision with the 5-level rating and extraction method.

    Returns:
        {"decision": BUY|HOLD|SELL|UNKNOWN,
         "rating_5": Buy|Overweight|Hold|Underweight|Sell|None,
         "method": rendered_structured_parse|regex|unknown}

    Method meanings (TRI-70):
      - rendered_structured_parse: the text matches v0.3.0's deterministic
        render_pm_decision() template -> the typed PortfolioDecision engaged
        (structured output worked) and its rating is recovered exactly.
      - regex: a free-text decision header was scraped (structured output
        did NOT produce the render — fallback path).
      - unknown: no decision found (loud; investigate, never count as HOLD).
    """
    if not full_signal or not isinstance(full_signal, str):
        return {"decision": "UNKNOWN", "rating_5": None, "method": "unknown"}

    # Fenced code (quoted/stale content) never counts — same rule as the regex path.
    stripped = _FENCED_CODE.sub("", full_signal)
    render_ratings = _PM_RENDER_RATING.findall(stripped)
    if render_ratings and _PM_RENDER_HEADERS.search(stripped):
        token = render_ratings[-1].title()
        return {
            "decision": PM_RATING_MAP[token.upper()],
            "rating_5": token,
            "method": "rendered_structured_parse",
        }

    decision, token = _regex_decision_detailed(full_signal)
    if decision == "UNKNOWN":
        return {"decision": "UNKNOWN", "rating_5": None, "method": "unknown"}
    return {"decision": decision, "rating_5": token, "method": "regex"}


def extract_decision(full_signal: str) -> str:
    """Extract the trade decision (BUY/HOLD/SELL/UNKNOWN) from pipeline text.

    Thin wrapper over extract_decision_detailed() — kept for backward
    compatibility with all existing callers and tests.
    """
    return extract_decision_detailed(full_signal)["decision"]


def deduplicate_repeated_blocks(text: str, min_block_length: int = 200) -> str:
    """Remove repeated text blocks from pipeline output.

    The TradingAgents pipeline sometimes repeats agent outputs when hitting
    max_recur_limit. This function detects and removes duplicate long lines/blocks.

    Strategy: any line >= min_block_length chars whose first min_block_length
    characters have already been seen is considered a duplicate and dropped.
    Short lines (separators, headers, etc.) are always kept.

    Args:
        text: Full pipeline output text
        min_block_length: Minimum line length to consider for dedup

    Returns:
        Text with duplicate long-line blocks removed
    """
    if not text or len(text) < min_block_length * 2:
        return text

    lines = text.split('\n')
    seen_keys: set = set()
    result_lines = []

    for line in lines:
        if len(line) >= min_block_length:
            key = line.strip()[:min_block_length]
            if key in seen_keys:
                continue
            seen_keys.add(key)
        result_lines.append(line)

    return '\n'.join(result_lines)
