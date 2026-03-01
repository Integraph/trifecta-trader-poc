"""
Enhanced LLM wrapper that prepends data citation instructions to local model calls.

The vendor agent prompts don't explicitly require data citations — they rely on
the model's natural tendency to cite data. Claude does this automatically; smaller
models need explicit instructions. This wrapper bridges that gap.

Usage:
    from src.enhanced_llm import create_enhanced_llm

    base_llm = ChatOpenAI(model="qwen2.5:14b", ...)
    enhanced_llm = create_enhanced_llm(base_llm, style="financial_analysis")
"""

import logging
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


# The enhancement prefix — based on Phase 1 experiment results.
# Phase 1 found:
#   - Variant C (structured output with sections) gets Qwen 14B to 27 numbers (93% of Claude's 29)
#   - Variant B (explicit citation count) gets Qwen 14B to 22 numbers for trader (76% of Claude's)
# This prefix combines both approaches: explicit count + structured output requirements.

FINANCIAL_ANALYSIS_PREFIX = """CRITICAL REQUIREMENTS FOR YOUR ANALYSIS:
- You MUST cite at least 10 specific data points using their exact values (dollar amounts, percentages, ratios, multiples)
- For EVERY key claim, include the exact number from the provided data
- Structure your analysis to include:
  * KEY METRICS: List exact values from the data (prices, ratios, growth rates, volumes)
  * For valuation: cite P/E, Forward P/E, EPS, and revenue figures with exact values
  * For technicals: cite RSI, MACD, SMA levels with their exact numerical values
  * For risk management: provide stop-loss (exact price + % risk), price target (exact + % upside), risk/reward ratio, and position size (% of portfolio)
- Every sentence should contain at least one specific number from the data
- Prioritize data density and quantitative precision over narrative prose
"""

# Alternative: structured output style (use if variant C wins)
STRUCTURED_OUTPUT_PREFIX = """IMPORTANT: Structure your analysis with specific data points.
Always include exact dollar values, percentages, and ratios from the data.
For trading decisions, always calculate and state:
- Entry price (exact), Stop-loss (exact price and % risk), Target (exact price and % upside)
- Risk/reward ratio, Position sizing (% of portfolio)
Reference at least 10 specific numbers from the provided data.
"""

# Alternative: few-shot style (use if variant D wins)
FEW_SHOT_PREFIX = """When analyzing financial data, be extremely specific with numbers.
For example, instead of "the stock is near its moving average", say
"price at $270.15 is 1.9% above the 50-day SMA of $265.20 and 6.8% above the 200-day SMA of $252.80."
Always calculate exact percentages, risk/reward ratios, and position sizes.
Cite at least 10 specific data points from the data provided.
"""

ENHANCEMENT_STYLES = {
    "financial_analysis": FINANCIAL_ANALYSIS_PREFIX,
    "structured": STRUCTURED_OUTPUT_PREFIX,
    "few_shot": FEW_SHOT_PREFIX,
}


class EnhancedChatModel:
    """Wraps a LangChain chat model to prepend enhancement instructions.

    This wrapper intercepts calls to invoke() and prepends a system-level
    instruction that improves data citation quality for local models.

    It handles both string and message-list invocations used by different
    vendor agents.
    """

    def __init__(self, base_llm: BaseChatModel, prefix: str):
        self._base_llm = base_llm
        self._prefix = prefix

    def invoke(self, input: Any, **kwargs) -> Any:
        """Intercept invoke calls and prepend enhancement instructions."""
        if isinstance(input, str):
            enhanced = f"{self._prefix}\n\n{input}"
            return self._base_llm.invoke(enhanced, **kwargs)
        elif isinstance(input, list):
            enhanced_messages = [SystemMessage(content=self._prefix)] + [
                m if isinstance(m, BaseMessage) else
                (SystemMessage(content=m["content"]) if m.get("role") == "system"
                 else HumanMessage(content=m["content"]))
                for m in input
            ]
            return self._base_llm.invoke(enhanced_messages, **kwargs)
        else:
            logger.warning("EnhancedChatModel: unknown input type %s, passing through", type(input))
            return self._base_llm.invoke(input, **kwargs)

    def bind_tools(self, *args, **kwargs):
        """Pass through to base LLM — tool binding should NOT be enhanced."""
        return self._base_llm.bind_tools(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy all other attributes to the base LLM."""
        return getattr(self._base_llm, name)


def create_enhanced_llm(
    base_llm: BaseChatModel,
    style: str = "financial_analysis",
) -> EnhancedChatModel:
    """Create an enhanced LLM wrapper.

    Args:
        base_llm: The base LangChain chat model to wrap
        style: Enhancement style — one of 'financial_analysis', 'structured', 'few_shot'

    Returns:
        EnhancedChatModel that prepends instructions to every invocation
    """
    if style not in ENHANCEMENT_STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(ENHANCEMENT_STYLES.keys())}")

    prefix = ENHANCEMENT_STYLES[style]
    logger.info("Creating enhanced LLM wrapper with style '%s'", style)
    return EnhancedChatModel(base_llm, prefix)
