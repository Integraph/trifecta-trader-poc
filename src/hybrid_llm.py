"""
Hybrid LLM router for assigning different providers to different agents.

This module creates a patched version of TradingAgentsGraph that supports
per-agent LLM provider routing. Tool-calling agents use cloud LLMs (Anthropic),
while pure-reasoning agents can use local Ollama models.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class HybridLLMConfig:
    """Configuration for hybrid LLM routing.

    Defines which provider/model to use for each agent category:
    - tool_calling: Agents that use bind_tools() -- must support tool calling
    - reasoning_quick: Pure reasoning agents on the quick path
    - reasoning_deep: Pure reasoning agents on the deep path (judges)
    - enhance_local: Whether to wrap local models with data-citation prefix
    - enhance_style: Which prefix style to use (see src/enhanced_llm.py)
    """

    def __init__(
        self,
        tool_provider: str = "anthropic",
        tool_model: str = "claude-sonnet-4-5-20250929",
        reasoning_quick_provider: str = "ollama",
        reasoning_quick_model: str = "qwen2.5:14b",
        reasoning_deep_provider: str = "anthropic",
        reasoning_deep_model: str = "claude-sonnet-4-5-20250929",
        enhance_local: bool = False,
        enhance_style: str = "financial_analysis",
    ):
        self.tool_provider = tool_provider
        self.tool_model = tool_model
        self.reasoning_quick_provider = reasoning_quick_provider
        self.reasoning_quick_model = reasoning_quick_model
        self.reasoning_deep_provider = reasoning_deep_provider
        self.reasoning_deep_model = reasoning_deep_model
        self.enhance_local = enhance_local
        self.enhance_style = enhance_style

    def to_dict(self) -> Dict[str, Any]:
        """Return a summary dict for logging/reporting."""
        d = {
            "tool_calling": f"{self.tool_provider}/{self.tool_model}",
            "reasoning_quick": f"{self.reasoning_quick_provider}/{self.reasoning_quick_model}",
            "reasoning_deep": f"{self.reasoning_deep_provider}/{self.reasoning_deep_model}",
        }
        if self.enhance_local:
            d["enhance_style"] = self.enhance_style
        return d


CONFIGS = {
    "all_cloud": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="anthropic",
        reasoning_quick_model="claude-sonnet-4-5-20250929",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    "hybrid_qwen": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    "hybrid_mistral": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="mistral-small:22b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    "hybrid_aggressive_qwen": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="qwen2.5:14b",
    ),

    "hybrid_aggressive_mistral": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="mistral-small:22b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="mistral-small:22b",
    ),

    "hybrid_qwen32": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:32b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    ),

    "hybrid_aggressive_qwen32": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:32b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="qwen2.5:32b",
    ),

    "hybrid_qwen_enhanced": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-sonnet-4-5-20250929",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=True,
        enhance_style="financial_analysis",
    ),
}


def create_hybrid_llms(hybrid_config: HybridLLMConfig) -> Dict[str, Any]:
    """Create three LLM instances for hybrid routing.

    Returns:
        Dict with keys: 'tool_calling_llm', 'reasoning_quick_llm', 'reasoning_deep_llm'
    """
    from tradingagents.llm_clients.factory import create_llm_client

    llms = {}

    tool_client = create_llm_client(
        provider=hybrid_config.tool_provider,
        model=hybrid_config.tool_model,
    )
    llms["tool_calling_llm"] = tool_client.get_llm()

    reasoning_quick_client = create_llm_client(
        provider=hybrid_config.reasoning_quick_provider,
        model=hybrid_config.reasoning_quick_model,
    )
    reasoning_quick_llm = reasoning_quick_client.get_llm()
    if hybrid_config.enhance_local and hybrid_config.reasoning_quick_provider == "ollama":
        from src.enhanced_llm import create_enhanced_llm
        reasoning_quick_llm = create_enhanced_llm(reasoning_quick_llm, style=hybrid_config.enhance_style)
        logger.info("Enhanced reasoning_quick_llm with style '%s'", hybrid_config.enhance_style)
    llms["reasoning_quick_llm"] = reasoning_quick_llm

    reasoning_deep_client = create_llm_client(
        provider=hybrid_config.reasoning_deep_provider,
        model=hybrid_config.reasoning_deep_model,
    )
    reasoning_deep_llm = reasoning_deep_client.get_llm()
    if hybrid_config.enhance_local and hybrid_config.reasoning_deep_provider == "ollama":
        from src.enhanced_llm import create_enhanced_llm
        reasoning_deep_llm = create_enhanced_llm(reasoning_deep_llm, style=hybrid_config.enhance_style)
        logger.info("Enhanced reasoning_deep_llm with style '%s'", hybrid_config.enhance_style)
    llms["reasoning_deep_llm"] = reasoning_deep_llm

    logger.info("Hybrid LLM routing: %s", hybrid_config.to_dict())
    return llms
