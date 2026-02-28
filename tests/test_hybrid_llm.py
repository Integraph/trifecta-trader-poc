"""Tests for hybrid LLM routing infrastructure."""

import pytest
from src.hybrid_llm import HybridLLMConfig, CONFIGS


class TestHybridLLMConfig:
    """Test configuration objects."""

    def test_default_config(self):
        config = HybridLLMConfig()
        assert config.tool_provider == "anthropic"
        assert config.reasoning_quick_provider == "ollama"
        assert config.reasoning_deep_provider == "anthropic"

    def test_predefined_configs_exist(self):
        assert "all_cloud" in CONFIGS
        assert "hybrid_qwen" in CONFIGS
        assert "hybrid_mistral" in CONFIGS
        assert "hybrid_aggressive_qwen" in CONFIGS
        assert "hybrid_aggressive_mistral" in CONFIGS

    def test_all_cloud_config(self):
        config = CONFIGS["all_cloud"]
        assert config.tool_provider == "anthropic"
        assert config.reasoning_quick_provider == "anthropic"
        assert config.reasoning_deep_provider == "anthropic"

    def test_hybrid_qwen_config(self):
        config = CONFIGS["hybrid_qwen"]
        assert config.tool_provider == "anthropic"
        assert config.reasoning_quick_provider == "ollama"
        assert config.reasoning_quick_model == "qwen2.5:14b"
        assert config.reasoning_deep_provider == "anthropic"

    def test_hybrid_mistral_config(self):
        config = CONFIGS["hybrid_mistral"]
        assert config.reasoning_quick_model == "mistral-small:22b"
        assert config.reasoning_deep_provider == "anthropic"

    def test_aggressive_configs_use_local_for_deep(self):
        for name in ("hybrid_aggressive_qwen", "hybrid_aggressive_mistral"):
            config = CONFIGS[name]
            assert config.tool_provider == "anthropic"
            assert config.reasoning_deep_provider == "ollama"

    def test_to_dict(self):
        config = CONFIGS["hybrid_qwen"]
        d = config.to_dict()
        assert "tool_calling" in d
        assert "reasoning_quick" in d
        assert "reasoning_deep" in d
        assert "ollama" in d["reasoning_quick"]
        assert "anthropic" in d["tool_calling"]

    def test_custom_config(self):
        config = HybridLLMConfig(
            tool_provider="anthropic",
            tool_model="claude-sonnet-4-5-20250929",
            reasoning_quick_provider="ollama",
            reasoning_quick_model="llama3.1:8b",
            reasoning_deep_provider="ollama",
            reasoning_deep_model="mistral-small:22b",
        )
        assert config.reasoning_quick_model == "llama3.1:8b"
        assert config.reasoning_deep_model == "mistral-small:22b"

    def test_to_dict_format(self):
        config = HybridLLMConfig(
            tool_provider="anthropic",
            tool_model="claude-sonnet-4-5-20250929",
            reasoning_quick_provider="ollama",
            reasoning_quick_model="qwen2.5:14b",
            reasoning_deep_provider="anthropic",
            reasoning_deep_model="claude-sonnet-4-5-20250929",
        )
        d = config.to_dict()
        assert d["tool_calling"] == "anthropic/claude-sonnet-4-5-20250929"
        assert d["reasoning_quick"] == "ollama/qwen2.5:14b"
        assert d["reasoning_deep"] == "anthropic/claude-sonnet-4-5-20250929"


class TestHybridGraphSetup:
    """Test the hybrid graph setup module is importable and structured correctly."""

    def test_hybrid_graph_setup_importable(self):
        from src.hybrid_graph import HybridGraphSetup, HybridTradingGraph
        assert HybridGraphSetup is not None
        assert HybridTradingGraph is not None

    def test_hybrid_graph_setup_accepts_three_llms(self):
        """Verify HybridGraphSetup constructor signature accepts three LLMs."""
        from src.hybrid_graph import HybridGraphSetup
        import inspect
        sig = inspect.signature(HybridGraphSetup.__init__)
        params = list(sig.parameters.keys())
        assert "tool_llm" in params
        assert "reasoning_quick_llm" in params
        assert "reasoning_deep_llm" in params
