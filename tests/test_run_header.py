"""Tests for the analysis run header (_format_run_header).

Regression guard for the TRI-69 audit issue: in HYBRID mode the header
printed config['deep_think_llm'] / config['quick_think_llm'], which do not
reflect the per-slot models actually routed by src/hybrid_llm.py.
"""

from src.hybrid_llm import HybridLLMConfig
from src.run_analysis import _format_run_header

CONFIG = {"deep_think_llm": "claude-sonnet-4-5-20250929",
          "quick_think_llm": "claude-sonnet-4-5-20250929"}


class TestNonHybridHeader:

    def test_shows_provider_and_config_models(self):
        header = _format_run_header("AAPL", "2026-07-01", "anthropic", CONFIG)
        assert "Ticker:    AAPL" in header
        assert "Date:      2026-07-01" in header
        assert "Provider:  anthropic" in header
        assert "Deep LLM:  claude-sonnet-4-5-20250929" in header
        assert "Quick LLM: claude-sonnet-4-5-20250929" in header
        assert "Mode:" not in header
        assert "Tool LLM:" not in header


class TestHybridHeader:
    """HYBRID header must report the hybrid config's actual slots."""

    # Mirrors tri69_config_a: all three slots differ from the config dict.
    HYBRID_CONFIG = HybridLLMConfig(
        tool_provider="ollama",
        tool_model="qwen3-coder:30b",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen3.5:9b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
    )

    def test_shows_all_three_slots_with_provider(self):
        header = _format_run_header("AAPL", "2026-07-01", "anthropic",
                                    CONFIG, hybrid="tri69_config_a",
                                    hybrid_config=self.HYBRID_CONFIG)
        assert "Mode:      HYBRID (tri69_config_a)" in header
        assert "Tool LLM:  ollama/qwen3-coder:30b" in header
        assert "Quick LLM: ollama/qwen3.5:9b" in header
        assert "Deep LLM:  anthropic/claude-sonnet-4-5-20250929" in header

    def test_does_not_leak_config_dict_models(self):
        config = {"deep_think_llm": "wrong-deep-model",
                  "quick_think_llm": "wrong-quick-model"}
        header = _format_run_header("NVDA", "2026-07-01", "anthropic",
                                    config, hybrid="tri69_config_a",
                                    hybrid_config=self.HYBRID_CONFIG)
        assert "wrong-deep-model" not in header
        assert "wrong-quick-model" not in header
        assert "Provider:  anthropic" not in header
