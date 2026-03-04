"""Tests for Task 010 cost optimization features.

Covers:
- New Haiku hybrid configs are properly constructed
- DataCache TTL expiry, key generation, set/get, clear
- TokenUsageCallback cost breakdown calculation
- Cache stats integration
"""

import json
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestHybridHaikuConfigs:

    def test_hybrid_haiku_tools_exists(self):
        """hybrid_haiku_tools config must exist in CONFIGS."""
        from src.hybrid_llm import CONFIGS
        assert "hybrid_haiku_tools" in CONFIGS

    def test_hybrid_haiku_aggressive_exists(self):
        """hybrid_haiku_aggressive config must exist in CONFIGS."""
        from src.hybrid_llm import CONFIGS
        assert "hybrid_haiku_aggressive" in CONFIGS

    def test_hybrid_haiku_tools_tool_model(self):
        """hybrid_haiku_tools must use Haiku for tool-calling agents."""
        from src.hybrid_llm import CONFIGS
        cfg = CONFIGS["hybrid_haiku_tools"]
        assert "haiku" in cfg.tool_model.lower()

    def test_hybrid_haiku_tools_deep_model(self):
        """hybrid_haiku_tools must keep Sonnet for deep reasoning."""
        from src.hybrid_llm import CONFIGS
        cfg = CONFIGS["hybrid_haiku_tools"]
        assert "sonnet" in cfg.reasoning_deep_model.lower()

    def test_hybrid_haiku_tools_quick_local(self):
        """hybrid_haiku_tools must use Ollama for quick reasoning."""
        from src.hybrid_llm import CONFIGS
        cfg = CONFIGS["hybrid_haiku_tools"]
        assert cfg.reasoning_quick_provider == "ollama"

    def test_hybrid_haiku_aggressive_all_cheap(self):
        """hybrid_haiku_aggressive must use Haiku + local for everything."""
        from src.hybrid_llm import CONFIGS
        cfg = CONFIGS["hybrid_haiku_aggressive"]
        assert "haiku" in cfg.tool_model.lower()
        assert cfg.reasoning_quick_provider == "ollama"
        assert cfg.reasoning_deep_provider == "ollama"

    def test_hybrid_haiku_tools_has_enhance_deep(self):
        """hybrid_haiku_tools must have enhance_deep=True for structured output."""
        from src.hybrid_llm import CONFIGS
        cfg = CONFIGS["hybrid_haiku_tools"]
        assert cfg.enhance_deep is True

    def test_haiku_model_string_format(self):
        """Haiku model string must follow the expected format."""
        from src.hybrid_llm import CONFIGS
        cfg = CONFIGS["hybrid_haiku_tools"]
        assert cfg.tool_model == "claude-haiku-4-5-20251001"

    def test_to_dict_includes_models(self):
        """to_dict() should include all three model tiers."""
        from src.hybrid_llm import CONFIGS
        d = CONFIGS["hybrid_haiku_tools"].to_dict()
        assert "tool_calling" in d
        assert "reasoning_quick" in d
        assert "reasoning_deep" in d
        assert "haiku" in d["tool_calling"].lower()


# ---------------------------------------------------------------------------
# DataCache tests
# ---------------------------------------------------------------------------

class TestDataCache:

    def test_key_for_format(self):
        """key_for() should return 'TICKER:analyst_type:YYYY-MM-DD'."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            key = cache.key_for("aapl", "market", "2026-03-02")
            assert key == "AAPL:market:2026-03-02"

    def test_key_for_uppercases_ticker(self):
        """Ticker in key should always be uppercase."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            assert cache.key_for("tsla", "news", "2026-03-02").startswith("TSLA:")

    def test_set_and_get_basic(self):
        """set() then get() should return the stored value."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp, default_ttl=60)
            key = cache.key_for("AAPL", "market", "2026-03-02")
            cache.set(key, "Market report content", ttl=60)
            result = cache.get(key)
            assert result == "Market report content"

    def test_get_returns_none_on_miss(self):
        """get() should return None for a key that was never set."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            assert cache.get("AAPL:market:2026-03-02") is None

    def test_ttl_expiry(self):
        """Expired entries should return None on get()."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            key = cache.key_for("AAPL", "market", "2026-03-02")
            cache.set(key, "Should expire", ttl=1)

            # Manually back-date the expiry
            path = cache._path_for(key)
            with open(path) as f:
                entry = json.load(f)
            entry["expires_at"] = time.time() - 1  # Already expired
            with open(path, "w") as f:
                json.dump(entry, f)

            assert cache.get(key) is None

    def test_empty_value_not_stored(self):
        """set() should not store empty string values."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            key = cache.key_for("AAPL", "market", "2026-03-02")
            cache.set(key, "", ttl=60)
            assert cache.get(key) is None

    def test_clear_by_ticker(self):
        """clear(ticker=X) should remove only that ticker's entries."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            key_aapl = cache.key_for("AAPL", "market", "2026-03-02")
            key_tsla = cache.key_for("TSLA", "market", "2026-03-02")
            cache.set(key_aapl, "AAPL data", ttl=3600)
            cache.set(key_tsla, "TSLA data", ttl=3600)

            deleted = cache.clear(ticker="AAPL")

            assert deleted == 1
            assert cache.get(key_aapl) is None
            assert cache.get(key_tsla) == "TSLA data"

    def test_clear_all(self):
        """clear() with no ticker should remove all cache entries."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            for analyst in ["market", "news", "social", "fundamentals"]:
                key = cache.key_for("AAPL", analyst, "2026-03-02")
                cache.set(key, f"{analyst} data", ttl=3600)

            deleted = cache.clear()
            assert deleted == 4

    def test_key_consistency(self):
        """Same inputs should always produce the same cache key."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            k1 = cache.key_for("AAPL", "fundamentals", "2026-03-02")
            k2 = cache.key_for("AAPL", "fundamentals", "2026-03-02")
            assert k1 == k2

    def test_stats_tracks_hits_and_misses(self):
        """stats() should accurately count hits and misses."""
        from src.data_cache import DataCache
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            key = cache.key_for("AAPL", "market", "2026-03-02")

            cache.get(key)           # miss
            cache.set(key, "data", ttl=3600)
            cache.get(key)           # hit
            cache.get(key)           # hit

            stats = cache.stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["total"] == 3
            assert stats["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)


# ---------------------------------------------------------------------------
# TokenUsageCallback tests
# ---------------------------------------------------------------------------

class TestTokenUsageCallback:

    def _make_response(self, model: str, input_tokens: int, output_tokens: int):
        """Build a mock LLM response with usage_metadata."""
        msg = MagicMock()
        msg.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        msg.response_metadata = {"model": model}
        gen = MagicMock()
        gen.message = msg
        response = MagicMock()
        response.generations = [[gen]]
        return response

    def test_cost_breakdown_haiku_cheaper_than_sonnet(self):
        """Haiku cost per token should be less than Sonnet cost per token."""
        from src.hybrid_graph import TokenUsageCallback, MODEL_PRICING
        haiku_price = MODEL_PRICING["claude-haiku-4-5-20251001"]["output"]
        sonnet_price = MODEL_PRICING["claude-sonnet-4-5-20250929"]["output"]
        assert haiku_price < sonnet_price

    def test_token_accumulation(self):
        """on_llm_end() should accumulate token counts across calls."""
        from src.hybrid_graph import TokenUsageCallback
        cb = TokenUsageCallback()
        resp1 = self._make_response("claude-haiku-4-5-20251001", 1000, 500)
        resp2 = self._make_response("claude-haiku-4-5-20251001", 2000, 800)
        cb.on_llm_end(resp1)
        cb.on_llm_end(resp2)

        breakdown = cb.cost_breakdown()
        haiku = breakdown["by_model"].get("claude-haiku-4-5-20251001", {})
        assert haiku["input_tokens"] == 3000
        assert haiku["output_tokens"] == 1300
        assert haiku["calls"] == 2

    def test_cost_calculation_haiku(self):
        """Cost should be computed correctly from pricing table."""
        from src.hybrid_graph import TokenUsageCallback, MODEL_PRICING
        cb = TokenUsageCallback()
        # 1M input tokens + 1M output tokens at Haiku pricing
        resp = self._make_response("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        cb.on_llm_end(resp)

        breakdown = cb.cost_breakdown()
        haiku = breakdown["by_model"]["claude-haiku-4-5-20251001"]
        pricing = MODEL_PRICING["claude-haiku-4-5-20251001"]
        expected = pricing["input"] + pricing["output"]  # $/1M * 1M tokens each
        assert abs(haiku["cost_usd"] - expected) < 0.001

    def test_total_usd_sums_across_models(self):
        """total_usd should be the sum across all models."""
        from src.hybrid_graph import TokenUsageCallback
        cb = TokenUsageCallback()
        cb.on_llm_end(self._make_response("claude-haiku-4-5-20251001", 100_000, 50_000))
        cb.on_llm_end(self._make_response("claude-sonnet-4-5-20250929", 50_000, 10_000))

        breakdown = cb.cost_breakdown()
        by_model_total = sum(v["cost_usd"] for v in breakdown["by_model"].values())
        assert abs(breakdown["total_usd"] - by_model_total) < 0.000001

    def test_cache_stats_included_in_breakdown(self):
        """cost_breakdown() should include cache stats when provided."""
        from src.hybrid_graph import TokenUsageCallback
        cb = TokenUsageCallback()
        cache_stats = {"hits": 3, "misses": 1, "total": 4, "hit_rate_pct": 75.0}
        breakdown = cb.cost_breakdown(cache_stats=cache_stats)
        assert breakdown["cache_hits"] == 3
        assert breakdown["cache_misses"] == 1
        assert breakdown["cache_hit_rate_pct"] == 75.0

    def test_callback_does_not_raise_on_bad_response(self):
        """on_llm_end() should silently handle malformed responses."""
        from src.hybrid_graph import TokenUsageCallback
        cb = TokenUsageCallback()
        bad_response = MagicMock()
        bad_response.generations = [None]  # Will cause AttributeError
        cb.on_llm_end(bad_response)  # Should not raise


# ---------------------------------------------------------------------------
# Qwen 3.5 config tests (Task 011)
# ---------------------------------------------------------------------------

class TestQwen35Configs:
    """Verify the three new Qwen 3.5 benchmark configs are correctly constructed."""

    QWEN35_CONFIGS = [
        "hybrid_haiku_qwen35_27b",
        "hybrid_haiku_qwen35_35b",
        "hybrid_haiku_qwen35_9b",
    ]

    EXPECTED_MODELS = {
        "hybrid_haiku_qwen35_27b":  "qwen3.5:27b",
        "hybrid_haiku_qwen35_35b":  "qwen3.5:35b-a3b",
        "hybrid_haiku_qwen35_9b":   "qwen3.5:9b",
    }

    def test_all_qwen35_configs_exist(self):
        """All three Qwen 3.5 configs must be present in CONFIGS."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            assert name in CONFIGS, f"{name} missing from CONFIGS"

    def test_qwen35_configs_use_haiku_for_tool_calling(self):
        """All Qwen 3.5 configs must use Haiku for tool-calling (same as baseline)."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            assert cfg.tool_provider == "anthropic"
            assert "haiku" in cfg.tool_model.lower(), (
                f"{name}: expected Haiku tool model, got {cfg.tool_model}"
            )

    def test_qwen35_configs_use_sonnet_for_deep_reasoning(self):
        """All Qwen 3.5 configs must use Sonnet as Risk Judge (same as baseline)."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            assert cfg.reasoning_deep_provider == "anthropic"
            assert "sonnet" in cfg.reasoning_deep_model.lower(), (
                f"{name}: expected Sonnet deep model, got {cfg.reasoning_deep_model}"
            )

    def test_qwen35_configs_use_ollama_for_quick_reasoning(self):
        """All Qwen 3.5 configs must use Ollama for the quick reasoning tier."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            assert cfg.reasoning_quick_provider == "ollama", (
                f"{name}: expected ollama provider, got {cfg.reasoning_quick_provider}"
            )

    def test_qwen35_model_strings_are_correct(self):
        """Each config must reference the right Qwen 3.5 model tag."""
        from src.hybrid_llm import CONFIGS
        for config_name, expected_model in self.EXPECTED_MODELS.items():
            cfg = CONFIGS[config_name]
            assert cfg.reasoning_quick_model == expected_model, (
                f"{config_name}: expected model '{expected_model}', "
                f"got '{cfg.reasoning_quick_model}'"
            )

    def test_qwen35_configs_have_enhance_local(self):
        """All Qwen 3.5 configs must have enhance_local=True for prompt enhancement."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            assert cfg.enhance_local is True, f"{name}: enhance_local should be True"

    def test_qwen35_configs_have_enhance_deep(self):
        """All Qwen 3.5 configs must have enhance_deep=True for structured output."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            assert cfg.enhance_deep is True, f"{name}: enhance_deep should be True"

    def test_qwen35_configs_match_baseline_except_model(self):
        """Qwen 3.5 configs should differ from hybrid_haiku_tools only in quick model."""
        from src.hybrid_llm import CONFIGS
        baseline = CONFIGS["hybrid_haiku_tools"]
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            assert cfg.tool_model == baseline.tool_model, \
                f"{name}: tool_model differs from baseline"
            assert cfg.reasoning_deep_model == baseline.reasoning_deep_model, \
                f"{name}: reasoning_deep_model differs from baseline"
            assert cfg.enhance_style == baseline.enhance_style, \
                f"{name}: enhance_style differs from baseline"
            # Only the quick model should differ
            assert cfg.reasoning_quick_model != baseline.reasoning_quick_model, \
                f"{name}: reasoning_quick_model should differ from baseline"

    def test_qwen35_model_string_format(self):
        """Qwen 3.5 model strings must follow the 'qwen3.5:variant' format."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            cfg = CONFIGS[name]
            model = cfg.reasoning_quick_model
            assert model.startswith("qwen3.5:"), (
                f"{name}: model '{model}' should start with 'qwen3.5:'"
            )

    def test_qwen35_to_dict_shows_ollama_prefix(self):
        """to_dict() should show 'ollama/qwen3.5:...' in reasoning_quick field."""
        from src.hybrid_llm import CONFIGS
        for name in self.QWEN35_CONFIGS:
            d = CONFIGS[name].to_dict()
            assert d["reasoning_quick"].startswith("ollama/qwen3.5:"), (
                f"{name}: to_dict reasoning_quick should have ollama prefix, "
                f"got: {d['reasoning_quick']}"
            )
