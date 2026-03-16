"""
Task 018 — Tests: hybrid_llm.py YAML load/save/delete/reload
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ── Global CONFIGS isolation (prevents cross-test contamination) ───────────────

@pytest.fixture(autouse=True)
def restore_configs():
    """Save and restore the module-level CONFIGS dict around every test."""
    import src.hybrid_llm as mod
    from src.hybrid_llm import HybridLLMConfig
    snapshot = {k: HybridLLMConfig(**v.to_flat_dict()) for k, v in mod.CONFIGS.items()}
    yield
    mod.CONFIGS.clear()
    mod.CONFIGS.update(snapshot)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_yaml(tmp_path):
    """Return a path to a temporary YAML file."""
    return tmp_path / "hybrid_llm.yaml"


@pytest.fixture
def yaml_with_two_configs(tmp_yaml):
    """Pre-populate YAML with two configs."""
    data = {
        "configs": {
            "config_a": {
                "tool_provider": "anthropic",
                "tool_model": "claude-haiku-4-5-20251001",
                "reasoning_quick_provider": "ollama",
                "reasoning_quick_model": "qwen2.5:14b",
                "reasoning_deep_provider": "anthropic",
                "reasoning_deep_model": "claude-sonnet-4-5-20250929",
                "enhance_local": False,
                "enhance_style": "financial_analysis",
                "enhance_deep": False,
                "enhance_deep_style": "execution_params_only",
            },
            "config_b": {
                "tool_provider": "openai",
                "tool_model": "gpt-4o",
                "reasoning_quick_provider": "ollama",
                "reasoning_quick_model": "qwen2.5:32b",
                "reasoning_deep_provider": "anthropic",
                "reasoning_deep_model": "claude-sonnet-4-5-20250929",
                "enhance_local": True,
                "enhance_style": "structured",
                "enhance_deep": True,
                "enhance_deep_style": "few_shot",
            },
        }
    }
    tmp_yaml.write_text(yaml.safe_dump(data))
    return tmp_yaml


# ── HybridLLMConfig unit tests ─────────────────────────────────────────────────

def test_hybrid_config_defaults():
    from src.hybrid_llm import HybridLLMConfig
    cfg = HybridLLMConfig()
    assert cfg.tool_provider == "anthropic"
    assert cfg.enhance_local is False
    assert cfg.enhance_deep is False


def test_hybrid_config_to_flat_dict():
    from src.hybrid_llm import HybridLLMConfig
    cfg = HybridLLMConfig(tool_provider="openai", tool_model="gpt-4o", enhance_local=True)
    d = cfg.to_flat_dict()
    assert d["tool_provider"] == "openai"
    assert d["tool_model"] == "gpt-4o"
    assert d["enhance_local"] is True
    assert len(d) == 10


def test_hybrid_config_to_dict_summary():
    from src.hybrid_llm import HybridLLMConfig
    cfg = HybridLLMConfig(enhance_local=True, enhance_style="structured")
    d = cfg.to_dict()
    assert "tool_calling" in d
    assert "reasoning_quick" in d
    assert "reasoning_deep" in d
    assert d["enhance_style"] == "structured"


def test_hybrid_config_to_dict_no_enhance():
    from src.hybrid_llm import HybridLLMConfig
    cfg = HybridLLMConfig(enhance_local=False, enhance_deep=False)
    d = cfg.to_dict()
    assert "enhance_style" not in d
    assert "enhance_deep_style" not in d


# ── YAML read/write helpers ────────────────────────────────────────────────────

def test_write_and_read_yaml(tmp_yaml):
    """_write_yaml then _read_yaml should round-trip correctly."""
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", tmp_yaml):
        data = {"configs": {"test": {"tool_provider": "openai"}}}
        module._write_yaml(data)
        assert tmp_yaml.exists()
        loaded = module._read_yaml()
        assert loaded["configs"]["test"]["tool_provider"] == "openai"


def test_read_yaml_missing_file(tmp_yaml):
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", tmp_yaml):
        result = module._read_yaml()
        assert result == {}


def test_yaml_entry_to_config():
    from src.hybrid_llm import _yaml_entry_to_config
    entry = {
        "tool_provider": "openai",
        "tool_model": "gpt-4o",
        "reasoning_quick_provider": "ollama",
        "reasoning_quick_model": "qwen2.5:14b",
        "reasoning_deep_provider": "anthropic",
        "reasoning_deep_model": "claude-sonnet-4-5-20250929",
        "enhance_local": True,
        "enhance_style": "structured",
        "enhance_deep": False,
        "enhance_deep_style": "execution_params_only",
    }
    cfg = _yaml_entry_to_config(entry)
    assert cfg.tool_provider == "openai"
    assert cfg.enhance_local is True
    assert cfg.enhance_style == "structured"


def test_yaml_entry_to_config_defaults():
    """Should fill in defaults for missing keys."""
    from src.hybrid_llm import _yaml_entry_to_config
    cfg = _yaml_entry_to_config({})
    assert cfg.tool_provider == "anthropic"
    assert cfg.enhance_local is False


# ── load_configs ───────────────────────────────────────────────────────────────

def test_load_configs_from_yaml(yaml_with_two_configs):
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", yaml_with_two_configs):
        configs = module.load_configs()
        assert "config_a" in configs
        assert "config_b" in configs
        assert configs["config_a"].tool_provider == "anthropic"
        assert configs["config_b"].tool_provider == "openai"
        assert configs["config_b"].enhance_local is True


def test_load_configs_generates_yaml_if_missing(tmp_yaml):
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", tmp_yaml):
        assert not tmp_yaml.exists()
        configs = module.load_configs()
        assert tmp_yaml.exists()
        # Should have loaded all 13 defaults
        assert len(configs) == 13
        assert "all_cloud" in configs


def test_load_configs_falls_back_on_empty_yaml(tmp_yaml):
    """Empty YAML file returns defaults."""
    import src.hybrid_llm as module
    tmp_yaml.write_text("configs: {}")
    with patch.object(module, "_YAML_PATH", tmp_yaml):
        configs = module.load_configs()
        assert len(configs) == 13  # falls back to _DEFAULT_CONFIGS


# ── save_config ────────────────────────────────────────────────────────────────

def test_save_config_creates_entry(yaml_with_two_configs):
    import src.hybrid_llm as module
    from src.hybrid_llm import HybridLLMConfig
    with patch.object(module, "_YAML_PATH", yaml_with_two_configs):
        cfg = HybridLLMConfig(tool_provider="openai", tool_model="gpt-4o-mini")
        # Temporarily update CONFIGS so save_config can mutate it
        original = dict(module.CONFIGS)
        module.CONFIGS.update(module.load_configs())
        module.save_config("new_config", cfg)

        # YAML should now have new_config
        with open(yaml_with_two_configs) as f:
            data = yaml.safe_load(f)
        assert "new_config" in data["configs"]
        assert data["configs"]["new_config"]["tool_provider"] == "openai"
        module.CONFIGS.clear()
        module.CONFIGS.update(original)


def test_save_config_updates_memory(yaml_with_two_configs):
    import src.hybrid_llm as module
    from src.hybrid_llm import HybridLLMConfig
    with patch.object(module, "_YAML_PATH", yaml_with_two_configs):
        module.CONFIGS.update(module.load_configs())
        cfg = HybridLLMConfig(tool_provider="xai", tool_model="grok-2")
        module.save_config("xai_config", cfg)
        assert "xai_config" in module.CONFIGS
        assert module.CONFIGS["xai_config"].tool_provider == "xai"


# ── delete_config ──────────────────────────────────────────────────────────────

def test_delete_config_removes_entry(yaml_with_two_configs):
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", yaml_with_two_configs):
        module.CONFIGS.update(module.load_configs())
        module.delete_config("config_b")

        with open(yaml_with_two_configs) as f:
            data = yaml.safe_load(f)
        assert "config_b" not in data["configs"]
        assert "config_a" in data["configs"]  # not affected


def test_delete_config_raises_key_error(yaml_with_two_configs):
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", yaml_with_two_configs):
        module.CONFIGS.update(module.load_configs())
        with pytest.raises(KeyError):
            module.delete_config("nonexistent")


# ── reload_configs ─────────────────────────────────────────────────────────────

def test_reload_configs_updates_memory(yaml_with_two_configs):
    import src.hybrid_llm as module
    with patch.object(module, "_YAML_PATH", yaml_with_two_configs):
        module.CONFIGS.clear()
        fresh = module.reload_configs()
        assert "config_a" in fresh
        assert "config_a" in module.CONFIGS


# ── KNOWN_PROVIDERS and KNOWN_ENHANCE_STYLES ──────────────────────────────────

def test_known_providers_list():
    from src.hybrid_llm import KNOWN_PROVIDERS
    assert "anthropic" in KNOWN_PROVIDERS
    assert "ollama" in KNOWN_PROVIDERS
    assert "openai" in KNOWN_PROVIDERS


def test_known_enhance_styles():
    from src.hybrid_llm import KNOWN_ENHANCE_STYLES
    assert "financial_analysis" in KNOWN_ENHANCE_STYLES
    assert "execution_params_only" in KNOWN_ENHANCE_STYLES
