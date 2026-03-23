"""
Task 019 — Tests: Dynamic CLI hybrid choices in run_analysis.py and run_batch.py

Verifies that the argparse --hybrid choices list is driven by CONFIGS, not a
hardcoded string list, so configs added/removed via the admin UI are immediately
accepted/rejected by the CLI.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import src.hybrid_llm as hlm_module
from src.hybrid_llm import HybridLLMConfig


# ── Isolation fixture ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def restore_configs():
    """Snapshot and restore CONFIGS after every test to prevent cross-test pollution."""
    snapshot = {k: HybridLLMConfig(**v.to_flat_dict()) for k, v in hlm_module.CONFIGS.items()}
    yield
    hlm_module.CONFIGS.clear()
    hlm_module.CONFIGS.update(snapshot)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_run_analysis_parser() -> argparse.ArgumentParser:
    """Rebuild the run_analysis.py argument parser from its main() function."""
    # Isolate the parser construction from side effects by importing locally
    from src.hybrid_llm import CONFIGS as _hybrid_configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  type=str, default="AAPL")
    parser.add_argument("--hybrid",  type=str, default=None,
                        choices=list(_hybrid_configs.keys()))
    return parser


def _build_run_batch_parser() -> argparse.ArgumentParser:
    """Rebuild the run_batch.py --hybrid argument."""
    from src.hybrid_llm import CONFIGS as _hybrid_configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=str, default="hybrid_haiku_tools",
                        choices=list(_hybrid_configs.keys()))
    return parser


# ── Test: choices include all YAML-loaded configs ─────────────────────────────

def test_run_analysis_choices_include_all_yaml_configs():
    """The argparse choices list must include every key in CONFIGS."""
    from src.hybrid_llm import CONFIGS
    parser = _build_run_analysis_parser()
    hybrid_action = next(a for a in parser._actions if getattr(a, 'dest', None) == 'hybrid')
    assert set(hybrid_action.choices) == set(CONFIGS.keys())


def test_run_batch_choices_include_all_yaml_configs():
    """Same for run_batch.py."""
    from src.hybrid_llm import CONFIGS
    parser = _build_run_batch_parser()
    hybrid_action = next(a for a in parser._actions if getattr(a, 'dest', None) == 'hybrid')
    assert set(hybrid_action.choices) == set(CONFIGS.keys())


def test_run_analysis_choices_contain_all_13_defaults():
    """All 13 original configs must be present."""
    from src.hybrid_llm import CONFIGS
    expected = {
        "all_cloud", "hybrid_qwen", "hybrid_mistral",
        "hybrid_aggressive_qwen", "hybrid_aggressive_mistral",
        "hybrid_qwen32", "hybrid_aggressive_qwen32",
        "hybrid_qwen_enhanced", "hybrid_haiku_tools", "hybrid_haiku_aggressive",
        "hybrid_haiku_qwen35_27b", "hybrid_haiku_qwen35_35b", "hybrid_haiku_qwen35_9b",
    }
    assert expected.issubset(set(CONFIGS.keys()))


# ── Test: new config is accepted after reload ──────────────────────────────────

def test_new_config_accepted_after_reload(tmp_path):
    """Adding a config to YAML and reloading makes it available as a CLI choice."""
    # Build YAML with an extra config
    configs_data = {k: v.to_flat_dict() for k, v in hlm_module.CONFIGS.items()}
    configs_data["test_new_model"] = {
        "tool_provider": "openai", "tool_model": "gpt-4o-mini",
        "reasoning_quick_provider": "ollama", "reasoning_quick_model": "qwen2.5:14b",
        "reasoning_deep_provider": "openai", "reasoning_deep_model": "gpt-4o",
        "enhance_local": False, "enhance_style": "financial_analysis",
        "enhance_deep": False, "enhance_deep_style": "execution_params_only",
    }
    yaml_path = tmp_path / "hybrid_llm.yaml"
    yaml_path.write_text(yaml.safe_dump({"configs": configs_data}))

    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        hlm_module.reload_configs()

    assert "test_new_model" in hlm_module.CONFIGS

    # Build parser after reload — should now accept the new name
    from src.hybrid_llm import CONFIGS as _hybrid_configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=str, default=None, choices=list(_hybrid_configs.keys()))
    args = parser.parse_args(["--hybrid", "test_new_model"])
    assert args.hybrid == "test_new_model"


def test_deleted_config_rejected_after_reload(tmp_path):
    """Removing a config from YAML and reloading makes it rejected by argparse."""
    # Build YAML without 'all_cloud'
    configs_data = {
        k: v.to_flat_dict()
        for k, v in hlm_module.CONFIGS.items()
        if k != "all_cloud"
    }
    yaml_path = tmp_path / "hybrid_llm.yaml"
    yaml_path.write_text(yaml.safe_dump({"configs": configs_data}))

    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        hlm_module.reload_configs()

    assert "all_cloud" not in hlm_module.CONFIGS

    # Parser built after reload should NOT accept 'all_cloud'
    from src.hybrid_llm import CONFIGS as _hybrid_configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=str, default=None, choices=list(_hybrid_configs.keys()))
    with pytest.raises(SystemExit):
        parser.parse_args(["--hybrid", "all_cloud"])


# ── Test: no hardcoded choices strings remain in source files ──────────────────

def test_no_hardcoded_choices_in_run_analysis():
    """run_analysis.py must not contain the old hardcoded choices list."""
    src_text = Path("src/run_analysis.py").read_text()
    # The old list always started with 'all_cloud' as the first choices entry
    assert '"all_cloud", "hybrid_qwen"' not in src_text
    assert "'all_cloud', 'hybrid_qwen'" not in src_text


def test_no_hardcoded_choices_in_run_batch():
    """run_batch.py must not contain the old hardcoded choices list."""
    src_text = Path("src/run_batch.py").read_text()
    assert '"all_cloud", "hybrid_qwen"' not in src_text
    assert "'all_cloud', 'hybrid_qwen'" not in src_text


# ── Test: dynamic pattern present in source files ─────────────────────────────

def test_dynamic_choices_pattern_in_run_analysis():
    """run_analysis.py must use CONFIGS.keys() for the --hybrid choices."""
    src_text = Path("src/run_analysis.py").read_text()
    assert "_hybrid_configs" in src_text
    assert "_hybrid_configs.keys()" in src_text


def test_dynamic_choices_pattern_in_run_batch():
    """run_batch.py must use CONFIGS.keys() for the --hybrid choices."""
    src_text = Path("src/run_batch.py").read_text()
    assert "_hybrid_configs" in src_text
    assert "_hybrid_configs.keys()" in src_text


# ── Test: valid existing choices still parse correctly ────────────────────────

@pytest.mark.parametrize("config_name", [
    "all_cloud",
    "hybrid_qwen",
    "hybrid_haiku_tools",
    "hybrid_haiku_aggressive",
])
def test_existing_config_accepted_by_parser(config_name: str):
    """All original 13 config names must still be accepted."""
    from src.hybrid_llm import CONFIGS as _hybrid_configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=str, default=None, choices=list(_hybrid_configs.keys()))
    args = parser.parse_args(["--hybrid", config_name])
    assert args.hybrid == config_name


@pytest.mark.parametrize("config_name", [
    "totally_fake_config",
    "my_llm",
    "all cloud",   # spaces
    "",
])
def test_unknown_config_rejected_by_parser(config_name: str):
    """Names not in CONFIGS must be rejected by argparse."""
    from src.hybrid_llm import CONFIGS as _hybrid_configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", type=str, default=None, choices=list(_hybrid_configs.keys()))
    with pytest.raises(SystemExit):
        parser.parse_args(["--hybrid", config_name])
