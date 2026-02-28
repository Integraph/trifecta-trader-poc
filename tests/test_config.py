"""Basic tests to verify project setup and configuration."""

import sys
from pathlib import Path

# Add vendor to path
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "TradingAgents"))


def test_tradingagents_importable():
    """Verify TradingAgents can be imported from vendor."""
    from tradingagents.default_config import DEFAULT_CONFIG
    assert "llm_provider" in DEFAULT_CONFIG
    assert "data_vendors" in DEFAULT_CONFIG


def test_config_creation():
    """Verify our config builder works."""
    from src.run_analysis import get_config

    # Test Anthropic config
    config = get_config("anthropic")
    assert config["llm_provider"] == "anthropic"
    assert "claude" in config["deep_think_llm"]

    # Test Ollama config
    config = get_config("ollama")
    assert config["llm_provider"] == "ollama"
    assert config["deep_think_llm"] == "llama3.1:8b"


def test_data_vendors_default_to_yfinance():
    """Verify all data vendors default to yfinance."""
    from src.run_analysis import get_config
    config = get_config("anthropic")
    for vendor_key, vendor_value in config["data_vendors"].items():
        assert vendor_value == "yfinance", f"{vendor_key} should default to yfinance"


def test_results_directory():
    """Verify results directory path is set correctly."""
    from src.run_analysis import get_config
    config = get_config("anthropic")
    assert "results" in config["results_dir"]
