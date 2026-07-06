"""
Hybrid LLM router for assigning different providers to different agents.

This module creates a patched version of TradingAgentsGraph that supports
per-agent LLM provider routing. Tool-calling agents use cloud LLMs (Anthropic),
while pure-reasoning agents can use local Ollama models.

Configurations are stored in config/hybrid_llm.yaml and loaded at import time.
The Python _DEFAULT_CONFIGS dict serves as a fallback when the YAML file is absent.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Canonical YAML file location (relative to this file's package root)
_YAML_PATH = Path(__file__).parent.parent / "config" / "hybrid_llm.yaml"

# Canonical provider and style enumerations
KNOWN_PROVIDERS = ["anthropic", "ollama", "openai", "google", "xai", "openrouter"]
KNOWN_ENHANCE_STYLES = ["financial_analysis", "structured", "few_shot", "execution_params_only"]


class HybridLLMConfig:
    """Configuration for hybrid LLM routing.

    Defines which provider/model to use for each agent category:
    - tool_calling: Agents that use bind_tools() -- must support tool calling
    - reasoning_quick: Pure reasoning agents on the quick path
    - reasoning_deep: Pure reasoning agents on the deep path (judges)
    - enhance_local: Whether to wrap local models with data-citation prefix
    - enhance_style: Which prefix style to use (see src/enhanced_llm.py)
    - temperature: Sampling temperature applied to ALL THREE slots when set
      (None = each provider's own default).
    - deep_temperature: Overrides temperature for the DEEP slot only.
    - local_seed: Fixed sampling seed applied to ollama-provider slots.
      TRI-69 checkpoint 1 finding: pinning temp=0 (greedy) wedges the local
      thinking model on content-dependent inputs (runaway thinking, 47-min
      requests); model-default sampling + a pinned seed gives byte-identical
      repeats WITHOUT the greedy pathology. Determinism is verified
      empirically by the battery, never assumed.
    - local_max_tokens: Generation cap for ollama-provider slots — a
      runaway fails in minutes (-> NO-DECISION) instead of wedging for hours.
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
        enhance_deep: bool = False,
        enhance_deep_style: str = "execution_params_only",
        temperature: Optional[float] = None,
        deep_temperature: Optional[float] = None,
        local_seed: Optional[int] = None,
        local_max_tokens: Optional[int] = None,
    ):
        self.tool_provider = tool_provider
        self.tool_model = tool_model
        self.reasoning_quick_provider = reasoning_quick_provider
        self.reasoning_quick_model = reasoning_quick_model
        self.reasoning_deep_provider = reasoning_deep_provider
        self.reasoning_deep_model = reasoning_deep_model
        self.enhance_local = enhance_local
        self.enhance_style = enhance_style
        self.enhance_deep = enhance_deep
        self.enhance_deep_style = enhance_deep_style
        self.temperature = temperature
        self.deep_temperature = deep_temperature
        self.local_seed = local_seed
        self.local_max_tokens = local_max_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Return a summary dict for logging/reporting."""
        d = {
            "tool_calling": f"{self.tool_provider}/{self.tool_model}",
            "reasoning_quick": f"{self.reasoning_quick_provider}/{self.reasoning_quick_model}",
            "reasoning_deep": f"{self.reasoning_deep_provider}/{self.reasoning_deep_model}",
        }
        if self.enhance_local:
            d["enhance_style"] = self.enhance_style
        if self.enhance_deep:
            d["enhance_deep_style"] = self.enhance_deep_style
        d.update(self._optional_sampling_fields())
        return d

    def _optional_sampling_fields(self) -> Dict[str, Any]:
        """Sampling knobs, emitted only when set — pre-existing configs
        round-trip byte-identically and older readers never see the keys."""
        d: Dict[str, Any] = {}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.deep_temperature is not None:
            d["deep_temperature"] = self.deep_temperature
        if self.local_seed is not None:
            d["local_seed"] = self.local_seed
        if self.local_max_tokens is not None:
            d["local_max_tokens"] = self.local_max_tokens
        return d

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return all fields as a flat dict (for YAML / API serialization)."""
        d = {
            "tool_provider":             self.tool_provider,
            "tool_model":                self.tool_model,
            "reasoning_quick_provider":  self.reasoning_quick_provider,
            "reasoning_quick_model":     self.reasoning_quick_model,
            "reasoning_deep_provider":   self.reasoning_deep_provider,
            "reasoning_deep_model":      self.reasoning_deep_model,
            "enhance_local":             self.enhance_local,
            "enhance_style":             self.enhance_style,
            "enhance_deep":              self.enhance_deep,
            "enhance_deep_style":        self.enhance_deep_style,
        }
        d.update(self._optional_sampling_fields())
        return d


# ── Python default configs (fallback when YAML is absent) ─────────────────────

_DEFAULT_CONFIGS: Dict[str, HybridLLMConfig] = {
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
        enhance_deep=True,
        enhance_deep_style="execution_params_only",
    ),

    "hybrid_haiku_tools": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-haiku-4-5-20251001",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=True,
        enhance_style="financial_analysis",
        enhance_deep=True,
        enhance_deep_style="execution_params_only",
    ),

    "hybrid_haiku_qwen35_27b": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-haiku-4-5-20251001",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen3.5:27b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=True,
        enhance_style="financial_analysis",
        enhance_deep=True,
        enhance_deep_style="execution_params_only",
    ),

    "hybrid_haiku_qwen35_35b": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-haiku-4-5-20251001",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen3.5:35b-a3b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=True,
        enhance_style="financial_analysis",
        enhance_deep=True,
        enhance_deep_style="execution_params_only",
    ),

    "hybrid_haiku_qwen35_9b": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-haiku-4-5-20251001",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen3.5:9b",
        reasoning_deep_provider="anthropic",
        reasoning_deep_model="claude-sonnet-4-5-20250929",
        enhance_local=True,
        enhance_style="financial_analysis",
        enhance_deep=True,
        enhance_deep_style="execution_params_only",
    ),

    "hybrid_haiku_aggressive": HybridLLMConfig(
        tool_provider="anthropic",
        tool_model="claude-haiku-4-5-20251001",
        reasoning_quick_provider="ollama",
        reasoning_quick_model="qwen2.5:14b",
        reasoning_deep_provider="ollama",
        reasoning_deep_model="qwen2.5:14b",
        enhance_local=True,
        enhance_style="financial_analysis",
        enhance_deep=True,
        enhance_deep_style="execution_params_only",
    ),
}


# ── YAML helpers ───────────────────────────────────────────────────────────────

def _read_yaml() -> dict:
    """Read the YAML file; return empty dict if absent or invalid."""
    if not _YAML_PATH.exists():
        return {}
    try:
        with open(_YAML_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to read %s: %s", _YAML_PATH, e)
        return {}


def _write_yaml(data: dict) -> None:
    """Atomically write the full YAML file (temp + rename)."""
    _YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=_YAML_PATH.parent, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("# Hybrid LLM Configurations — managed by Admin UI\n")
            f.write("# Edit via localhost:8420/config or directly here.\n\n")
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, _YAML_PATH)
        logger.debug("Wrote hybrid LLM configs to %s", _YAML_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _yaml_entry_to_config(entry: dict) -> HybridLLMConfig:
    """Convert a YAML dict entry to a HybridLLMConfig instance."""
    return HybridLLMConfig(
        tool_provider            = entry.get("tool_provider",            "anthropic"),
        tool_model               = entry.get("tool_model",               "claude-sonnet-4-5-20250929"),
        reasoning_quick_provider = entry.get("reasoning_quick_provider", "ollama"),
        reasoning_quick_model    = entry.get("reasoning_quick_model",    "qwen2.5:14b"),
        reasoning_deep_provider  = entry.get("reasoning_deep_provider",  "anthropic"),
        reasoning_deep_model     = entry.get("reasoning_deep_model",     "claude-sonnet-4-5-20250929"),
        enhance_local            = bool(entry.get("enhance_local",       False)),
        enhance_style            = entry.get("enhance_style",            "financial_analysis"),
        enhance_deep             = bool(entry.get("enhance_deep",        False)),
        enhance_deep_style       = entry.get("enhance_deep_style",       "execution_params_only"),
        temperature              = (float(entry["temperature"])
                                    if entry.get("temperature") is not None else None),
        deep_temperature         = (float(entry["deep_temperature"])
                                    if entry.get("deep_temperature") is not None else None),
        local_seed               = (int(entry["local_seed"])
                                    if entry.get("local_seed") is not None else None),
        local_max_tokens         = (int(entry["local_max_tokens"])
                                    if entry.get("local_max_tokens") is not None else None),
    )


def _generate_yaml_from_defaults() -> None:
    """Write the default Python configs to YAML (first-run bootstrap)."""
    data = {
        "configs": {
            name: cfg.to_flat_dict()
            for name, cfg in _DEFAULT_CONFIGS.items()
        }
    }
    _write_yaml(data)
    logger.info("Generated %s from Python defaults (%d configs)", _YAML_PATH, len(_DEFAULT_CONFIGS))


# ── Public API ─────────────────────────────────────────────────────────────────

def load_configs() -> Dict[str, HybridLLMConfig]:
    """Load hybrid configs from YAML, falling back to Python defaults.

    On first call, if config/hybrid_llm.yaml doesn't exist, generates it
    from the Python _DEFAULT_CONFIGS dict and writes to disk.
    """
    if not _YAML_PATH.exists():
        _generate_yaml_from_defaults()
        return dict(_DEFAULT_CONFIGS)

    raw = _read_yaml()
    configs_raw = raw.get("configs", {})
    if not configs_raw:
        logger.warning("hybrid_llm.yaml has no 'configs' section — using defaults")
        return dict(_DEFAULT_CONFIGS)

    result: Dict[str, HybridLLMConfig] = {}
    for name, entry in configs_raw.items():
        try:
            result[name] = _yaml_entry_to_config(entry)
        except Exception as e:
            logger.error("Skipping invalid config '%s': %s", name, e)

    logger.debug("Loaded %d hybrid LLM configs from %s", len(result), _YAML_PATH)
    return result


def save_config(name: str, config: HybridLLMConfig) -> None:
    """Save or update a single config entry. Writes the full YAML file."""
    raw = _read_yaml()
    if "configs" not in raw:
        raw["configs"] = {}
    raw["configs"][name] = config.to_flat_dict()
    _write_yaml(raw)
    CONFIGS[name] = config
    logger.info("Saved hybrid LLM config '%s'", name)


def delete_config(name: str) -> None:
    """Remove a config entry. Raises KeyError if not found."""
    raw = _read_yaml()
    configs_raw = raw.get("configs", {})
    if name not in configs_raw:
        raise KeyError(f"Config '{name}' not found")
    del configs_raw[name]
    raw["configs"] = configs_raw
    _write_yaml(raw)
    CONFIGS.pop(name, None)
    logger.info("Deleted hybrid LLM config '%s'", name)


def reload_configs() -> Dict[str, HybridLLMConfig]:
    """Re-read YAML from disk and update in-memory CONFIGS. Use after external edits."""
    fresh = load_configs()
    CONFIGS.clear()
    CONFIGS.update(fresh)
    logger.info("Reloaded %d hybrid LLM configs", len(CONFIGS))
    return fresh


# ── Module-level CONFIGS dict (backward-compatible) ───────────────────────────
# All existing callers: `from src.hybrid_llm import CONFIGS` still work.
# `CONFIGS[hybrid]` in run_analysis.py continues to resolve correctly.

CONFIGS: Dict[str, HybridLLMConfig] = load_configs()


# ── LLM factory ───────────────────────────────────────────────────────────────

def create_hybrid_llms(hybrid_config: HybridLLMConfig) -> Dict[str, Any]:
    """Create three LLM instances for hybrid routing.

    Returns:
        Dict with keys: 'tool_calling_llm', 'reasoning_quick_llm', 'reasoning_deep_llm'
    """
    from tradingagents.llm_clients.factory import create_llm_client

    llms = {}

    # Slot-shared kwargs. temperature (when set) is threaded into ALL THREE
    # create_llm_client calls; both the Anthropic and OpenAI-compatible/Ollama
    # clients forward it to the underlying Chat model via _PASSTHROUGH_KWARGS.
    slot_kwargs = {}
    if hybrid_config.temperature is not None:
        slot_kwargs["temperature"] = hybrid_config.temperature

    def _apply_local_pins(llm, provider: str):
        """Reproducibility pins for ollama slots (TRI-69 checkpoint 1):
        a fixed seed makes model-default sampling repeat-identical without
        the greedy (temp=0) runaway-thinking pathology; max_tokens turns a
        runaway into a fast, loud failure. Set post-construction because the
        vendor client's passthrough list doesn't carry these keys (zero-mod)."""
        if provider != "ollama":
            return llm
        if hybrid_config.local_seed is not None:
            llm.seed = hybrid_config.local_seed
        if hybrid_config.local_max_tokens is not None:
            llm.max_tokens = hybrid_config.local_max_tokens
        return llm

    tool_client = create_llm_client(
        provider=hybrid_config.tool_provider,
        model=hybrid_config.tool_model,
        **slot_kwargs,
    )
    llms["tool_calling_llm"] = _apply_local_pins(
        tool_client.get_llm(), hybrid_config.tool_provider)

    reasoning_quick_client = create_llm_client(
        provider=hybrid_config.reasoning_quick_provider,
        model=hybrid_config.reasoning_quick_model,
        **slot_kwargs,
    )
    reasoning_quick_llm = _apply_local_pins(
        reasoning_quick_client.get_llm(), hybrid_config.reasoning_quick_provider)
    if hybrid_config.enhance_local and hybrid_config.reasoning_quick_provider == "ollama":
        from src.enhanced_llm import create_enhanced_llm
        reasoning_quick_llm = create_enhanced_llm(reasoning_quick_llm, style=hybrid_config.enhance_style)
        logger.info("Enhanced reasoning_quick_llm with style '%s'", hybrid_config.enhance_style)
    llms["reasoning_quick_llm"] = reasoning_quick_llm

    deep_kwargs = dict(slot_kwargs)
    if hybrid_config.deep_temperature is not None:
        deep_kwargs["temperature"] = hybrid_config.deep_temperature
    reasoning_deep_client = create_llm_client(
        provider=hybrid_config.reasoning_deep_provider,
        model=hybrid_config.reasoning_deep_model,
        **deep_kwargs,
    )
    reasoning_deep_llm = _apply_local_pins(
        reasoning_deep_client.get_llm(), hybrid_config.reasoning_deep_provider)
    if hybrid_config.enhance_local and hybrid_config.reasoning_deep_provider == "ollama":
        from src.enhanced_llm import create_enhanced_llm
        reasoning_deep_llm = create_enhanced_llm(reasoning_deep_llm, style=hybrid_config.enhance_style)
        logger.info("Enhanced reasoning_deep_llm with style '%s'", hybrid_config.enhance_style)
    elif hybrid_config.enhance_deep:
        from src.enhanced_llm import create_enhanced_llm
        reasoning_deep_llm = create_enhanced_llm(reasoning_deep_llm, style=hybrid_config.enhance_deep_style)
        logger.info("Enhanced reasoning_deep_llm with style '%s'", hybrid_config.enhance_deep_style)
    llms["reasoning_deep_llm"] = reasoning_deep_llm

    logger.info("Hybrid LLM routing: %s", hybrid_config.to_dict())
    return llms
