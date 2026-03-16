"""
Configuration Endpoints — /config/*

GET    /config/automation                  Current automation.yaml as JSON
PUT    /config/automation                  Write partial/full config update to disk
GET    /config/supabase                    Current supabase.yaml as JSON
PUT    /config/supabase                    Write supabase config update to disk
GET    /config/watchlists                  List available watchlist files + tickers
PUT    /config/watchlists/{name}           Create / update a watchlist file
GET    /config/hybrid-configs              List all hybrid LLM configs (YAML-backed)
POST   /config/hybrid-configs              Create a new hybrid LLM config
PUT    /config/hybrid-configs/{name}       Update an existing config
DELETE /config/hybrid-configs/{name}       Delete a config
POST   /config/hybrid-configs/{name}/clone Clone a config with a new name
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.admin.dependencies import get_daemon
from src.hybrid_llm import (
    KNOWN_PROVIDERS,
    KNOWN_ENHANCE_STYLES,
    HybridLLMConfig,
    load_configs,
    save_config,
    delete_config,
    CONFIGS,
)

logger = logging.getLogger(__name__)

config_router = APIRouter()

_NAME_RE = re.compile(r"^[a-zA-Z0-9_]{1,64}$")

# Fields that take effect immediately vs. require daemon restart
_IMMEDIATE_FIELDS = {
    "queue_reader.poll_interval_seconds",
    "queue_reader.max_retries",
    "queue_reader.cooldown_seconds",
    "accuracy.backfill_on_first_run",
    "supabase.write_enabled",
    "supabase.signal_ttl_hours",
}

_RESTART_FIELDS = {
    "scheduler.watchlist_hour",
    "scheduler.watchlist_minute",
    "scheduler.timezone",
    "scheduler.hybrid_config",
    "queue_reader.queue_dir",
    "queue_reader.target_trader",
    "accuracy.update_hour",
    "accuracy.update_minute",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _flatten_keys(d: dict, prefix: str = "") -> list:
    """Return dot-separated keys for a nested dict."""
    keys = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(_flatten_keys(v, full))
        else:
            keys.append(full)
    return keys


def _classify_changes(new_keys: list) -> tuple:
    applied       = [k for k in new_keys if k in _IMMEDIATE_FIELDS]
    needs_restart = [k for k in new_keys if k in _RESTART_FIELDS]
    return applied, needs_restart


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


# ── Request models ────────────────────────────────────────────────────────────

class WatchlistUpdate(BaseModel):
    tickers: list


# ── Automation config ─────────────────────────────────────────────────────────

@config_router.get("/automation")
async def get_automation_config():
    """Return current automation.yaml merged with defaults."""
    daemon = get_daemon()
    if daemon is not None:
        return daemon._cfg

    # Standalone: load directly
    from src.automation.daemon import _CONFIG_DEFAULTS, _deep_merge as daemon_merge
    path    = Path("config/automation.yaml")
    loaded  = _load_yaml(path)
    merged  = daemon_merge(_CONFIG_DEFAULTS, loaded)
    return merged


@config_router.put("/automation")
async def update_automation_config(body: Dict[str, Any]):
    """Write updated automation config to disk (deep merge with existing).

    Returns which changes take effect immediately vs. require daemon restart.
    """
    path      = Path("config/automation.yaml")
    existing  = _load_yaml(path)
    merged    = _deep_merge(existing, body)
    _write_yaml(path, merged)

    # Update daemon in-memory config if running
    daemon = get_daemon()
    if daemon is not None:
        from src.automation.daemon import _deep_merge as dm
        daemon._cfg = dm(daemon._cfg, body)

    changed_keys  = _flatten_keys(body)
    applied, needs_restart = _classify_changes(changed_keys)

    return {
        "applied":          {k: True for k in applied},
        "requires_restart": needs_restart,
        "written_to":       str(path),
    }


# ── Supabase config ───────────────────────────────────────────────────────────

@config_router.get("/supabase")
async def get_supabase_config():
    """Return current supabase.yaml as JSON."""
    path = Path("config/supabase.yaml")
    return _load_yaml(path)


@config_router.put("/supabase")
async def update_supabase_config(body: Dict[str, Any]):
    """Write updated Supabase config to disk."""
    path     = Path("config/supabase.yaml")
    existing = _load_yaml(path)
    merged   = _deep_merge(existing, body)
    _write_yaml(path, merged)

    changed_keys          = _flatten_keys(body)
    applied, needs_restart = _classify_changes(changed_keys)

    return {
        "applied":          {k: True for k in applied},
        "requires_restart": needs_restart,
        "written_to":       str(path),
    }


# ── Watchlists ────────────────────────────────────────────────────────────────

@config_router.get("/watchlists")
async def get_watchlists():
    """List available watchlist files and their tickers."""
    wl_dir = Path("config/watchlists")
    if not wl_dir.exists():
        return {"watchlists": []}

    watchlists = []
    for path in sorted(wl_dir.glob("*.yaml")):
        try:
            data    = _load_yaml(path)
            tickers = data.get("tickers", []) if isinstance(data, dict) else (data or [])
            if isinstance(tickers, list):
                tickers = [t.upper() for t in tickers]
        except Exception:
            tickers = []
        watchlists.append({
            "name":    path.stem,
            "path":    str(path),
            "tickers": tickers,
        })

    return {"watchlists": watchlists}


@config_router.put("/watchlists/{name}")
async def update_watchlist(name: str, body: WatchlistUpdate):
    """Create or update a watchlist file."""
    wl_dir = Path("config/watchlists")
    wl_dir.mkdir(parents=True, exist_ok=True)
    path   = wl_dir / f"{name}.yaml"
    tickers = [t.upper().strip() for t in body.tickers if t.strip()]
    _write_yaml(path, {"tickers": tickers})
    return {"name": name, "path": str(path), "tickers": tickers}


# ── Hybrid configs ────────────────────────────────────────────────────────────

class HybridConfigBody(BaseModel):
    tool_provider:            str = "anthropic"
    tool_model:               str = "claude-haiku-4-5-20251001"
    reasoning_quick_provider: str = "ollama"
    reasoning_quick_model:    str = "qwen2.5:14b"
    reasoning_deep_provider:  str = "anthropic"
    reasoning_deep_model:     str = "claude-sonnet-4-5-20250929"
    enhance_local:            bool = False
    enhance_style:            str = "financial_analysis"
    enhance_deep:             bool = False
    enhance_deep_style:       str = "execution_params_only"

class CreateHybridConfigBody(HybridConfigBody):
    name: str

class CloneBody(BaseModel):
    new_name: str


def _validate_config_body(body: HybridConfigBody) -> None:
    """Raise HTTPException if provider or style values are invalid."""
    for field, val in [
        ("tool_provider",            body.tool_provider),
        ("reasoning_quick_provider", body.reasoning_quick_provider),
        ("reasoning_deep_provider",  body.reasoning_deep_provider),
    ]:
        if val not in KNOWN_PROVIDERS:
            raise HTTPException(
                status_code=422,
                detail={"error": f"Invalid {field}: '{val}'. Must be one of {KNOWN_PROVIDERS}"},
            )
    for field, val in [
        ("enhance_style",      body.enhance_style),
        ("enhance_deep_style", body.enhance_deep_style),
    ]:
        if val not in KNOWN_ENHANCE_STYLES:
            raise HTTPException(
                status_code=422,
                detail={"error": f"Invalid {field}: '{val}'. Must be one of {KNOWN_ENHANCE_STYLES}"},
            )


def _active_hybrid_config() -> Optional[str]:
    """Return the currently active hybrid_config from automation.yaml or daemon."""
    daemon = get_daemon()
    if daemon is not None:
        return daemon._cfg.get("scheduler", {}).get("hybrid_config")
    path = Path("config/automation.yaml")
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("scheduler", {}).get("hybrid_config")
    return None


def _cfg_to_response(name: str, cfg: HybridLLMConfig) -> dict:
    d = cfg.to_flat_dict()
    d["name"] = name
    return d


@config_router.get("/hybrid-configs")
async def get_hybrid_configs():
    """List all hybrid LLM configurations from YAML. Includes providers + styles for UI dropdowns."""
    try:
        configs_dict = load_configs()
        active = _active_hybrid_config()
        return {
            "configs":        [_cfg_to_response(n, c) for n, c in configs_dict.items()],
            "active":         active,
            "providers":      KNOWN_PROVIDERS,
            "enhance_styles": KNOWN_ENHANCE_STYLES,
        }
    except Exception as e:
        logger.error("Failed to load hybrid configs: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@config_router.post("/hybrid-configs", status_code=201)
async def create_hybrid_config(body: CreateHybridConfigBody):
    """Create a new hybrid LLM config."""
    name = body.name.strip()
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail={"error": "name must be 1-64 chars, alphanumeric + underscores only"},
        )
    if name in CONFIGS:
        raise HTTPException(status_code=409, detail={"error": f"Config '{name}' already exists"})
    _validate_config_body(body)

    cfg = HybridLLMConfig(
        tool_provider            = body.tool_provider,
        tool_model               = body.tool_model,
        reasoning_quick_provider = body.reasoning_quick_provider,
        reasoning_quick_model    = body.reasoning_quick_model,
        reasoning_deep_provider  = body.reasoning_deep_provider,
        reasoning_deep_model     = body.reasoning_deep_model,
        enhance_local            = body.enhance_local,
        enhance_style            = body.enhance_style,
        enhance_deep             = body.enhance_deep,
        enhance_deep_style       = body.enhance_deep_style,
    )
    save_config(name, cfg)
    return _cfg_to_response(name, cfg)


@config_router.put("/hybrid-configs/{name}")
async def update_hybrid_config(name: str, body: HybridConfigBody):
    """Update an existing hybrid LLM config."""
    if name not in CONFIGS:
        raise HTTPException(status_code=404, detail={"error": f"Config '{name}' not found"})
    _validate_config_body(body)

    cfg = HybridLLMConfig(
        tool_provider            = body.tool_provider,
        tool_model               = body.tool_model,
        reasoning_quick_provider = body.reasoning_quick_provider,
        reasoning_quick_model    = body.reasoning_quick_model,
        reasoning_deep_provider  = body.reasoning_deep_provider,
        reasoning_deep_model     = body.reasoning_deep_model,
        enhance_local            = body.enhance_local,
        enhance_style            = body.enhance_style,
        enhance_deep             = body.enhance_deep,
        enhance_deep_style       = body.enhance_deep_style,
    )
    save_config(name, cfg)
    return _cfg_to_response(name, cfg)


@config_router.delete("/hybrid-configs/{name}", status_code=204)
async def delete_hybrid_config(name: str):
    """Delete a hybrid LLM config. Returns 409 if config is currently active."""
    if name not in CONFIGS:
        raise HTTPException(status_code=404, detail={"error": f"Config '{name}' not found"})
    active = _active_hybrid_config()
    if active and active == name:
        raise HTTPException(
            status_code=409,
            detail={"error": f"Cannot delete active config '{name}'. Change active config first."},
        )
    try:
        delete_config(name)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": f"Config '{name}' not found"})


@config_router.post("/hybrid-configs/{name}/clone", status_code=201)
async def clone_hybrid_config(name: str, body: CloneBody):
    """Clone an existing config to a new name."""
    if name not in CONFIGS:
        raise HTTPException(status_code=404, detail={"error": f"Config '{name}' not found"})
    new_name = body.new_name.strip()
    if not _NAME_RE.match(new_name):
        raise HTTPException(
            status_code=422,
            detail={"error": "new_name must be 1-64 chars, alphanumeric + underscores only"},
        )
    if new_name in CONFIGS:
        raise HTTPException(status_code=409, detail={"error": f"Config '{new_name}' already exists"})

    src_cfg = CONFIGS[name]
    new_cfg = HybridLLMConfig(**src_cfg.to_flat_dict())
    save_config(new_name, new_cfg)
    return _cfg_to_response(new_name, new_cfg)
