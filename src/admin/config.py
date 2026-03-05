"""
Configuration Endpoints — /config/*

GET  /config/automation             Current automation.yaml as JSON
PUT  /config/automation             Write partial/full config update to disk
GET  /config/supabase               Current supabase.yaml as JSON
PUT  /config/supabase               Write supabase config update to disk
GET  /config/watchlists             List available watchlist files + tickers
PUT  /config/watchlists/{name}      Create / update a watchlist file
GET  /config/hybrid-configs         List all hybrid LLM configs from CONFIGS
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.admin.dependencies import get_daemon

logger = logging.getLogger(__name__)

config_router = APIRouter()

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

@config_router.get("/hybrid-configs")
async def get_hybrid_configs():
    """List all available hybrid LLM configurations."""
    try:
        from src.hybrid_llm import CONFIGS
        daemon = get_daemon()
        active = None
        if daemon is not None:
            active = daemon._cfg.get("scheduler", {}).get("hybrid_config")

        configs = []
        for name, cfg in CONFIGS.items():
            # HybridLLMConfig is a class with a to_dict() method
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else (cfg if isinstance(cfg, dict) else {})
            configs.append({
                "name":                     name,
                "tool_provider":            getattr(cfg, "tool_provider", cfg_dict.get("tool_provider")),
                "tool_model":               getattr(cfg, "tool_model", cfg_dict.get("tool_model")),
                "reasoning_quick_provider": getattr(cfg, "reasoning_quick_provider", None),
                "reasoning_quick_model":    getattr(cfg, "reasoning_quick_model", None),
                "reasoning_deep_provider":  getattr(cfg, "reasoning_deep_provider", None),
                "reasoning_deep_model":     getattr(cfg, "reasoning_deep_model", None),
            })

        return {"configs": configs, "active": active}
    except Exception as e:
        logger.error("Failed to load hybrid configs: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})
