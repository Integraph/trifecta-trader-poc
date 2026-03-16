"""
Sanity Check Endpoint — /config/hybrid-configs/{name}/sanity-check

POST /config/hybrid-configs/{name}/sanity-check

Lightweight HTTP-level connectivity test for each provider/model in a config.
Does NOT import langchain, TradingAgents, or any ML library — pure HTTP only.
All 3 slots run concurrently via ThreadPoolExecutor.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException

from src.hybrid_llm import CONFIGS

logger = logging.getLogger(__name__)

sanity_check_router = APIRouter()

_TIMEOUT = 10  # seconds per slot


# ── Per-provider check functions ───────────────────────────────────────────────

def _check_ollama(model: str) -> dict:
    url = "http://localhost:11434/api/generate"
    try:
        t0 = time.time()
        resp = requests.post(
            url,
            json={"model": model, "prompt": "Respond with only the word OK.", "stream": False},
            timeout=_TIMEOUT,
        )
        latency = round((time.time() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("response"):
                return {"status": "pass", "latency_ms": latency, "error": None}
            return {"status": "fail", "latency_ms": latency, "error": "Empty response from Ollama"}
        return {"status": "fail", "latency_ms": latency, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.ConnectionError:
        return {"status": "fail", "latency_ms": None, "error": "Connection refused: Ollama not running"}
    except requests.exceptions.Timeout:
        return {"status": "fail", "latency_ms": None, "error": f"Timeout after {_TIMEOUT}s"}
    except Exception as e:
        return {"status": "fail", "latency_ms": None, "error": str(e)}


def _check_anthropic(model: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"status": "fail", "latency_ms": None, "error": "API key not configured (missing ANTHROPIC_API_KEY)"}
    try:
        t0 = time.time()
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Say OK"}],
            },
            timeout=_TIMEOUT,
        )
        latency = round((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "pass", "latency_ms": latency, "error": None}
        return {"status": "fail", "latency_ms": latency, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.Timeout:
        return {"status": "fail", "latency_ms": None, "error": f"Timeout after {_TIMEOUT}s"}
    except Exception as e:
        return {"status": "fail", "latency_ms": None, "error": str(e)}


def _check_openai_compat(model: str, base_url: str, api_key_env: str) -> dict:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {"status": "fail", "latency_ms": None, "error": f"API key not configured (missing {api_key_env})"}
    try:
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Say OK"}],
            },
            timeout=_TIMEOUT,
        )
        latency = round((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "pass", "latency_ms": latency, "error": None}
        return {"status": "fail", "latency_ms": latency, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.Timeout:
        return {"status": "fail", "latency_ms": None, "error": f"Timeout after {_TIMEOUT}s"}
    except Exception as e:
        return {"status": "fail", "latency_ms": None, "error": str(e)}


def _check_google(model: str) -> dict:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return {"status": "skip", "latency_ms": None, "error": "google SDK not installed (pip install google-generativeai)"}

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"status": "fail", "latency_ms": None, "error": "API key not configured (missing GOOGLE_API_KEY)"}
    try:
        genai.configure(api_key=api_key)
        t0 = time.time()
        gmodel = genai.GenerativeModel(model)
        response = gmodel.generate_content("Say OK", generation_config={"max_output_tokens": 5})
        latency = round((time.time() - t0) * 1000)
        if response.text:
            return {"status": "pass", "latency_ms": latency, "error": None}
        return {"status": "fail", "latency_ms": latency, "error": "Empty response from Google"}
    except Exception as e:
        return {"status": "fail", "latency_ms": None, "error": str(e)}


_PROVIDER_DISPATCH = {
    "ollama":      lambda model: _check_ollama(model),
    "anthropic":   lambda model: _check_anthropic(model),
    "openai":      lambda model: _check_openai_compat(model, "https://api.openai.com/v1",         "OPENAI_API_KEY"),
    "openrouter":  lambda model: _check_openai_compat(model, "https://openrouter.ai/api/v1",      "OPENROUTER_API_KEY"),
    "xai":         lambda model: _check_openai_compat(model, "https://api.x.ai/v1",               "XAI_API_KEY"),
    "google":      lambda model: _check_google(model),
}


def _check_slot(provider: str, model: str) -> dict:
    """Run the connectivity check for a single provider/model slot."""
    fn = _PROVIDER_DISPATCH.get(provider)
    if fn is None:
        return {"status": "skip", "latency_ms": None, "error": f"Unknown provider '{provider}'"}
    result = fn(model)
    result.update({"provider": provider, "model": model})
    return result


# ── Endpoint ───────────────────────────────────────────────────────────────────

@sanity_check_router.post("/{name}/sanity-check")
async def run_sanity_check(name: str):
    """Test connectivity for all 3 LLM slots in a config (concurrent, lightweight HTTP)."""
    if name not in CONFIGS:
        raise HTTPException(status_code=404, detail={"error": f"Config '{name}' not found"})

    cfg = CONFIGS[name]
    slots = {
        "tool_calling":     (cfg.tool_provider,            cfg.tool_model),
        "reasoning_quick":  (cfg.reasoning_quick_provider, cfg.reasoning_quick_model),
        "reasoning_deep":   (cfg.reasoning_deep_provider,  cfg.reasoning_deep_model),
    }

    results: dict = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_check_slot, provider, model): slot_name
            for slot_name, (provider, model) in slots.items()
        }
        for future in as_completed(futures):
            slot_name = futures[future]
            try:
                results[slot_name] = future.result()
            except Exception as e:
                provider, model = slots[slot_name]
                results[slot_name] = {
                    "provider": provider, "model": model,
                    "status": "fail", "latency_ms": None, "error": str(e),
                }

    statuses = [r["status"] for r in results.values()]
    if all(s == "pass" for s in statuses):
        overall = "pass"
    elif any(s == "pass" for s in statuses):
        overall = "partial"
    else:
        overall = "fail"

    return {
        "config_name": name,
        "overall":     overall,
        "checks":      results,
    }
