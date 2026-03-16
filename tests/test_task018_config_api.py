"""
Task 018 — Tests: Admin API endpoints for hybrid LLM configs
POST /config/hybrid-configs
PUT  /config/hybrid-configs/{name}
DELETE /config/hybrid-configs/{name}
POST /config/hybrid-configs/{name}/clone
GET  /config/hybrid-configs (enhanced)
"""
import pytest
import yaml
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

import src.hybrid_llm as hlm_module
from src.hybrid_llm import HybridLLMConfig


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_configs():
    """Snapshot and restore CONFIGS around each test."""
    original = {k: HybridLLMConfig(**v.to_flat_dict()) for k, v in hlm_module.CONFIGS.items()}
    yield
    hlm_module.CONFIGS.clear()
    hlm_module.CONFIGS.update(original)


@pytest.fixture
def app():
    from src.admin.app import create_app
    return create_app(daemon=None, db=None)


# ── Helper ─────────────────────────────────────────────────────────────────────

def make_client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


VALID_PAYLOAD = {
    "name": "test_new_config",
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
}


# ── GET /config/hybrid-configs ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_hybrid_configs_returns_providers(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.get("/config/hybrid-configs")
    assert r.status_code == 200
    data = r.json()
    assert "configs" in data
    assert "providers" in data
    assert "enhance_styles" in data
    assert "anthropic" in data["providers"]
    assert "financial_analysis" in data["enhance_styles"]


@pytest.mark.asyncio
async def test_get_hybrid_configs_returns_full_fields(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.get("/config/hybrid-configs")
    assert r.status_code == 200
    config = r.json()["configs"][0]
    required = {"name", "tool_provider", "tool_model", "reasoning_quick_provider",
                "reasoning_quick_model", "reasoning_deep_provider", "reasoning_deep_model",
                "enhance_local", "enhance_style", "enhance_deep", "enhance_deep_style"}
    assert required.issubset(set(config.keys()))


# ── POST /config/hybrid-configs ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_config_success(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs", json=VALID_PAYLOAD)
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "test_new_config"
    assert data["tool_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_create_config_duplicate_name(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs", json={**VALID_PAYLOAD, "name": "all_cloud"})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_create_config_invalid_name(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs", json={**VALID_PAYLOAD, "name": "bad name!"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_config_invalid_provider(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs", json={**VALID_PAYLOAD, "tool_provider": "badprovider"})
    assert r.status_code == 422


# ── PUT /config/hybrid-configs/{name} ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_config_success(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    update = {k: v for k, v in VALID_PAYLOAD.items() if k != "name"}
    update["tool_model"] = "claude-sonnet-4-5-20250929"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.put("/config/hybrid-configs/all_cloud", json=update)
    assert r.status_code == 200
    assert r.json()["tool_model"] == "claude-sonnet-4-5-20250929"


@pytest.mark.asyncio
async def test_update_config_not_found(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    update = {k: v for k, v in VALID_PAYLOAD.items() if k != "name"}
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.put("/config/hybrid-configs/nonexistent_xyz", json=update)
    assert r.status_code == 404


# ── DELETE /config/hybrid-configs/{name} ──────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_config_success(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            create_r = await client.post("/config/hybrid-configs", json=VALID_PAYLOAD)
            assert create_r.status_code == 201
            del_r = await client.delete("/config/hybrid-configs/test_new_config")
    assert del_r.status_code == 204


@pytest.mark.asyncio
async def test_delete_config_not_found(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.delete("/config/hybrid-configs/does_not_exist")
    assert r.status_code == 404


# ── POST /config/hybrid-configs/{name}/clone ──────────────────────────────────

@pytest.mark.asyncio
async def test_clone_config_success(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs/all_cloud/clone",
                                  json={"new_name": "all_cloud_copy"})
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "all_cloud_copy"
    assert data["tool_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_clone_config_source_not_found(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs/ghost_config/clone",
                                  json={"new_name": "ghost_copy"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_clone_config_name_conflict(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs/all_cloud/clone",
                                  json={"new_name": "hybrid_qwen"})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_clone_config_invalid_name(app, tmp_path):
    yaml_path = tmp_path / "hybrid_llm.yaml"
    with patch.object(hlm_module, "_YAML_PATH", yaml_path):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs/all_cloud/clone",
                                  json={"new_name": "bad name!"})
    assert r.status_code == 422
