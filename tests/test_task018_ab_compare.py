"""
Task 018 — Tests: A/B comparison endpoints + sanity check
POST /test-run/ab
GET  /test-run/ab/{ab_id}
POST /config/hybrid-configs/{name}/sanity-check
"""
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport


# ── Global CONFIGS isolation ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def restore_configs():
    import src.hybrid_llm as mod
    from src.hybrid_llm import HybridLLMConfig
    snapshot = {k: HybridLLMConfig(**v.to_flat_dict()) for k, v in mod.CONFIGS.items()}
    yield
    mod.CONFIGS.clear()
    mod.CONFIGS.update(snapshot)


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    from src.admin.app import create_app
    return create_app(daemon=None, db=None)


def make_client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── POST /test-run/ab ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_ab_compare_success(app):
    async with make_client(app) as client:
        r = await client.post("/test-run/ab", json={
            "ticker": "aapl",
            "trade_date": "2025-10-01",
            "config_a": "all_cloud",
            "config_b": "hybrid_qwen",
        })
    assert r.status_code == 202
    data = r.json()
    assert "ab_id" in data
    assert "task_id_a" in data
    assert "task_id_b" in data
    assert data["ticker"] == "AAPL"
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_submit_ab_normalises_ticker_uppercase(app):
    async with make_client(app) as client:
        r = await client.post("/test-run/ab", json={
            "ticker": "nvda", "trade_date": "2025-10-01",
            "config_a": "all_cloud", "config_b": "hybrid_qwen",
        })
    assert r.status_code == 202
    assert r.json()["ticker"] == "NVDA"


@pytest.mark.asyncio
async def test_submit_ab_invalid_config_a(app):
    async with make_client(app) as client:
        r = await client.post("/test-run/ab", json={
            "ticker": "TSLA", "trade_date": "2025-10-01",
            "config_a": "nonexistent_config", "config_b": "hybrid_qwen",
        })
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_submit_ab_invalid_config_b(app):
    async with make_client(app) as client:
        r = await client.post("/test-run/ab", json={
            "ticker": "TSLA", "trade_date": "2025-10-01",
            "config_a": "all_cloud", "config_b": "ghost_config",
        })
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_submit_ab_both_configs_invalid(app):
    async with make_client(app) as client:
        r = await client.post("/test-run/ab", json={
            "ticker": "TSLA", "trade_date": "2025-10-01",
            "config_a": "ghost_a", "config_b": "ghost_b",
        })
    assert r.status_code == 404


# ── GET /test-run/ab/{ab_id} ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_poll_ab_not_found(app):
    async with make_client(app) as client:
        r = await client.get("/test-run/ab/does_not_exist")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_poll_ab_returns_both_sides(app):
    """After submitting, polling should return both config sides in the response."""
    async with make_client(app) as client:
        post_r = await client.post("/test-run/ab", json={
            "ticker": "GOOGL", "trade_date": "2025-10-01",
            "config_a": "all_cloud", "config_b": "hybrid_qwen",
        })
        ab_id = post_r.json()["ab_id"]
        get_r = await client.get(f"/test-run/ab/{ab_id}")
    assert get_r.status_code == 200
    data = get_r.json()
    assert data["ab_id"] == ab_id
    assert "config_a" in data
    assert "config_b" in data
    assert data["config_a"]["name"] == "all_cloud"
    assert data["config_b"]["name"] == "hybrid_qwen"
    assert data["ticker"] == "GOOGL"


@pytest.mark.asyncio
async def test_poll_ab_status_field_present(app):
    async with make_client(app) as client:
        post_r = await client.post("/test-run/ab", json={
            "ticker": "MSFT", "trade_date": "2025-10-01",
            "config_a": "all_cloud", "config_b": "hybrid_qwen",
        })
        ab_id = post_r.json()["ab_id"]
        get_r = await client.get(f"/test-run/ab/{ab_id}")
    data = get_r.json()
    assert data["status"] in ("running", "complete")


# ── _AB_STORE capacity cap ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ab_store_capped_at_20(app):
    """Submitting more than 20 A/B comparisons evicts the oldest."""
    import src.admin.test_run as tr_module
    from src.admin.test_run import _AB_STORE

    _AB_STORE.clear()
    async with make_client(app) as client:
        for i in range(25):
            await client.post("/test-run/ab", json={
                "ticker": f"T{i:02d}", "trade_date": "2025-10-01",
                "config_a": "all_cloud", "config_b": "hybrid_qwen",
            })

    assert len(_AB_STORE) <= 20
    _AB_STORE.clear()


# ── Sanity check endpoint smoke test ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_sanity_check_not_found(app):
    async with make_client(app) as client:
        r = await client.post("/config/hybrid-configs/ghost_config/sanity-check")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_sanity_check_returns_structure(app):
    """Sanity check runs (may fail connectivity) but returns correct structure."""
    with patch("src.admin.sanity_check._check_slot", return_value={
        "provider": "anthropic", "model": "test", "status": "pass", "latency_ms": 42, "error": None,
    }):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs/all_cloud/sanity-check")

    assert r.status_code == 200
    data = r.json()
    assert "config_name" in data
    assert "overall" in data
    assert "checks" in data
    assert "tool_calling" in data["checks"]
    assert "reasoning_quick" in data["checks"]
    assert "reasoning_deep" in data["checks"]
    assert data["overall"] in ("pass", "partial", "fail")


@pytest.mark.asyncio
async def test_sanity_check_config_name_in_response(app):
    with patch("src.admin.sanity_check._check_slot", return_value={
        "provider": "ollama", "model": "qwen2.5:14b", "status": "pass", "latency_ms": 200, "error": None,
    }):
        async with make_client(app) as client:
            r = await client.post("/config/hybrid-configs/hybrid_qwen/sanity-check")
    assert r.json()["config_name"] == "hybrid_qwen"
