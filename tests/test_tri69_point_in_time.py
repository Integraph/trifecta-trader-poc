"""TRI-69 — tests for temperature plumbing and point-in-time no-leak mode.

Integrity-doc discipline: each guard has a negative test (the guard REJECTS
bad input — passes by rejecting). The on-demand mutation demonstrations
(break the guard → suite goes red) are recorded in docs/TASK_TRI-69_REPORT.md.

No network: vendor implementations are replaced with spies before enabling
point-in-time mode, so these tests exercise the real wrapping/clamping code
against real-shaped data without touching yfinance/Polymarket.
"""

import copy

import pytest
from dotenv import load_dotenv

load_dotenv()


# ── Temperature plumbing (Config A prerequisite) ──────────────────────────────

class TestTemperaturePlumbing:
    def _make_config(self, temperature, **kw):
        from src.hybrid_llm import HybridLLMConfig
        return HybridLLMConfig(
            tool_provider="ollama", tool_model="qwen3-coder:30b",
            reasoning_quick_provider="ollama", reasoning_quick_model="qwen3.5:9b",
            reasoning_deep_provider="anthropic",
            reasoning_deep_model="claude-sonnet-4-5-20250929",
            enhance_local=False, enhance_deep=False,
            temperature=temperature, **kw,
        )

    def test_temperature_zero_reaches_all_three_slots(self):
        """temp=0 must be applied by every client's get_llm(), not just deep."""
        from src.hybrid_llm import create_hybrid_llms
        llms = create_hybrid_llms(self._make_config(0.0))
        for slot in ("tool_calling_llm", "reasoning_quick_llm", "reasoning_deep_llm"):
            temp = getattr(llms[slot], "temperature", None)
            assert temp == 0.0, f"{slot} temperature={temp!r}, expected 0.0"

    def test_temperature_none_leaves_provider_defaults(self):
        """Without the pin, slots must NOT silently get temp=0 (negative)."""
        from src.hybrid_llm import create_hybrid_llms
        llms = create_hybrid_llms(self._make_config(None))
        for slot in ("tool_calling_llm", "reasoning_quick_llm"):
            temp = getattr(llms[slot], "temperature", None)
            assert temp != 0.0, f"{slot} got temp=0 without the config asking"

    def test_amended_config_a_pins(self):
        """Checkpoint-1 amendment: local slots get seed + max_tokens at
        model-default sampling; deep slot alone gets temperature=0."""
        from src.hybrid_llm import create_hybrid_llms
        cfg = self._make_config(None, deep_temperature=0.0,
                                local_seed=69, local_max_tokens=16384)
        llms = create_hybrid_llms(cfg)
        for slot in ("tool_calling_llm", "reasoning_quick_llm"):
            assert getattr(llms[slot], "seed", None) == 69, slot
            assert getattr(llms[slot], "max_tokens", None) == 16384, slot
            assert getattr(llms[slot], "temperature", None) != 0.0, slot
        deep = llms["reasoning_deep_llm"]
        assert deep.temperature == 0.0
        # the anthropic slot must NOT get the ollama pins
        assert getattr(deep, "seed", None) != 69

    def test_amended_yaml_round_trip(self):
        from src.hybrid_llm import _yaml_entry_to_config
        cfg = self._make_config(None, deep_temperature=0.0,
                                local_seed=69, local_max_tokens=16384)
        entry = cfg.to_flat_dict()
        back = _yaml_entry_to_config(entry)
        assert (back.deep_temperature, back.local_seed, back.local_max_tokens) \
            == (0.0, 69, 16384)
        assert "temperature" not in entry  # unset base temp stays absent

    def test_temperature_yaml_round_trip(self):
        from src.hybrid_llm import _yaml_entry_to_config
        cfg = self._make_config(0.0)
        entry = cfg.to_flat_dict()
        assert entry["temperature"] == 0.0
        assert _yaml_entry_to_config(entry).temperature == 0.0
        # Absent key -> None (legacy configs untouched)
        entry_legacy = self._make_config(None).to_flat_dict()
        assert "temperature" not in entry_legacy
        assert _yaml_entry_to_config(entry_legacy).temperature is None


# ── Point-in-time mode ─────────────────────────────────────────────────────────

ASOF = "2026-03-13"


@pytest.fixture
def pit(monkeypatch):
    """Enable point-in-time mode against spy vendor impls; restore after.

    Yields the spy call-recorder dict. VENDOR_METHODS and the patched module
    attributes are snapshotted and restored so the process-global patch never
    leaks into other tests.
    """
    import src.point_in_time as pit_mod
    from tradingagents.dataflows import interface as vi
    from tradingagents.dataflows import reddit as _reddit
    from tradingagents.dataflows import stocktwits as _st
    from tradingagents.dataflows import market_data_validator as _mdv
    from tradingagents.agents.analysts import sentiment_analyst as _sa
    from tradingagents.agents.utils import market_data_validation_tools as _mdvt

    saved_methods = copy.copy({m: copy.copy(v) for m, v in vi.VENDOR_METHODS.items()})
    saved = {
        "reddit": _reddit.fetch_reddit_posts,
        "st": _st.fetch_stocktwits_messages,
        "sa_reddit": _sa.fetch_reddit_posts,
        "sa_st": _sa.fetch_stocktwits_messages,
        "mdv": _mdv.build_verified_market_snapshot,
        "mdvt": _mdvt.build_verified_market_snapshot,
    }

    calls = {}

    def spy(method):
        def impl(*args, **kwargs):
            calls[method] = {"args": args, "kwargs": kwargs}
            return f"spy-output for {method}"
        return impl

    # Real-shaped registry entries replaced by spies BEFORE enabling, so the
    # wrapper wraps the spy and we can observe exactly what it forwards.
    for method in ("get_stock_data", "get_balance_sheet", "get_cashflow",
                   "get_income_statement", "get_news"):
        for vendor_name in vi.VENDOR_METHODS[method]:
            vi.VENDOR_METHODS[method][vendor_name] = spy(method)

    pit_mod.enable_point_in_time_mode(ASOF)
    yield calls

    vi.VENDOR_METHODS.clear()
    vi.VENDOR_METHODS.update(saved_methods)
    _reddit.fetch_reddit_posts = saved["reddit"]
    _st.fetch_stocktwits_messages = saved["st"]
    _sa.fetch_reddit_posts = saved["sa_reddit"]
    _sa.fetch_stocktwits_messages = saved["sa_st"]
    _mdv.build_verified_market_snapshot = saved["mdv"]
    _mdvt.build_verified_market_snapshot = saved["mdvt"]
    pit_mod._enabled_asof = None


class TestPointInTimeMode:
    def test_fundamentals_overview_neutralized(self, pit):
        from tradingagents.dataflows.interface import route_to_vendor
        out = route_to_vendor("get_fundamentals", "AAPL", ASOF)
        assert "point-in-time" in out
        assert "get_balance_sheet" in out  # points the agent at safe tools
        from src.point_in_time import FORBIDDEN_INFO_KEYS_RE, \
            FORBIDDEN_FUNDAMENTALS_LABELS_RE
        assert not FORBIDDEN_INFO_KEYS_RE.search(out)
        assert not FORBIDDEN_FUNDAMENTALS_LABELS_RE.search(out)

    def test_prediction_markets_and_insider_disabled(self, pit):
        from tradingagents.dataflows.interface import route_to_vendor
        assert "disabled in point-in-time" in route_to_vendor(
            "get_prediction_markets", "Fed rate cut", 6)
        assert "disabled in point-in-time" in route_to_vendor(
            "get_insider_transactions", "AAPL")

    def test_statement_curr_date_forced_when_omitted(self, pit):
        """The vendor date filter no-ops on curr_date=None — PIT must force it."""
        from tradingagents.dataflows.interface import route_to_vendor
        route_to_vendor("get_balance_sheet", "AAPL", "quarterly", None)
        assert pit["get_balance_sheet"]["args"][2] == ASOF
        route_to_vendor("get_cashflow", "AAPL")
        assert pit["get_cashflow"]["kwargs"]["curr_date"] == ASOF

    def test_future_date_args_clamped(self, pit):
        from tradingagents.dataflows.interface import route_to_vendor
        route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-12-31")
        assert pit["get_stock_data"]["args"] == ("AAPL", "2026-01-01", ASOF)
        # Past dates pass through untouched
        route_to_vendor("get_news", "AAPL", "2026-03-01", "2026-03-10")
        assert pit["get_news"]["args"] == ("AAPL", "2026-03-01", "2026-03-10")

    def test_leak_guard_rejects_forbidden_fields(self, pit):
        """Negative test: a leaked info-key in any output must RAISE."""
        from tradingagents.dataflows import interface as vi
        from src.point_in_time import LookAheadLeakError, _wrap_vendor_impl

        def leaky_impl(*a, **k):
            return "Beta: 1.2\nfiftyTwoWeekHigh: 260.10\n"

        wrapped = _wrap_vendor_impl("get_news", leaky_impl, ASOF)
        with pytest.raises(LookAheadLeakError):
            wrapped("AAPL", "2026-03-01", "2026-03-10")

        # Formatted labels are enforced on the fundamentals category
        def leaky_stmt(*a, **k):
            return "52 Week High: 260.10"
        wrapped_stmt = _wrap_vendor_impl("get_balance_sheet", leaky_stmt, ASOF)
        with pytest.raises(LookAheadLeakError):
            wrapped_stmt("AAPL", "quarterly", ASOF)

        # ...but prose mentions in news do NOT false-positive
        def prose_impl(*a, **k):
            return "Shares touched a 52-week high on optimism."
        wrapped_news = _wrap_vendor_impl("get_news", prose_impl, ASOF)
        assert "52-week high" in wrapped_news("AAPL", "2026-03-01", "2026-03-10")

    def test_retrieval_timestamp_normalized(self, pit):
        """Wall-clock 'Data retrieved on' stamps must be normalized: they
        made same-day repeats diverge (tri69b SELL/HOLD flip)."""
        from src.point_in_time import _wrap_vendor_impl

        def stamped(*a, **k):
            return ("# Prices for AAPL\n"
                    "# Data retrieved on: 2026-07-06 03:21:01\n\nrows...")
        def stamped2(*a, **k):
            return ("# Prices for AAPL\n"
                    "# Data retrieved on: 2026-07-06 03:21:03\n\nrows...")
        w1 = _wrap_vendor_impl("get_stock_data", stamped, ASOF)
        w2 = _wrap_vendor_impl("get_stock_data", stamped2, ASOF)
        out1 = w1("AAPL", "2026-03-01", ASOF)
        out2 = w2("AAPL", "2026-03-01", ASOF)
        assert out1 == out2                    # stamps no longer diverge
        assert "03:21:01" not in out1          # wall clock gone
        assert "point-in-time mode" in out1

    def test_live_social_disabled_in_analyst_namespace(self, pit):
        """The sentiment analyst binds these at import — its namespace must
        hold the stubs, not just the source dataflow modules."""
        from tradingagents.agents.analysts import sentiment_analyst as _sa
        assert "disabled in point-in-time" in _sa.fetch_reddit_posts("AAPL")
        assert "disabled in point-in-time" in _sa.fetch_stocktwits_messages(
            "AAPL", limit=30)

    def test_snapshot_curr_date_clamped(self, pit):
        from tradingagents.dataflows import market_data_validator as _mdv
        seen = {}

        # The clamp wrapper closed over the real builder at enable time; to
        # observe forwarding, call the wrapper with a future date and let the
        # real builder fail on the spy-free path — instead patch via module
        # indirection: wrapper calls the captured real function, so we assert
        # on the wrapper's clamping logic directly.
        from src.point_in_time import _clamp_date_args
        args, _ = _clamp_date_args(("AAPL", "2026-12-31", 30), {}, ASOF)
        assert args == ("AAPL", ASOF, 30)
        assert _mdv.build_verified_market_snapshot.__name__ == "clamped_snapshot"

    def test_reenable_same_date_idempotent_other_date_refused(self, pit):
        from src.point_in_time import enable_point_in_time_mode
        enable_point_in_time_mode(ASOF)  # idempotent
        with pytest.raises(RuntimeError):
            enable_point_in_time_mode("2026-04-02")
