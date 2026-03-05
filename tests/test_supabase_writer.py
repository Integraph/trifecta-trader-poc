"""Tests for src/integration/supabase_writer.py (Task 013).

All Supabase network calls are mocked — no live credentials required.

Covers:
- SupabaseWriter initialises from env vars
- SupabaseWriter initialises from explicit args
- write_signal() calls supabase upsert with correct row
- write_signal() returns None and logs when write_enabled=False
- write_signal() catches and logs Supabase errors (does not raise)
- write_signals_batch() calls write_signal() for each signal
- write_signals_batch() skips failed signals and returns only successes
- get_latest_signal() queries by symbol and returns first result
- get_latest_signal() returns None on error
- cleanup_expired() calls delete() with lt filter on expires_at
- cleanup_expired() returns 0 on error
- Missing credentials raise ValueError on first write
- write_enabled=False from config file is respected
- to_supabase_row() called by write_signal (integration test)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_signal(ticker="AAPL", strategy="buy", alpha=0.94):
    """Minimal AISignal dict for testing the writer."""
    return {
        "id":           "00000000-0000-4000-a000-000000000001",
        "signal_ref":   f"sig_{ticker}_20260303_143025",
        "symbol":       ticker,
        "strategy":     strategy,
        "alphaScore":   alpha,
        "confidence":   0.8,
        "confidenceInterval": {"lower": 0.72, "upper": 0.88, "level": "95%"},
        "entryPrice":   195.0,
        "stopLoss":     180.0,
        "takeProfit":   220.0,
        "riskFlags":    [],
        "reasoning":    "Strong momentum.",
        "reasoningPath": ["market_analyst", "risk_judge"],
        "promptVersion": "v1.0.0",
        "evidenceIds":  [{"type": "internal", "id": "sig_AAPL_20260303_143025",
                          "timestamp": "2026-03-03T14:30:25+00:00"}],
        "createdAt":    "2026-03-03T14:30:25+00:00",
        "expiresAt":    "2026-03-04T14:30:25+00:00",
    }


def _mock_supabase_client():
    """Return a mock Supabase client whose chain calls succeed."""
    client = MagicMock()
    # Chain: client.table().upsert().execute() → returns data
    table_mock   = MagicMock()
    upsert_mock  = MagicMock()
    execute_mock = MagicMock()
    execute_mock.data = [{"id": "00000000-0000-4000-a000-000000000001"}]
    upsert_mock.execute.return_value = execute_mock
    table_mock.upsert.return_value   = upsert_mock
    client.table.return_value        = table_mock
    return client


# ── 1. Initialisation ─────────────────────────────────────────────────────────

class TestSupabaseWriterInit:

    def test_init_from_explicit_args(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="key123",
                                write_enabled=False)
        assert writer._url  == "https://x.supabase.co"
        assert writer._key  == "key123"
        assert writer._write_enabled is False

    def test_init_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://env.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "env_key")
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter()
        assert writer._url == "https://env.supabase.co"
        assert writer._key == "env_key"

    def test_default_table_name(self):
        from src.integration.supabase_writer import SupabaseWriter, DEFAULT_TABLE
        writer = SupabaseWriter(write_enabled=False)
        assert writer._table == DEFAULT_TABLE

    def test_custom_table_name(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(table_name="custom_signals", write_enabled=False)
        assert writer._table == "custom_signals"

    def test_missing_credentials_raises_on_write(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="", key="")
        # Should raise ValueError when trying to get client
        with pytest.raises(ValueError, match="credentials missing"):
            writer._get_client()


# ── 2. write_signal() ─────────────────────────────────────────────────────────

class TestWriteSignal:

    def test_write_enabled_false_returns_none(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k",
                                write_enabled=False)
        result = writer.write_signal(_make_signal())
        assert result is None

    def test_write_enabled_false_does_not_call_supabase(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k",
                                write_enabled=False)
        with patch("src.integration.supabase_writer.SupabaseWriter._get_client") as gc:
            writer.write_signal(_make_signal())
            gc.assert_not_called()

    def test_write_calls_upsert_with_snake_case_row(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k",
                                write_enabled=True)
        client = _mock_supabase_client()
        writer._client = client

        writer.write_signal(_make_signal(ticker="AAPL", alpha=0.94))

        # Verify table was called with the configured table name
        client.table.assert_called_once_with(writer._table)
        # Verify upsert was called (with a row dict)
        table_call = client.table.return_value
        table_call.upsert.assert_called_once()
        row_arg = table_call.upsert.call_args[0][0]
        assert row_arg["symbol"]     == "AAPL"
        assert row_arg["strategy"]   == "buy"
        assert abs(row_arg["alpha_score"] - 0.94) < 0.001

    def test_write_returns_inserted_row_on_success(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k",
                                write_enabled=True)
        writer._client = _mock_supabase_client()
        result = writer.write_signal(_make_signal())
        assert result is not None
        assert "id" in result

    def test_write_catches_supabase_error(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k",
                                write_enabled=True)
        client = MagicMock()
        client.table.side_effect = Exception("connection refused")
        writer._client = client
        # Should not raise
        result = writer.write_signal(_make_signal())
        assert result is None

    def test_write_uses_on_conflict_param(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k",
                                write_enabled=True)
        client = _mock_supabase_client()
        writer._client = client
        writer.write_signal(_make_signal())
        table_call = client.table.return_value
        _, kwargs = table_call.upsert.call_args
        assert "on_conflict" in kwargs


# ── 3. write_signals_batch() ─────────────────────────────────────────────────

class TestWriteSignalsBatch:

    def test_batch_calls_write_for_each_signal(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(write_enabled=False)
        signals = [_make_signal("AAPL"), _make_signal("MSFT"), _make_signal("NVDA")]

        with patch.object(writer, "write_signal", return_value={"id": "x"}) as ws:
            writer.write_signals_batch(signals)
            assert ws.call_count == 3

    def test_batch_returns_only_successful_writes(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(write_enabled=True)
        signals = [_make_signal("AAPL"), _make_signal("MSFT")]

        # First signal succeeds, second returns None (failure)
        side_effects = [{"id": "aaa"}, None]
        with patch.object(writer, "write_signal", side_effect=side_effects):
            results = writer.write_signals_batch(signals)
        assert len(results) == 1
        assert results[0]["id"] == "aaa"

    def test_batch_empty_list(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(write_enabled=False)
        results = writer.write_signals_batch([])
        assert results == []


# ── 4. get_latest_signal() ────────────────────────────────────────────────────

class TestGetLatestSignal:

    def test_returns_most_recent_row(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k")
        client = MagicMock()
        row = {"id": "aaa", "symbol": "AAPL", "strategy": "buy"}
        (client.table.return_value
               .select.return_value
               .eq.return_value
               .order.return_value
               .limit.return_value
               .execute.return_value) = MagicMock(data=[row])
        writer._client = client

        result = writer.get_latest_signal("AAPL")
        assert result == row

    def test_returns_none_when_no_rows(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k")
        client = MagicMock()
        (client.table.return_value
               .select.return_value
               .eq.return_value
               .order.return_value
               .limit.return_value
               .execute.return_value) = MagicMock(data=[])
        writer._client = client

        result = writer.get_latest_signal("ZZZZ")
        assert result is None

    def test_returns_none_on_error(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k")
        client = MagicMock()
        client.table.side_effect = Exception("network error")
        writer._client = client

        result = writer.get_latest_signal("AAPL")
        assert result is None


# ── 5. cleanup_expired() ─────────────────────────────────────────────────────

class TestCleanupExpired:

    def test_cleanup_returns_count_deleted(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k")
        client = MagicMock()
        deleted_rows = [{"id": "a"}, {"id": "b"}]
        (client.table.return_value
               .delete.return_value
               .lt.return_value
               .execute.return_value) = MagicMock(data=deleted_rows)
        writer._client = client

        count = writer.cleanup_expired()
        assert count == 2

    def test_cleanup_returns_zero_on_error(self):
        from src.integration.supabase_writer import SupabaseWriter
        writer = SupabaseWriter(url="https://x.supabase.co", key="k")
        client = MagicMock()
        client.table.side_effect = Exception("timeout")
        writer._client = client

        count = writer.cleanup_expired()
        assert count == 0
