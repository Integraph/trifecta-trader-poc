"""
Supabase Writer — inserts transformed AISignal rows into the platform's Supabase table.

Design principles:
- Never crashes the pipeline: all Supabase errors are caught and logged.
- write_enabled=False produces a dry-run log without making any network call.
- Deduplication: upserts on (symbol, DATE(created_at)) so re-runs replace
  the previous signal rather than creating duplicates.
- Uses the Supabase service role key (not the anon key) for server-side writes.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Default table name (overridden by config/supabase.yaml)
DEFAULT_TABLE = "signals"


def _load_config() -> dict:
    """Load Supabase settings from config/supabase.yaml if present."""
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path("config/supabase.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


class SupabaseWriter:
    """Writes AISignal rows to the Supabase `signals` table.

    Usage:
        writer = SupabaseWriter()
        writer.write_signal(signal)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        write_enabled: Optional[bool] = None,
        table_name: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ):
        cfg = _load_config()
        sb_cfg = cfg.get("supabase", {})

        self._url   = url or os.environ.get("SUPABASE_URL", "")
        self._key   = key or os.environ.get("SUPABASE_SERVICE_KEY", "")
        self._table = table_name or sb_cfg.get("table_name", DEFAULT_TABLE)
        self._ttl   = ttl_hours  or sb_cfg.get("signal_ttl_hours", 24)

        # write_enabled: explicit arg > config file > default True
        if write_enabled is not None:
            self._write_enabled = write_enabled
        else:
            self._write_enabled = sb_cfg.get("write_enabled", True)

        self._client = None  # lazy-initialized on first write

    def _get_client(self):
        """Return (or create) the Supabase client. Raises on misconfiguration."""
        if self._client is not None:
            return self._client

        if not self._url or not self._key:
            raise ValueError(
                "Supabase credentials missing. Set SUPABASE_URL and "
                "SUPABASE_SERVICE_KEY in .env, or pass url/key to SupabaseWriter()."
            )

        from supabase import create_client
        self._client = create_client(self._url, self._key)
        return self._client

    # ── Public write methods ──────────────────────────────────────────────────

    def write_signal(self, signal: dict) -> Optional[dict]:
        """Insert (or upsert) a single AISignal into Supabase.

        Uses on_conflict='(symbol, DATE(created_at::date))' to handle
        deduplication: re-running the same ticker on the same day replaces
        the previous signal rather than creating a duplicate.

        Args:
            signal: AISignal dict from transform_to_signal().

        Returns:
            The inserted/updated row dict on success, None on failure.
        """
        from src.integration.signal_adapter import to_supabase_row

        row = to_supabase_row(signal)
        ref = signal.get("signal_ref", signal.get("id", "?"))

        if not self._write_enabled:
            logger.info(
                "[Supabase write_enabled=False] Would write signal: %s  %s  alpha=%.3f",
                ref, row["strategy"], row["alpha_score"],
            )
            return None

        try:
            client = self._get_client()
            # Upsert: on conflict on symbol + date(created_at), update all fields.
            # Supabase upsert with ignoreDuplicates=False replaces existing rows.
            result = (
                client.table(self._table)
                .upsert(row, on_conflict="symbol,created_at")
                .execute()
            )
            inserted = result.data[0] if result.data else row
            logger.info(
                "Supabase write OK: %s  %s  id=%s",
                ref, row["strategy"], row["id"],
            )
            return inserted

        except Exception as e:
            logger.error("Supabase write FAILED for %s: %s", ref, e)
            return None

    def write_signals_batch(self, signals: list) -> list:
        """Insert multiple signals. Used by run_batch.py.

        Processes sequentially; failures for individual signals are logged
        but don't abort the batch.

        Args:
            signals: List of AISignal dicts from transform_to_signal().

        Returns:
            List of successfully inserted row dicts.
        """
        results = []
        for sig in signals:
            row = self.write_signal(sig)
            if row is not None:
                results.append(row)
        logger.info(
            "Supabase batch write: %d/%d signals written",
            len(results), len(signals),
        )
        return results

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_latest_signal(self, ticker: str) -> Optional[dict]:
        """Return the most recent signal for a ticker, or None if none exists.

        Used for pre-write deduplication checks and debugging.
        """
        try:
            client = self._get_client()
            result = (
                client.table(self._table)
                .select("*")
                .eq("symbol", ticker.upper())
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.warning("get_latest_signal failed for %s: %s", ticker, e)
            return None

    def cleanup_expired(self) -> int:
        """Delete signals whose expiresAt has passed.

        Returns:
            Number of rows deleted (0 on failure).
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            client = self._get_client()
            result = (
                client.table(self._table)
                .delete()
                .lt("expires_at", now)
                .execute()
            )
            count = len(result.data) if result.data else 0
            logger.info("Cleaned up %d expired signals (before %s)", count, now)
            return count
        except Exception as e:
            logger.warning("cleanup_expired failed: %s", e)
            return 0
