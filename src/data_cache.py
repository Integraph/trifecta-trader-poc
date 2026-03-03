"""
TTL-based file cache for analyst outputs.

Caches the full text output of each analyst agent (market, social, news,
fundamentals) by ticker + analyst_type + date. When a cached value is still
valid (within TTL), HybridTradingGraph can inject it directly into graph state
and skip re-running that analyst's LLM call.

Cache layout:
    cache/
        <ticker>/
            <analyst_type>_<YYYY-MM-DD>.json   # {"value": "...", "expires_at": 12345.6}

Keys:
    Format: "{ticker}:{analyst_type}:{YYYY-MM-DD}"
    Example: "AAPL:market:2026-03-02"
"""

import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Per-analyst TTLs (seconds)
ANALYST_TTL = {
    "fundamentals": 86400,   # 24 hours — financials don't change intraday
    "market":       3600,    # 1 hour — price-sensitive
    "news":         1800,    # 30 minutes
    "social":       900,     # 15 minutes
}

DEFAULT_TTL = 3600  # 1 hour fallback


class DataCache:
    """TTL-based file cache for analyst-level text outputs.

    Stores each analyst's report as a JSON file under cache/<ticker>/.
    Expiry is encoded in the file itself as a Unix timestamp.

    Usage:
        cache = DataCache(cache_dir="cache")
        key = cache.key_for("AAPL", "market", "2026-03-02")

        value = cache.get(key)          # Returns None if expired/missing
        cache.set(key, text, ttl=3600)  # Store with TTL
        cache.clear(ticker="AAPL")      # Invalidate all entries for a ticker
    """

    def __init__(self, cache_dir: str = "cache", default_ttl: int = DEFAULT_TTL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def key_for(self, ticker: str, analyst_type: str, trade_date: str) -> str:
        """Generate a cache key.

        Args:
            ticker: Stock ticker (e.g. "AAPL")
            analyst_type: One of: market, social, news, fundamentals
            trade_date: Date string "YYYY-MM-DD"

        Returns:
            Cache key string "TICKER:analyst_type:YYYY-MM-DD"
        """
        return f"{ticker.upper()}:{analyst_type}:{trade_date}"

    def get(self, key: str) -> Optional[str]:
        """Return cached value if present and within TTL, else None."""
        path = self._path_for(key)
        if not path.exists():
            self._misses += 1
            return None

        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None

        if time.time() > entry.get("expires_at", 0):
            logger.debug("Cache expired for key %s", key)
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

        self._hits += 1
        logger.info("Cache HIT: %s", key)
        return entry["value"]

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store value with TTL.

        Args:
            key: Cache key from key_for()
            value: Text to cache (analyst report content)
            ttl: Time-to-live in seconds (defaults to self.default_ttl)
        """
        if not value:
            return  # Don't cache empty strings

        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "key": key,
            "value": value,
            "expires_at": time.time() + (ttl if ttl is not None else self.default_ttl),
            "cached_at": time.time(),
        }
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

        logger.info("Cache SET: %s (ttl=%ds)", key, ttl or self.default_ttl)

    def clear(self, ticker: Optional[str] = None) -> int:
        """Delete cache entries.

        Args:
            ticker: If given, delete only entries for this ticker.
                    If None, delete all cache entries.

        Returns:
            Number of entries deleted.
        """
        deleted = 0
        if ticker:
            ticker_dir = self.cache_dir / ticker.upper()
            if ticker_dir.exists():
                for f in ticker_dir.glob("*.json"):
                    f.unlink()
                    deleted += 1
                logger.info("Cleared %d cache entries for %s", deleted, ticker)
        else:
            for f in self.cache_dir.rglob("*.json"):
                f.unlink()
                deleted += 1
            logger.info("Cleared %d total cache entries", deleted)
        return deleted

    def stats(self) -> dict:
        """Return hit/miss statistics since this instance was created."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate_pct": round(hit_rate, 1),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> Path:
        """Convert a cache key to a file path.

        Key format: "TICKER:analyst_type:YYYY-MM-DD"
        File path:  cache/TICKER/analyst_type_YYYY-MM-DD.json
        """
        parts = key.split(":", 2)
        if len(parts) == 3:
            ticker, analyst_type, date = parts
            return self.cache_dir / ticker / f"{analyst_type}_{date}.json"
        # Fallback: hash the key
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.json"
