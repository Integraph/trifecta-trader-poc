"""
FastAPI dependency injection — provides shared daemon and DB instances.

In full mode (started via run_daemon --api), init_dependencies() is called
before the API server starts, wiring in the live daemon and DB objects.

In standalone/degraded mode (API started directly for dev/testing),
get_daemon() returns None, and endpoints that require a live daemon
return 503 Service Unavailable.
"""

from typing import Optional
from fastapi import HTTPException

_daemon = None
_db     = None


def init_dependencies(daemon, db) -> None:
    """Wire shared instances into the DI layer. Called once at server startup."""
    global _daemon, _db
    _daemon = daemon
    _db     = db


def get_daemon():
    """Return the shared PipelineDaemon instance (may be None in standalone mode)."""
    return _daemon


def get_db():
    """Return the shared PortfolioDatabase instance.

    Creates a default instance if none was injected (standalone dev mode).
    """
    global _db
    if _db is None:
        from src.portfolio.database import PortfolioDatabase
        _db = PortfolioDatabase()
    return _db


def require_daemon():
    """Dependency that raises 503 if daemon is not running.

    Use as a FastAPI dependency for endpoints that need the live daemon.
    """
    daemon = get_daemon()
    if daemon is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "daemon_not_running",
                "message": "This endpoint requires a running daemon. "
                           "Start with: python -m src.run_daemon --api",
            },
        )
    return daemon
