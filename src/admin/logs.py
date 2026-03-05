"""
Log Streaming — REST and WebSocket endpoints for daemon log access.

Two complementary approaches:
1. GET /logs/recent   — returns last N entries from the AdminLogHandler buffer
2. WebSocket /ws/logs — real-time streaming via asyncio.Queue

AdminLogHandler is a custom logging.Handler that:
- Captures all log records into a bounded in-memory deque (last 500 entries)
- Pushes each record into subscriber queues for WebSocket delivery
- Is injected into the root logger when the API server starts (--api flag)
- Has zero overhead when the API is not running (not installed by default)
"""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

logs_router = APIRouter()


# ── AdminLogHandler ────────────────────────────────────────────────────────────

class AdminLogHandler(logging.Handler):
    """Custom logging.Handler that feeds log entries to the Admin API.

    - Maintains a bounded ring buffer (last 500 entries) for GET /logs/recent.
    - Pushes entries into subscriber queues for WebSocket streaming.
    - Thread-safe: Python's GIL protects deque appends and list iteration.
    """

    def __init__(self, max_buffer: int = 500):
        super().__init__()
        self._buffer: deque = deque(maxlen=max_buffer)
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self.setFormatter(formatter)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level":     record.levelname,
                "logger":    record.name,
                "message":   record.getMessage(),
            }
            self._buffer.append(entry)

            if self._subscribers and self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(entry), self._loop
                )
        except Exception:
            self.handleError(record)

    async def _broadcast(self, entry: dict) -> None:
        disconnected = []
        for q in self._subscribers:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass
            except Exception:
                disconnected.append(q)
        for q in disconnected:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def recent(self, lines: int = 100, level: str = "DEBUG") -> list:
        """Return the last N log entries at or above the specified level."""
        min_level = _LOG_LEVELS.get(level.upper(), 0)
        entries   = [e for e in self._buffer if _LOG_LEVELS.get(e["level"], 0) >= min_level]
        return list(entries[-lines:])


# Module-level singleton handler (installed into root logger by app.py)
_admin_handler: Optional[AdminLogHandler] = None


def get_admin_handler() -> Optional[AdminLogHandler]:
    return _admin_handler


def install_admin_handler(loop: asyncio.AbstractEventLoop) -> AdminLogHandler:
    """Create and inject AdminLogHandler into the root logger.

    Called once when the API server starts.
    """
    global _admin_handler
    if _admin_handler is None:
        _admin_handler = AdminLogHandler()
    _admin_handler.set_loop(loop)
    root = logging.getLogger()
    # Avoid duplicate installation
    if _admin_handler not in root.handlers:
        root.addHandler(_admin_handler)
    return _admin_handler


# ── REST endpoints ─────────────────────────────────────────────────────────────

@logs_router.get("/recent")
async def get_recent_logs(
    lines: int = Query(default=100, ge=1, le=2000),
    level: str = Query(default="DEBUG"),
):
    """Return recent log entries from the in-memory buffer.

    Falls back to tailing logs/daemon.log if the AdminLogHandler buffer
    is empty (e.g., API started standalone without daemon logging).
    """
    handler = get_admin_handler()
    if handler:
        entries = handler.recent(lines=lines, level=level)
        if entries:
            return {"lines": entries[::-1]}  # newest first

    # Fallback: read from log file
    from pathlib import Path
    log_path = Path("logs/daemon.log")
    if not log_path.exists():
        return {"lines": []}

    try:
        all_lines = log_path.read_text(errors="replace").splitlines()
        # Filter by level
        min_level = _LOG_LEVELS.get(level.upper(), 0)
        parsed = []
        for raw in all_lines:
            # Parse: "2026-03-05T14:30:00 [INFO] src.module: message"
            try:
                parts = raw.split(" ", 2)
                if len(parts) >= 3 and "[" in parts[1]:
                    lvl = parts[1].strip("[]")
                    if _LOG_LEVELS.get(lvl, 0) >= min_level:
                        parsed.append({
                            "timestamp": parts[0],
                            "level":     lvl,
                            "logger":    "",
                            "message":   parts[2],
                        })
            except Exception:
                continue
        return {"lines": parsed[-lines:][::-1]}
    except Exception as e:
        return {"lines": [], "error": str(e)}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@logs_router.websocket("/ws/logs")
async def ws_logs(
    websocket: WebSocket,
    level: str = Query(default="DEBUG"),
):
    """WebSocket endpoint — streams log entries in real-time.

    Connect with: ws://localhost:8420/ws/logs?level=INFO
    """
    await websocket.accept()
    handler = get_admin_handler()
    min_level = _LOG_LEVELS.get(level.upper(), 0)
    logger.info("WebSocket /ws/logs connected (level=%s)", level)

    if handler is None:
        await websocket.send_text(json.dumps({
            "level": "WARNING",
            "message": "AdminLogHandler not installed — log streaming unavailable in standalone mode.",
        }))
        await websocket.close()
        return

    # Ensure the handler knows about the current event loop
    if handler._loop is None:
        handler.set_loop(asyncio.get_event_loop())

    queue = handler.subscribe()
    try:
        while True:
            entry = await queue.get()
            if _LOG_LEVELS.get(entry.get("level", ""), 0) >= min_level:
                await websocket.send_text(json.dumps(entry))
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/logs client disconnected")
    except Exception as e:
        logger.warning("WebSocket /ws/logs error: %s", e)
    finally:
        handler.unsubscribe(queue)
