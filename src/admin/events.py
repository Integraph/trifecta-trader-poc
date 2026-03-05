"""
Admin Event Bus — collects pipeline lifecycle events and broadcasts
them to connected WebSocket clients via /ws/events.

Design:
- In-process singleton event bus (asyncio-friendly).
- Subsystems call publish(event_type, data) via their _event_callback.
- The WebSocket handler subscribes and receives events via asyncio.Queue.
- Multiple clients can be connected simultaneously (fan-out).

Thread safety:
- publish() may be called from background threads (scheduler, queue reader).
- It uses asyncio.run_coroutine_threadsafe() to safely enqueue into the
  asyncio event loop running the FastAPI server.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


class EventBus:
    """In-process publish/subscribe event bus for admin WebSocket streaming."""

    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._history: list = []        # last 200 events for late joiners
        self._max_history = 200

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the asyncio event loop. Called once at server startup."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket client. Returns a Queue to read from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a disconnected WebSocket client."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def publish(self, event_type: str, data: dict) -> None:
        """Publish an event from any thread.

        Can be called from background threads (scheduler, queue reader).
        """
        event = {
            "event":     event_type,
            "data":      data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Store in bounded history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if not self._subscribers:
            return

        if self._loop and self._loop.is_running():
            # Called from a background thread — schedule into the event loop
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self._loop)
        else:
            logger.debug("EventBus: no running loop, event %s not broadcast", event_type)

    async def _broadcast(self, event: dict) -> None:
        """Push an event to all subscribed queues (async, runs on event loop)."""
        disconnected = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("EventBus: subscriber queue full, dropping event")
            except Exception as e:
                logger.warning("EventBus: subscriber error: %s", e)
                disconnected.append(q)
        for q in disconnected:
            self.unsubscribe(q)

    def recent_events(self, limit: int = 50) -> list:
        """Return recent events from the in-memory history."""
        return self._history[-limit:]


# ── Module-level singleton ────────────────────────────────────────────────────

_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def make_event_callback():
    """Return a thread-safe callback suitable for passing to subsystems."""
    bus = get_event_bus()

    def _callback(event_type: str, data: dict) -> None:
        try:
            bus.publish(event_type, data)
        except Exception:
            pass  # never let event delivery break the caller

    return _callback


# ── FastAPI Router ─────────────────────────────────────────────────────────────

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

events_router = APIRouter()


@events_router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """WebSocket endpoint — pushes real-time pipeline events to the client.

    Connect with: ws://localhost:8420/ws/events
    """
    await websocket.accept()
    bus = get_event_bus()

    # Register event loop on first connection (needed for thread-safe publish)
    if bus._loop is None:
        bus.set_loop(asyncio.get_event_loop())

    queue = bus.subscribe()
    logger.info("WebSocket /ws/events client connected")

    try:
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event, default=str))
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/events client disconnected")
    except Exception as e:
        logger.warning("WebSocket /ws/events error: %s", e)
    finally:
        bus.unsubscribe(queue)
