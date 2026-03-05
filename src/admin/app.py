"""
Admin API — FastAPI application factory.

Usage (full mode via daemon):
    # Started automatically by run_daemon.py --api
    # Shared PipelineDaemon and PortfolioDatabase instances are injected.

Usage (standalone dev mode):
    python -m src.admin.app
    # DB-backed endpoints work; daemon-dependent endpoints return 503.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def create_app(daemon=None, db=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        daemon: Optional PipelineDaemon instance (None = standalone/degraded mode).
        db:     Optional PortfolioDatabase instance (auto-created if None).
    """
    from src.admin.dependencies import init_dependencies

    # Wire shared instances into DI layer
    init_dependencies(daemon=daemon, db=db)

    app = FastAPI(
        title="Trifecta Trader Admin API",
        version="1.0.0",
        description="Operational visibility and control for the Trifecta Trader pipeline.",
    )

    # CORS — allow all localhost origins for the admin frontend (Task 017)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception at %s: %s", request.url, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "message": str(exc)},
        )

    # ── Mount routers ─────────────────────────────────────────────────────────

    from src.admin.health    import health_router
    from src.admin.scheduler import scheduler_router
    from src.admin.queue     import queue_router
    from src.admin.accuracy  import accuracy_router
    from src.admin.config    import config_router
    from src.admin.test_run  import test_run_router
    from src.admin.logs      import logs_router
    from src.admin.analyses  import analyses_router
    from src.admin.events    import events_router
    from src.admin.tasks     import tasks_router

    app.include_router(health_router,    tags=["Health"])
    app.include_router(scheduler_router, prefix="/scheduler",  tags=["Scheduler"])
    app.include_router(queue_router,     prefix="/queue",      tags=["Queue"])
    app.include_router(accuracy_router,  prefix="/accuracy",   tags=["Accuracy"])
    app.include_router(config_router,    prefix="/config",     tags=["Configuration"])
    app.include_router(test_run_router,  prefix="/test-run",   tags=["Test Run"])
    app.include_router(logs_router,      prefix="/logs",       tags=["Logs"])
    app.include_router(analyses_router,  prefix="/analyses",   tags=["Analyses"])
    app.include_router(events_router,    tags=["Events"])
    app.include_router(tasks_router,     prefix="/tasks",      tags=["Tasks"])

    # ── Startup: install admin log handler and register event loop ────────────

    @app.on_event("startup")
    async def on_startup():
        import asyncio
        from src.admin.logs   import install_admin_handler
        from src.admin.events import get_event_bus

        loop    = asyncio.get_event_loop()
        handler = install_admin_handler(loop)
        get_event_bus().set_loop(loop)
        logger.info("Admin API started. daemon=%s", daemon is not None)

    return app


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from src.portfolio.database import PortfolioDatabase

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    standalone_app = create_app(daemon=None, db=PortfolioDatabase())
    uvicorn.run(standalone_app, host="0.0.0.0", port=8420, log_level="info")
