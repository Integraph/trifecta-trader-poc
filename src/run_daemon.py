"""
Pipeline Daemon — entry point.

Usage:
    python -m src.run_daemon
    python -m src.run_daemon --config config/automation.yaml
    python -m src.run_daemon --no-scheduler    # Queue reader only
    python -m src.run_daemon --no-queue        # Scheduler only
    python -m src.run_daemon --run-now         # Immediate watchlist scan, then exit
    python -m src.run_daemon --health          # Print health status and exit
    python -m src.run_daemon --api             # Start Admin API alongside daemon (default)
    python -m src.run_daemon --no-api          # Disable Admin API
    python -m src.run_daemon --api-port 8420   # Override Admin API port
"""

import argparse
import json
import logging
import sys
import threading

logger = logging.getLogger(__name__)


def _start_api_server(daemon, port: int, host: str = "0.0.0.0") -> threading.Thread:
    """Start the FastAPI Admin API in a daemon background thread.

    Returns the thread (already started).
    """
    import uvicorn
    from src.admin.app import create_app
    from src.portfolio.database import PortfolioDatabase

    db        = PortfolioDatabase()
    admin_app = create_app(daemon=daemon, db=db)

    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(admin_app,),
        kwargs={"host": host, "port": port, "log_level": "warning"},
        daemon=True,
        name="admin_api",
    )
    api_thread.start()
    logger.info("Admin API started on http://%s:%d", host, port)
    return api_thread


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trifecta Trader pipeline daemon — runs scheduler and queue reader."
    )
    parser.add_argument(
        "--config",
        default="config/automation.yaml",
        help="Path to automation config file (default: config/automation.yaml)",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable the watchlist scheduler (queue reader only)",
    )
    parser.add_argument(
        "--no-queue",
        action="store_true",
        help="Disable the queue reader (scheduler only)",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Trigger an immediate watchlist scan then exit",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Print health status and exit (daemon must already be running)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        default=True,
        help="Start the Admin API server alongside the daemon (default: true)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable the Admin API server",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8420,
        help="Admin API port (default: 8420)",
    )
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="Admin API bind host (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    # --no-api overrides --api
    api_enabled = args.api and not args.no_api

    # Basic console logging for daemon mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from src.automation.daemon import PipelineDaemon

    daemon = PipelineDaemon(config_path=args.config)

    if args.health:
        status = daemon.health_check()
        print(json.dumps(status, indent=2, default=str))
        return

    # Optionally start the API in a background thread BEFORE daemon.start()
    # so it can serve /health requests while the daemon initialises.
    if api_enabled and not args.run_now:
        _start_api_server(daemon, port=args.api_port, host=args.api_host)

    try:
        daemon.start(
            enable_scheduler=not args.no_scheduler,
            enable_queue=not args.no_queue,
            run_now=args.run_now,
        )
    except RuntimeError as e:
        logger.error("Daemon failed to start: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
        daemon.stop()


if __name__ == "__main__":
    main()
