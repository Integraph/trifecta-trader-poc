"""
Pipeline Daemon — entry point.

Usage:
    python -m src.run_daemon
    python -m src.run_daemon --config config/automation.yaml
    python -m src.run_daemon --no-scheduler    # Queue reader only
    python -m src.run_daemon --no-queue        # Scheduler only
    python -m src.run_daemon --run-now         # Immediate watchlist scan, then exit
    python -m src.run_daemon --health          # Print health status and exit
"""

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

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
