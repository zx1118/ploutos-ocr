"""
Main Entry Point
================

Entry points for OCR Worker and API.
"""

import argparse
import sys


def run_worker():
    """Run OCR consumer worker."""
    from .consumer import run_consumer
    run_consumer()


def run_api():
    """Run FastAPI server."""
    from .api.main import run_api as _run_api
    _run_api()


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Ploutos OCR Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Worker command
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run OCR worker (Redis Stream consumer)",
    )
    worker_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip OCR engine warm-up",
    )
    
    # API command
    api_parser = subparsers.add_parser(
        "api",
        help="Run FastAPI server",
    )
    api_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API host (default: 0.0.0.0)",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="API port (default: 8100)",
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    # Both command
    both_parser = subparsers.add_parser(
        "both",
        help="Run both worker and API",
    )
    
    args = parser.parse_args()
    
    if args.command == "worker":
        run_worker()
    elif args.command == "api":
        # Override settings from CLI args
        from .config import settings
        settings.api.host = args.host
        settings.api.port = args.port
        run_api()
    elif args.command == "both":
        # Run both in separate threads
        import threading
        
        def run_worker_with_exception_handling():
            """Worker wrapper with exception handling for daemon thread."""
            try:
                run_worker()
            except Exception as e:
                from loguru import logger
                logger.exception(f"Worker thread crashed: {e}")
        
        worker_thread = threading.Thread(
            target=run_worker_with_exception_handling, 
            daemon=True,
            name="ocr-worker"
        )
        worker_thread.start()
        
        # Give worker time to start
        import time
        time.sleep(1)
        
        run_api()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

