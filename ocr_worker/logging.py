"""
Logging Configuration
=====================

Setup logging using loguru with rotation and retention.
Supports structured logging, performance tracking, and debug contexts.
"""

import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable

from loguru import logger

from .config import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    log_settings = settings.log
    
    # Remove default handler
    logger.remove()
    
    # Console handler - human readable (simple format without extra fields)
    logger.add(
        sys.stdout,
        level=log_settings.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # File handler - regular logs
    log_dir = Path(log_settings.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / "ocr_worker_{time:YYYY-MM-DD}.log",
        level=log_settings.level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        ),
        rotation=log_settings.rotation,
        retention=log_settings.retention,
        compression="zip",
        encoding="utf-8",
    )
    
    # JSON file handler - structured logs for analysis (using serialize=True)
    logger.add(
        log_dir / "ocr_worker_json_{time:YYYY-MM-DD}.log",
        level=log_settings.level,
        rotation=log_settings.rotation,
        retention=log_settings.retention,
        compression="zip",
        encoding="utf-8",
        serialize=True,  # Use loguru's built-in JSON serialization
    )
    
    # Error file handler
    logger.add(
        log_dir / "ocr_worker_error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}\n{exception}"
        ),
        rotation=log_settings.rotation,
        retention=log_settings.retention,
        compression="zip",
        encoding="utf-8",
    )
    
    # Performance log - separate file for timing data
    logger.add(
        log_dir / "ocr_worker_perf_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        rotation=log_settings.rotation,
        retention=log_settings.retention,
        compression="zip",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("perf", False),
    )
    
    logger.info(f"Logging configured: level={log_settings.level}, dir={log_dir}")




@contextmanager
def log_context(**kwargs):
    """
    Context manager for adding extra fields to all logs within the context.
    
    Usage:
        with log_context(task_id="123", doc_id="abc"):
            logger.info("Processing task")  # Will include task_id and doc_id
    """
    with logger.contextualize(**kwargs):
        yield


@contextmanager
def log_duration(operation: str, **extra_fields):
    """
    Context manager for logging operation duration.
    
    Usage:
        with log_duration("OCR recognition", file="test.png"):
            result = ocr_engine.recognize(file)
    """
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.bind(perf=True).info(
            f"[PERF] {operation}: {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **extra_fields
        )


def log_performance(operation: str = None):
    """
    Decorator for logging function execution time.
    
    Usage:
        @log_performance("OCR recognition")
        def recognize(self, file_path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.bind(perf=True).debug(
                    f"[PERF] {op_name}: {duration_ms:.2f}ms (success)",
                    operation=op_name,
                    duration_ms=round(duration_ms, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.bind(perf=True).error(
                    f"[PERF] {op_name}: {duration_ms:.2f}ms (failed: {e})",
                    operation=op_name,
                    duration_ms=round(duration_ms, 2),
                    status="failed",
                    error=str(e),
                )
                raise
        
        return wrapper
    return decorator


def log_ocr_task(
    task_id: str,
    task_type: str,
    file_path: str,
    duration_ms: int,
    success: bool,
    **extra_fields
) -> None:
    """
    Log OCR task completion with structured data.
    
    Creates a structured log entry for task tracking and analysis.
    """
    logger.bind(
        task_id=task_id,
        task_type=task_type,
        file_path=file_path,
        duration_ms=duration_ms,
        success=success,
        **extra_fields
    ).info(
        f"Task {'completed' if success else 'failed'}: {task_id} ({duration_ms}ms)"
    )

