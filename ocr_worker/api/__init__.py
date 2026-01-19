"""
FastAPI Module
==============

Optional FastAPI service for health checks and synchronous OCR calls.
"""

from .main import app, run_api

__all__ = ["app", "run_api"]

