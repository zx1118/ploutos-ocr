"""
Ploutos OCR Worker
==================

A Python-based OCR service using PaddleOCR with Redis Stream for task queue
and FastAPI for optional synchronous API calls.

Components:
- Redis Client: Generic Redis operations and Stream-specific operations
- OCR Engine: PaddleOCR wrapper with text and structure recognition
- Post Processor: Rule-based correction engine
- Consumer: Redis Stream consumer with retry and DLQ handling
- API: FastAPI service for health check and sync OCR calls
"""

import os

# ============================================================================
# PaddlePaddle 3.0+ 兼容性修复
# 修复 PIR (Paddle Intermediate Representation) 与 oneDNN 的兼容性问题
# 错误信息: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]
# ============================================================================
os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")
os.environ.setdefault("FLAGS_pir_apply_inplace_pass", "false")

__version__ = "1.0.0"
__author__ = "RCoE Team"

