"""
FastAPI Application
===================

Optional FastAPI service for health checks, sync OCR, and admin endpoints.
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config import settings
from ..logging import setup_logging
from ..ocr.engine import OCREngine
from ..ocr.result import OCRResult
from ..processor.processor import PostProcessor
from ..redis.client import RedisClient
from ..redis.stream import StreamClient


# Global instances
ocr_engine: Optional[OCREngine] = None
post_processor: Optional[PostProcessor] = None
stream_client: Optional[StreamClient] = None
redis_client: Optional[RedisClient] = None

# Task cancellation registry
# Maps task_id -> CancellationToken for active tasks
_active_tasks: dict = {}
_tasks_lock = None  # Will be initialized lazily


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ocr_engine, post_processor, stream_client, redis_client
    
    setup_logging()
    
    # Initialize components
    redis_client = RedisClient()
    stream_client = StreamClient()
    ocr_engine = OCREngine()
    post_processor = PostProcessor(redis_client=redis_client)
    
    # Warm up OCR engine
    ocr_engine.warm_up()
    
    yield
    
    # Cleanup
    pass


app = FastAPI(
    title="Ploutos OCR API",
    description="OCR Service API for health checks and synchronous OCR calls",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Models ====================

class HealthResponse(BaseModel):
    status: str
    version: str
    redis: bool
    ocr_loaded: bool


class OCRRequest(BaseModel):
    file_path: str
    mode: str = "auto"
    doc_type: str = "AUTO"


class OCRResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: int = 0


class BatchOCRRequest(BaseModel):
    """Batch OCR request for processing multiple files."""
    files: list[OCRRequest]
    parallel: bool = True  # Enable parallel processing
    max_workers: int = 4   # Maximum parallel workers


class BatchOCRItem(BaseModel):
    """Single item result in batch OCR."""
    file_path: str
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: int = 0


class BatchOCRResponse(BaseModel):
    """Batch OCR response."""
    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BatchOCRItem]
    total_duration_ms: int = 0


class StreamStats(BaseModel):
    stream_length: int
    dlq_length: int
    pending_count: int
    groups: list
    consumers: list


class RulesResponse(BaseModel):
    count: int
    rules: list


class MetricsResponse(BaseModel):
    """Detailed metrics for monitoring."""
    # OCR metrics
    ocr_requests_total: int = 0
    ocr_success_total: int = 0
    ocr_error_total: int = 0
    ocr_cache_hits: int = 0
    ocr_cache_misses: int = 0
    ocr_avg_duration_ms: float = 0.0
    
    # System metrics
    gpu_available: bool = False
    gpu_enabled: bool = False
    model_version: str = ""
    model_loaded: bool = False
    
    # Redis metrics
    redis_connected: bool = False
    stream_length: int = 0
    pending_count: int = 0
    dlq_length: int = 0
    
    # Cache metrics
    cache_enabled: bool = False
    cache_ttl_seconds: int = 0


# Global metrics storage
class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self):
        self.requests_total = 0
        self.success_total = 0
        self.error_total = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_duration_ms = 0
        
    def record_request(self, success: bool, duration_ms: int, cache_hit: bool = False):
        self.requests_total += 1
        self.total_duration_ms += duration_ms
        
        if success:
            self.success_total += 1
        else:
            self.error_total += 1
            
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    @property
    def avg_duration_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_duration_ms / self.requests_total


metrics_collector = MetricsCollector()


# ==================== Health Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    redis_ok = False
    
    try:
        if redis_client:
            redis_ok = redis_client.ping()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if redis_ok else "degraded",
        version="1.0.0",
        redis=redis_ok,
        ocr_loaded=ocr_engine is not None and ocr_engine._ocr_text is not None,
    )


@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    if redis_client is None or not redis_client.ping():
        raise HTTPException(status_code=503, detail="Redis not available")
    
    return {"status": "ready"}


@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get detailed metrics for monitoring (Grafana, Prometheus, etc.).
    
    Returns comprehensive metrics about OCR performance, cache, and system status.
    """
    response = MetricsResponse(
        # OCR metrics
        ocr_requests_total=metrics_collector.requests_total,
        ocr_success_total=metrics_collector.success_total,
        ocr_error_total=metrics_collector.error_total,
        ocr_cache_hits=metrics_collector.cache_hits,
        ocr_cache_misses=metrics_collector.cache_misses,
        ocr_avg_duration_ms=metrics_collector.avg_duration_ms,
    )
    
    # System metrics
    if ocr_engine:
        response.model_version = ocr_engine.model_version
        response.model_loaded = ocr_engine._ocr_text is not None
        response.gpu_enabled = ocr_engine._use_gpu
    
    # GPU detection
    try:
        from ..config import settings as cfg
        response.gpu_available = cfg.ocr._detect_gpu_available()
        response.cache_enabled = cfg.ocr.cache_enabled
        response.cache_ttl_seconds = cfg.ocr.cache_ttl_seconds
    except Exception:
        pass
    
    # Redis/Stream metrics
    if redis_client:
        try:
            response.redis_connected = redis_client.ping()
        except Exception:
            pass
    
    if stream_client:
        try:
            stream_info = stream_client.get_stream_info()
            response.stream_length = stream_info.get("length", 0)
            
            pending_info = stream_client.get_pending_info()
            response.pending_count = pending_info.get("pending_count", 0)
            
            response.dlq_length = stream_client.get_dlq_length()
        except Exception:
            pass
    
    return response


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus text format.
    
    Can be scraped by Prometheus for monitoring.
    """
    lines = [
        "# HELP ocr_requests_total Total number of OCR requests",
        "# TYPE ocr_requests_total counter",
        f"ocr_requests_total {metrics_collector.requests_total}",
        "",
        "# HELP ocr_success_total Total number of successful OCR requests",
        "# TYPE ocr_success_total counter",
        f"ocr_success_total {metrics_collector.success_total}",
        "",
        "# HELP ocr_error_total Total number of failed OCR requests",
        "# TYPE ocr_error_total counter",
        f"ocr_error_total {metrics_collector.error_total}",
        "",
        "# HELP ocr_cache_hits_total Total number of cache hits",
        "# TYPE ocr_cache_hits_total counter",
        f"ocr_cache_hits_total {metrics_collector.cache_hits}",
        "",
        "# HELP ocr_cache_misses_total Total number of cache misses",
        "# TYPE ocr_cache_misses_total counter",
        f"ocr_cache_misses_total {metrics_collector.cache_misses}",
        "",
        "# HELP ocr_duration_avg_ms Average OCR duration in milliseconds",
        "# TYPE ocr_duration_avg_ms gauge",
        f"ocr_duration_avg_ms {metrics_collector.avg_duration_ms:.2f}",
        "",
    ]
    
    # Model info
    if ocr_engine:
        gpu_val = 1 if ocr_engine._use_gpu else 0
        model_loaded_val = 1 if ocr_engine._ocr_text is not None else 0
        
        lines.extend([
            "# HELP ocr_gpu_enabled Whether GPU is enabled",
            "# TYPE ocr_gpu_enabled gauge",
            f"ocr_gpu_enabled {gpu_val}",
            "",
            "# HELP ocr_model_loaded Whether OCR model is loaded",
            "# TYPE ocr_model_loaded gauge",
            f"ocr_model_loaded {model_loaded_val}",
            "",
        ])
    
    # Redis/Stream info
    if redis_client:
        try:
            redis_ok = 1 if redis_client.ping() else 0
            lines.extend([
                "# HELP redis_connected Whether Redis is connected",
                "# TYPE redis_connected gauge",
                f"redis_connected {redis_ok}",
                "",
            ])
        except Exception:
            pass
    
    if stream_client:
        try:
            stream_info = stream_client.get_stream_info()
            pending_info = stream_client.get_pending_info()
            dlq_length = stream_client.get_dlq_length()
            
            lines.extend([
                "# HELP ocr_stream_length Number of messages in OCR task stream",
                "# TYPE ocr_stream_length gauge",
                f"ocr_stream_length {stream_info.get('length', 0)}",
                "",
                "# HELP ocr_pending_count Number of pending OCR tasks",
                "# TYPE ocr_pending_count gauge",
                f"ocr_pending_count {pending_info.get('pending_count', 0)}",
                "",
                "# HELP ocr_dlq_length Number of messages in dead letter queue",
                "# TYPE ocr_dlq_length gauge",
                f"ocr_dlq_length {dlq_length}",
                "",
            ])
        except Exception:
            pass
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines), media_type="text/plain")


# ==================== OCR Endpoints ====================

@app.post("/v1/ocr/text", response_model=OCRResponse)
async def ocr_text(request: OCRRequest):
    """
    Perform text OCR on file.
    
    Synchronous endpoint for debugging and verification.
    """
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    start_time = time.time()
    
    try:
        result = ocr_engine.recognize(
            request.file_path,
            mode="text",
            doc_type=request.doc_type,
        )
        
        # Apply post-processing
        if post_processor:
            result = post_processor.process(result, request.doc_type)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Record metrics
        metrics_collector.record_request(success=True, duration_ms=duration_ms)
        
        return OCRResponse(
            success=True,
            data=result.to_dict(),
            duration_ms=duration_ms,
        )
        
    except FileNotFoundError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_request(success=False, duration_ms=duration_ms)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        metrics_collector.record_request(success=False, duration_ms=duration_ms)
        return OCRResponse(
            success=False,
            error=str(e),
            duration_ms=duration_ms,
        )


@app.post("/v1/ocr/structure", response_model=OCRResponse)
async def ocr_structure(request: OCRRequest):
    """
    Perform structure OCR on file.
    
    Extracts tables and structured content.
    """
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    start_time = time.time()
    
    try:
        result = ocr_engine.recognize(
            request.file_path,
            mode="structure",
            doc_type=request.doc_type,
        )
        
        # Apply post-processing
        if post_processor:
            result = post_processor.process(result, request.doc_type)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return OCRResponse(
            success=True,
            data=result.to_dict(),
            duration_ms=duration_ms,
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return OCRResponse(
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )


@app.post("/v1/ocr/upload", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(...),
    mode: str = Form(default="auto"),
    doc_type: str = Form(default="AUTO"),
):
    """
    Upload file and perform OCR.
    
    For testing purposes - files are saved to temp directory.
    """
    import tempfile
    import os
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    start_time = time.time()
    
    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".png"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = ocr_engine.recognize(
            tmp_path,
            mode=mode,
            doc_type=doc_type,
        )
        
        # Apply post-processing
        if post_processor:
            result = post_processor.process(result, doc_type)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return OCRResponse(
            success=True,
            data=result.to_dict(),
            duration_ms=duration_ms,
        )
        
    except Exception as e:
        return OCRResponse(
            success=False,
            error=str(e),
            duration_ms=int((time.time() - start_time) * 1000),
        )
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ==================== Batch OCR Endpoints ====================

@app.post("/v1/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(request: BatchOCRRequest):
    """
    Perform batch OCR on multiple files.
    
    Supports parallel processing for improved throughput.
    Files are processed independently and results are aggregated.
    
    Args:
        request: BatchOCRRequest with list of files and parallel settings
        
    Returns:
        BatchOCRResponse with individual results for each file
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    if not request.files:
        return BatchOCRResponse(
            success=True,
            total=0,
            succeeded=0,
            failed=0,
            results=[],
            total_duration_ms=0,
        )
    
    start_time = time.time()
    results: list[BatchOCRItem] = []
    
    def process_single(req: OCRRequest) -> BatchOCRItem:
        """Process a single OCR request."""
        item_start = time.time()
        try:
            result = ocr_engine.recognize(
                req.file_path,
                mode=req.mode,
                doc_type=req.doc_type,
            )
            
            # Apply post-processing
            if post_processor:
                result = post_processor.process(result, req.doc_type)
            
            duration_ms = int((time.time() - item_start) * 1000)
            metrics_collector.record_request(success=True, duration_ms=duration_ms)
            
            return BatchOCRItem(
                file_path=req.file_path,
                success=True,
                data=result.to_dict(),
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.time() - item_start) * 1000)
            metrics_collector.record_request(success=False, duration_ms=duration_ms)
            
            return BatchOCRItem(
                file_path=req.file_path,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
    
    if request.parallel and len(request.files) > 1:
        # Parallel processing using ThreadPoolExecutor
        max_workers = min(request.max_workers, len(request.files))
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [
                loop.run_in_executor(executor, process_single, req)
                for req in request.files
            ]
            # Wait for all to complete
            results = await asyncio.gather(*futures)
    else:
        # Sequential processing
        for req in request.files:
            results.append(process_single(req))
    
    # Aggregate results
    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    total_duration_ms = int((time.time() - start_time) * 1000)
    
    return BatchOCRResponse(
        success=failed == 0,
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        results=results,
        total_duration_ms=total_duration_ms,
    )


@app.post("/v1/ocr/batch/async")
async def ocr_batch_async(request: BatchOCRRequest):
    """
    Submit batch OCR tasks asynchronously via Redis Stream.
    
    Tasks are queued for processing by the consumer worker.
    Use this for large batches or when immediate response is not required.
    
    Args:
        request: BatchOCRRequest with list of files
        
    Returns:
        Dict with task IDs for tracking
    """
    import uuid
    
    if stream_client is None:
        raise HTTPException(status_code=503, detail="Stream client not initialized")
    
    if not request.files:
        return {"success": True, "tasks": [], "message": "No files to process"}
    
    task_ids = []
    batch_id = str(uuid.uuid4())
    
    for idx, req in enumerate(request.files):
        task_id = f"{batch_id}:{idx}"
        
        # Submit to Redis Stream
        message_id = stream_client.submit_task(
            task_id=task_id,
            task_type="EVIDENCE_ENRICH",  # Or appropriate task type
            payload={
                "file_path": req.file_path,
                "mode": req.mode,
                "doc_type": req.doc_type,
                "batch_id": batch_id,
                "batch_index": idx,
                "batch_total": len(request.files),
            },
        )
        
        task_ids.append({
            "task_id": task_id,
            "message_id": message_id,
            "file_path": req.file_path,
        })
    
    return {
        "success": True,
        "batch_id": batch_id,
        "total": len(task_ids),
        "tasks": task_ids,
        "message": f"Submitted {len(task_ids)} tasks for async processing",
    }


# ==================== Admin Endpoints ====================

@app.get("/admin/stats", response_model=StreamStats)
async def get_stream_stats():
    """Get Redis Stream statistics."""
    if stream_client is None:
        raise HTTPException(status_code=503, detail="Stream client not initialized")
    
    try:
        stream_info = stream_client.get_stream_info()
        groups_info = stream_client.get_groups_info()
        consumers_info = stream_client.get_consumers_info()
        pending_info = stream_client.get_pending_info()
        
        return StreamStats(
            stream_length=stream_info.get("length", 0),
            dlq_length=stream_client.get_dlq_length(),
            pending_count=pending_info.get("pending_count", 0),
            groups=groups_info,
            consumers=consumers_info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/dlq")
async def get_dlq_messages(count: int = 100):
    """Get messages from dead letter queue."""
    if stream_client is None:
        raise HTTPException(status_code=503, detail="Stream client not initialized")
    
    messages = stream_client.get_dlq_messages(count)
    
    return {
        "count": len(messages),
        "messages": [
            {
                "messageId": m.message_id,
                "taskId": m.task_id,
                "data": m.data,
            }
            for m in messages
        ],
    }


@app.post("/admin/dlq/reprocess")
async def reprocess_dlq_messages(count: int = 10, reset_retry: bool = True):
    """
    Reprocess messages from dead letter queue.
    
    Moves messages from DLQ back to main stream for reprocessing.
    Useful for manually retrying failed tasks after fixing issues.
    
    Args:
        count: Number of messages to reprocess (default 10)
        reset_retry: Reset retry count to 0 (default True)
    """
    if stream_client is None:
        raise HTTPException(status_code=503, detail="Stream client not initialized")
    
    try:
        results = stream_client.reprocess_dlq(count=count, reset_retry=reset_retry)
        
        return {
            "success": True,
            "count": len(results),
            "reprocessed": [
                {"dlq_id": dlq_id, "new_id": new_id}
                for dlq_id, new_id in results
            ],
            "message": f"Reprocessed {len(results)} messages from DLQ",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/dlq")
async def clear_dlq():
    """
    Clear all messages from dead letter queue.
    
    WARNING: This permanently deletes all DLQ messages!
    """
    if stream_client is None:
        raise HTTPException(status_code=503, detail="Stream client not initialized")
    
    try:
        deleted = stream_client.clear_dlq()
        
        return {
            "success": True,
            "deleted": deleted,
            "message": f"Cleared {deleted} messages from DLQ",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/tasks/active")
async def get_active_tasks():
    """
    Get list of currently active (in-progress) OCR tasks.
    
    Returns task IDs that can be cancelled.
    """
    return {
        "count": len(_active_tasks),
        "tasks": list(_active_tasks.keys()),
    }


@app.post("/admin/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    Cancel an active OCR task.
    
    Sends cancellation signal to the task. The task will stop at the next
    cancellation check point (typically between pages for PDFs).
    
    Args:
        task_id: ID of the task to cancel
    """
    import threading
    
    global _tasks_lock
    if _tasks_lock is None:
        _tasks_lock = threading.Lock()
    
    with _tasks_lock:
        if task_id not in _active_tasks:
            raise HTTPException(
                status_code=404, 
                detail=f"Task {task_id} not found or already completed"
            )
        
        token = _active_tasks[task_id]
        token.cancel()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"Cancellation signal sent to task {task_id}",
        }


def register_active_task(task_id: str, token) -> None:
    """Register a task as active (internal use)."""
    import threading
    
    global _tasks_lock
    if _tasks_lock is None:
        _tasks_lock = threading.Lock()
    
    with _tasks_lock:
        _active_tasks[task_id] = token


def unregister_active_task(task_id: str) -> None:
    """Unregister a task when completed (internal use)."""
    import threading
    
    global _tasks_lock
    if _tasks_lock is None:
        _tasks_lock = threading.Lock()
    
    with _tasks_lock:
        _active_tasks.pop(task_id, None)


@app.get("/admin/rules", response_model=RulesResponse)
async def get_rules():
    """Get all correction rules."""
    if post_processor is None:
        raise HTTPException(status_code=503, detail="Post processor not initialized")
    
    rules = post_processor.get_rules()
    
    return RulesResponse(
        count=len(rules),
        rules=[r.to_dict() for r in rules],
    )


@app.post("/admin/rules/refresh")
async def refresh_rules():
    """Refresh rules from API."""
    if post_processor is None:
        raise HTTPException(status_code=503, detail="Post processor not initialized")
    
    count = post_processor.refresh_rules()
    
    return {"message": "Rules refreshed", "count": count}


@app.post("/admin/model/reload")
async def reload_model(version: Optional[str] = None):
    """Reload OCR model."""
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    ocr_engine.reload_model(version=version)
    
    return {"message": "Model reload scheduled", "version": version}


@app.post("/admin/config/reload")
async def reload_config():
    """
    Reload configuration from environment variables.
    
    Use this to apply configuration changes without restarting the service.
    """
    from ..config import reload_settings
    
    try:
        old_settings = {
            "ocr.pdf_dpi": settings.ocr.pdf_dpi,
            "ocr.cache_enabled": settings.ocr.cache_enabled,
            "ocr.cache_ttl_seconds": settings.ocr.cache_ttl_seconds,
            "ocr.default_timeout": settings.ocr.default_timeout,
        }
        
        new_settings = reload_settings()
        
        new_values = {
            "ocr.pdf_dpi": new_settings.ocr.pdf_dpi,
            "ocr.cache_enabled": new_settings.ocr.cache_enabled,
            "ocr.cache_ttl_seconds": new_settings.ocr.cache_ttl_seconds,
            "ocr.default_timeout": new_settings.ocr.default_timeout,
        }
        
        changes = {
            k: {"old": old_settings[k], "new": new_values[k]}
            for k in old_settings
            if old_settings[k] != new_values[k]
        }
        
        return {
            "success": True,
            "changes": changes,
            "message": f"Configuration reloaded, {len(changes)} settings changed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/config")
async def get_config():
    """
    Get current configuration values.
    
    Returns non-sensitive configuration for inspection.
    """
    return {
        "ocr": {
            "pdf_dpi": settings.ocr.pdf_dpi,
            "use_gpu": settings.ocr.use_gpu,
            "auto_detect_gpu": settings.ocr.auto_detect_gpu,
            "default_lang": settings.ocr.default_lang,
            "confidence_threshold": settings.ocr.confidence_threshold,
            "cache_enabled": settings.ocr.cache_enabled,
            "cache_ttl_seconds": settings.ocr.cache_ttl_seconds,
            "default_timeout": settings.ocr.default_timeout,
        },
        "stream": {
            "stream_key": settings.stream.stream_key,
            "consumer_group": settings.stream.consumer_group,
            "max_retry": settings.stream.max_retry,
            "batch_size": settings.stream.batch_size,
        },
        "api": {
            "host": settings.api.host,
            "port": settings.api.port,
            "workers": settings.api.workers,
            "debug": settings.api.debug,
        },
        "log": {
            "level": settings.log.level,
            "dir": settings.log.dir,
        },
    }


@app.get("/admin/model/info")
async def get_model_info():
    """Get OCR model information."""
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    return ocr_engine.get_info()


def run_api():
    """Run FastAPI server."""
    import uvicorn
    
    api_settings = settings.api
    
    uvicorn.run(
        "ocr_worker.api.main:app",
        host=api_settings.host,
        port=api_settings.port,
        workers=api_settings.workers,
        reload=api_settings.debug,
    )


if __name__ == "__main__":
    run_api()

