"""
Callback Client
===============

Sends OCR results back to JeecgBoot via Redis Stream AND HTTP callback.
Ensures reliable delivery through multiple channels.
"""

import json
import time
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from .config import settings
from .ocr.result import OCRResult
from .redis.pool import get_redis_client


class CallbackClient:
    """
    Redis Stream callback client for JeecgBoot.
    
    Features:
    - Send OCR results via Redis Stream (no HTTP required)
    - Automatic retry on Redis errors
    - JSON serialization of results
    """
    
    def __init__(
        self,
        result_stream_key: Optional[str] = None,
    ):
        """
        Initialize callback client.
        
        Args:
            result_stream_key: Redis Stream key for results
        """
        self._result_key = result_stream_key or settings.stream.result_key
        self._client = get_redis_client()
        
        logger.info(f"Callback client initialized: Redis Stream -> {self._result_key}")
    
    def send_result(
        self,
        task_id: str,
        status: str,
        ocr_result: Optional[OCRResult] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Send OCR result to Redis Stream.
        
        Args:
            task_id: Task ID
            status: Result status (SUCCESS, FAILED, NEED_REVIEW)
            ocr_result: OCR result object
            error: Error message if failed
            
        Returns:
            True if message sent successfully
        """
        payload = {
            "taskId": task_id,
            "status": status,
            "timestamp": str(int(time.time() * 1000)),
        }
        
        if ocr_result:
            result_dict = ocr_result.to_dict()
            
            # Build complete OCR result with all structure data
            ocr_result_payload = {
                "text": result_dict["fullText"],
                "blocks": result_dict["blocks"],
            }
            
            # Include row clustering results (行聚类结果)
            if result_dict.get("rows"):
                ocr_result_payload["rows"] = result_dict["rows"]
            
            # Include table structure (表格结构：列、行区域)
            if result_dict.get("tableStructure"):
                ocr_result_payload["tableStructure"] = result_dict["tableStructure"]
            
            # Include PPStructure tables (方案 A: PPStructure 表格识别)
            if result_dict.get("tables"):
                ocr_result_payload["tables"] = result_dict["tables"]
                logger.info(f"Including {len(result_dict['tables'])} PPStructure tables in callback")
            
            # Include layout analysis (版式分析)
            if result_dict.get("layoutAnalysis"):
                ocr_result_payload["layoutAnalysis"] = result_dict["layoutAnalysis"]
            
            payload.update({
                "ocrResult": json.dumps(ocr_result_payload),
                "structuredData": json.dumps(result_dict["structuredData"]) if result_dict["structuredData"] else "",
                "fieldConfidences": json.dumps(
                    result_dict["structuredData"]["fieldConfidences"]
                    if result_dict["structuredData"]
                    else {}
                ),
                "overallConfidence": str(result_dict["overallConfidence"]),
                "needsReview": str(result_dict["needsReview"]).lower(),
                "lowConfidenceFields": json.dumps(result_dict["lowConfidenceFields"]),
                "appliedRules": json.dumps(result_dict["appliedRules"]),
                "durationMs": str(result_dict["durationMs"]),
                "modelVersion": result_dict["modelVersion"] or "",
                # 添加图片尺寸用于前端坐标转换
                "imageWidth": str(result_dict.get("imageWidth", 0)),
                "imageHeight": str(result_dict.get("imageHeight", 0)),
            })
        
        if error:
            payload["error"] = error
        
        try:
            # Add message to Redis Stream
            msg_id = self._client.xadd(
                self._result_key,
                payload,
                maxlen=10000,  # Keep last 10000 results
                approximate=True,
            )
            
            logger.info(f"Callback sent via Redis Stream: task={task_id}, status={status}, msgId={msg_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send callback for task {task_id}: {e}")
            # Retry once
            try:
                time.sleep(0.5)
                msg_id = self._client.xadd(self._result_key, payload)
                logger.info(f"Callback retry succeeded: task={task_id}, msgId={msg_id}")
                return True
            except Exception as e2:
                logger.error(f"Callback retry failed for task {task_id}: {e2}")
                return False
    
    def send_success(
        self,
        task_id: str,
        ocr_result: OCRResult,
    ) -> bool:
        """Send successful OCR result."""
        status = "NEED_REVIEW" if ocr_result.needs_review else "SUCCESS"
        return self.send_result(task_id, status, ocr_result)
    
    def send_failed(
        self,
        task_id: str,
        error: str,
    ) -> bool:
        """Send failure notification."""
        return self.send_result(task_id, "FAILED", error=error)
    
    def send_evidence_result(
        self,
        task_id: str,
        doc_id: str,
        success: bool,
        evidence_map: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: int = 0,
    ) -> bool:
        """
        Send evidence enrichment result via Redis Stream AND HTTP callback.
        
        Args:
            task_id: Task ID
            doc_id: Document ID
            success: Whether enrichment succeeded
            evidence_map: {field_key: {bbox: [x1,y1,x2,y2], textSnippet: "..."}}
            error: Error message if failed
            duration_ms: Processing duration
            
        Returns:
            True if sent successfully
        """
        status = "SUCCESS" if success else "FAILED"
        
        payload = {
            "taskId": task_id,
            "taskType": "EVIDENCE_ENRICH",
            "docId": doc_id,
            "status": status,
            "timestamp": str(int(time.time() * 1000)),
            "durationMs": str(duration_ms),
        }
        
        if evidence_map:
            payload["evidenceMap"] = json.dumps(evidence_map)
        
        if error:
            payload["error"] = error
        
        redis_success = False
        http_success = False
        
        # 1. Send to Redis Stream
        try:
            msg_id = self._client.xadd(
                self._result_key,
                payload,
                maxlen=10000,
                approximate=True,
            )
            
            logger.info(
                f"Evidence result sent via Redis Stream: task={task_id}, doc={doc_id}, "
                f"success={success}, fields={len(evidence_map or {})}, msgId={msg_id}"
            )
            redis_success = True
            
        except Exception as e:
            logger.error(f"Failed to send evidence result to Redis for task {task_id}: {e}")
        
        # 2. Send HTTP callback to Java backend (primary notification)
        http_success = self._send_http_callback(
            task_id=task_id,
            doc_id=doc_id,
            success=success,
            evidence_map=evidence_map,
            error=error,
            duration_ms=duration_ms,
        )
        
        return redis_success or http_success
    
    def _send_http_callback(
        self,
        task_id: str,
        doc_id: str,
        success: bool,
        evidence_map: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: int = 0,
    ) -> bool:
        """
        Send HTTP callback to Java backend.
        
        Args:
            task_id: Task ID
            doc_id: Document ID
            success: Whether enrichment succeeded
            evidence_map: Evidence mapping
            error: Error message
            duration_ms: Processing duration
            
        Returns:
            True if HTTP callback succeeded
        """
        base_url = settings.callback.base_url
        if not base_url:
            logger.warning("HTTP callback base_url not configured, skipping HTTP callback")
            return False
        
        endpoint = "/callback/success" if success else "/callback/failed"
        url = f"{base_url}/invoice/ocr{endpoint}"
        
        # Build payload matching Java DTO structure
        if success:
            payload = {
                "taskId": task_id,
                "taskType": "EVIDENCE_ENRICH",
                "docId": doc_id,
                "evidenceMap": evidence_map or {},
                "durationMs": duration_ms,
            }
        else:
            payload = {
                "taskId": task_id,
                "taskType": "EVIDENCE_ENRICH",
                "docId": doc_id,
                "errorMessage": error or "Unknown error",
            }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add auth token if configured
        if settings.callback.api_token:
            headers["X-Access-Token"] = settings.callback.api_token
        
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            
            if response.status_code == 200:
                logger.info(
                    f"HTTP callback sent: task={task_id}, doc={doc_id}, "
                    f"success={success}, url={url}"
                )
                return True
            else:
                logger.warning(
                    f"HTTP callback failed: task={task_id}, status={response.status_code}, "
                    f"response={response.text[:200]}"
                )
                return False
                
        except httpx.TimeoutException:
            logger.warning(f"HTTP callback timeout: task={task_id}, url={url}")
            return False
        except Exception as e:
            logger.warning(f"HTTP callback error: task={task_id}, error={e}")
            return False
    
    def send_progress(
        self,
        task_id: str,
        progress: int,
        message: str = "",
    ) -> bool:
        """
        Send progress update (optional).
        
        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            message: Progress message
        """
        try:
            payload = {
                "taskId": task_id,
                "status": "PROGRESS",
                "progress": str(progress),
                "message": message,
                "timestamp": str(int(time.time() * 1000)),
            }
            self._client.xadd(self._result_key, payload)
            return True
        except Exception as e:
            logger.debug(f"Progress update failed: {e}")
            return False
    
    def close(self) -> None:
        """Close client (no-op for Redis, connection managed by pool)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
