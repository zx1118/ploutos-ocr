"""
OCR Consumer
============

Redis Stream consumer that processes OCR tasks.
"""

import hashlib
import json
import signal
import time
from typing import Optional

from loguru import logger

from .callback import CallbackClient
from .config import settings
from .ocr.engine import OCREngine
from .processor.processor import PostProcessor
from .redis.client import RedisClient
from .redis.stream import StreamClient, StreamMessage
from .evidence.matcher import FieldMatcher
from .ocr.vertical import VerticalTextHandler, merge_vertical_fields


class OCRConsumer:
    """
    OCR task consumer that reads from Redis Stream.

    Features:
    - Consumer group with unique consumer name
    - Automatic stale message reclamation
    - Retry with exponential backoff
    - Dead letter queue for failed tasks
    - Graceful shutdown
    - OCR result caching to avoid duplicate processing
    """

    def __init__(
        self,
        stream_client: Optional[StreamClient] = None,
        ocr_engine: Optional[OCREngine] = None,
        post_processor: Optional[PostProcessor] = None,
        callback_client: Optional[CallbackClient] = None,
        field_matcher: Optional[FieldMatcher] = None,
        vertical_handler: Optional[VerticalTextHandler] = None,
        redis_client: Optional[RedisClient] = None,
    ):
        """
        Initialize consumer.

        Args:
            stream_client: Redis Stream client
            ocr_engine: OCR engine instance
            post_processor: Post processor instance
            callback_client: Callback client instance
            field_matcher: Field evidence matcher
            vertical_handler: Vertical text handler
            redis_client: Redis client for caching
        """
        self._stream = stream_client or StreamClient()
        self._ocr = ocr_engine or OCREngine()
        self._redis = redis_client or RedisClient()
        self._processor = post_processor or PostProcessor(
            redis_client=self._redis
        )
        self._callback = callback_client or CallbackClient()
        self._matcher = field_matcher or FieldMatcher()
        self._vertical = vertical_handler or VerticalTextHandler(ocr_engine=self._ocr)

        self._running = False
        self._processed_count = 0
        self._error_count = 0
        self._cache_hit_count = 0
        
        # Cache settings from config
        self._cache_enabled = settings.ocr.cache_enabled
        self._cache_ttl = settings.ocr.cache_ttl_seconds
        self._cache_use_file_hash = settings.ocr.cache_use_file_hash

        # Register signal handlers (only in main thread)
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        else:
            logger.info("Running in non-main thread, signal handlers not registered")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._running = False

    def _get_cache_key(self, file_path: str) -> str:
        """
        Generate cache key for OCR results.
        
        Uses file content hash if enabled (more accurate, handles duplicate files),
        otherwise uses file path hash (faster, but may miss duplicates).
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Cache key string
        """
        if self._cache_use_file_hash:
            # Use MD5 hash of file content for more accurate caching
            return self._get_file_content_hash(file_path)
        else:
            # Use MD5 hash of file path (faster but less accurate)
            path_hash = hashlib.md5(file_path.encode()).hexdigest()
            return f"ocr:cache:path:{path_hash}"
    
    def _get_file_content_hash(self, file_path: str) -> str:
        """
        Generate cache key based on file content hash.
        
        This ensures identical files (even with different paths) share cache.
        For large files, only hash first 64KB + file size for performance.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Cache key string
        """
        try:
            import os
            file_size = os.path.getsize(file_path)
            
            # For small files (<= 1MB), hash entire content
            # For large files, hash first 64KB + size for performance
            hash_size_threshold = 1024 * 1024  # 1MB
            
            md5 = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                if file_size <= hash_size_threshold:
                    md5.update(f.read())
                else:
                    # Hash first 64KB
                    md5.update(f.read(65536))
                    # Add file size to distinguish files with same header
                    md5.update(str(file_size).encode())
            
            content_hash = md5.hexdigest()
            return f"ocr:cache:hash:{content_hash}"
            
        except Exception as e:
            logger.warning(f"Failed to compute file hash: {e}, falling back to path hash")
            path_hash = hashlib.md5(file_path.encode()).hexdigest()
            return f"ocr:cache:path:{path_hash}"

    def _get_cached_ocr(self, file_path: str) -> Optional[list]:
        """
        Get cached OCR results if available.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of OCR blocks if cached, None otherwise
        """
        if not self._cache_enabled:
            return None
            
        try:
            cache_key = self._get_cache_key(file_path)
            cached = self._redis.client.get(cache_key)
            
            if cached:
                blocks = json.loads(cached)
                cache_type = "hash" if self._cache_use_file_hash else "path"
                logger.info(f"OCR cache hit ({cache_type}): {file_path} ({len(blocks)} blocks)")
                self._cache_hit_count += 1
                return blocks
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        
        return None

    def _set_cached_ocr(self, file_path: str, blocks: list) -> None:
        """
        Cache OCR results.
        
        Args:
            file_path: Path to the image file
            blocks: List of OCR blocks to cache
        """
        if not self._cache_enabled:
            return
            
        try:
            cache_key = self._get_cache_key(file_path)
            
            # Serialize blocks to JSON
            blocks_data = []
            for block in blocks:
                if hasattr(block, 'bbox'):
                    # Block object
                    blocks_data.append({
                        "bbox": block.bbox,
                        "text": block.text,
                        "confidence": block.confidence,
                    })
                else:
                    # Already a dict
                    blocks_data.append(block)
            
            self._redis.client.setex(
                cache_key,
                self._cache_ttl,
                json.dumps(blocks_data)
            )
            
            cache_type = "hash" if self._cache_use_file_hash else "path"
            logger.debug(f"OCR result cached ({cache_type}): {file_path} ({len(blocks_data)} blocks)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def start(self, warm_up: bool = True) -> None:
        """
        Start consuming messages.

        Args:
            warm_up: Whether to warm up OCR engine
        """
        logger.info(
            f"Starting OCR consumer: {self._stream.consumer_name} "
            f"on group {self._stream.group_name}"
        )

        # Ensure consumer group exists
        self._stream.ensure_group()

        # Warm up OCR engine
        if warm_up:
            logger.info("Warming up OCR engine...")
            self._ocr.warm_up()

        self._running = True
        logger.info("Consumer entering main loop...")

        # Main consume loop
        while self._running:
            try:
                self._consume_cycle()
            except Exception as e:
                logger.error(f"Error in consume cycle: {e}")
                time.sleep(1)  # Backoff on error

        logger.info(
            f"Consumer stopped. Processed: {self._processed_count}, "
            f"Errors: {self._error_count}, Cache hits: {self._cache_hit_count}"
        )

    def _consume_cycle(self) -> None:
        """Single consume cycle."""
        try:
            # 1. First, claim any stale pending messages
            stale_messages = self._stream.claim_stale(count=5)
            if stale_messages:
                logger.debug(f"Claimed {len(stale_messages)} stale messages")
            for msg in stale_messages:
                self._process_message(msg)

            # 2. Read new messages
            messages = self._stream.read(count=1, block=5000)

            if messages:
                logger.debug(f"Read {len(messages)} new messages")
            for msg in messages:
                self._process_message(msg)
        except Exception as e:
            logger.error(f"Exception in _consume_cycle: {e}")
            raise

    def _process_message(self, message: StreamMessage) -> None:
        """
        Process single message.

        Args:
            message: Stream message to process
        """
        task_id = message.task_id

        if not task_id:
            logger.warning(f"Message missing taskId: {message.message_id}")
            self._stream.ack(message.message_id)
            return

        logger.info(
            f"Processing task: {task_id} "
            f"(type: {message.task_type}, file: {message.file_name}, retry: {message.retry_count})"
        )

        start_time = time.time()

        try:
            # Route by task type
            if message.task_type == "EVIDENCE_ENRICH":
                self._process_evidence_enrich(message, start_time)
            else:
                self._process_ocr_task(message, start_time)

        except FileNotFoundError as e:
            # File not found - no retry, send to DLQ
            logger.error(f"File not found for task {task_id}: {e}")
            self._handle_permanent_failure(message, str(e))

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            self._handle_failure(message, str(e))

    def _process_ocr_task(self, message: StreamMessage, start_time: float) -> None:
        """Process standard OCR task (FULL_PAGE or ROI)."""
        task_id = message.task_id

        # 1. Check cache first
        cached_blocks = self._get_cached_ocr(message.file_path)
        
        if cached_blocks:
            # Use cached result
            from .ocr.result import OCRResult, OCRBlock
            
            blocks = [
                OCRBlock(
                    bbox=b["bbox"],
                    text=b["text"],
                    confidence=b["confidence"]
                )
                for b in cached_blocks
            ]
            
            # Reconstruct OCR result from cache
            ocr_result = OCRResult(blocks=blocks)
            ocr_result.overall_confidence = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0
        else:
            # 2. Perform OCR
            # For invoices, use PPStructure table recognition for better table extraction
            use_table_mode = (
                message.doc_type in ("INVOICE", "AUTO") and 
                getattr(settings.ocr, 'use_ppstructure_table', True)
            )
            
            if use_table_mode:
                logger.info(f"Using PPStructure table mode for {message.doc_type}")
                try:
                    ocr_result = self._ocr.recognize_with_table(
                        message.file_path,
                    )
                except Exception as e:
                    logger.warning(f"PPStructure table mode failed: {e}, falling back to text mode")
                    ocr_result = self._ocr.recognize(
                        message.file_path,
                        mode=message.prefer_mode,
                        doc_type=message.doc_type,
                    )
            else:
                ocr_result = self._ocr.recognize(
                    message.file_path,
                    mode=message.prefer_mode,
                    doc_type=message.doc_type,
                )
            
            # Cache the result
            self._set_cached_ocr(message.file_path, ocr_result.blocks)

        # 3. Post-process
        ocr_result = self._processor.process(
            ocr_result,
            doc_type=message.doc_type,
        )

        # 4. Handle vertical text (for Chinese invoices)
        if message.doc_type in ("INVOICE", "AUTO") and self._vertical:
            try:
                self._process_vertical_text(message, ocr_result)
            except Exception as e:
                logger.warning(f"Vertical text processing failed: {e}")

        # 5. DocLayout-YOLO layout analysis (方案 B - MinerU 级别)
        if getattr(settings.ocr, 'use_doclayout_yolo', False):
            try:
                self._analyze_with_doclayout_yolo(message, ocr_result)
            except Exception as e:
                logger.warning(f"DocLayout-YOLO analysis failed: {e}")
        
        # 6. Legacy layout analysis with PaddleStructure (if enabled and not using DocLayout-YOLO)
        if not getattr(settings.ocr, 'use_doclayout_yolo', False):
            try:
                self._analyze_layout_structure(message, ocr_result)
            except Exception as e:
                logger.warning(f"Layout analysis failed: {e}")

        # 7. Row clustering - group blocks into logical rows
        try:
            self._cluster_blocks_to_rows(ocr_result)
        except Exception as e:
            logger.warning(f"Row clustering failed: {e}")

        # 7. Calculate duration
        ocr_result.duration_ms = int((time.time() - start_time) * 1000)

        # 8. Send callback
        self._callback.send_success(task_id, ocr_result)

        # 9. ACK message
        self._stream.ack(message.message_id)

        self._processed_count += 1

        logger.info(
            f"Task completed: {task_id} "
            f"(confidence: {ocr_result.overall_confidence:.2%}, "
            f"duration: {ocr_result.duration_ms}ms, "
            f"review: {ocr_result.needs_review})"
        )

    def _process_vertical_text(self, message: StreamMessage, ocr_result) -> None:
        """
        Process vertical text regions for Chinese invoices.
        
        Args:
            message: Stream message
            ocr_result: OCR result to enhance
        """
        from PIL import Image

        # Load image
        try:
            image = Image.open(message.file_path)
        except Exception as e:
            logger.debug(f"Cannot load image for vertical processing: {e}")
            return

        # Convert blocks to dict format
        blocks = []
        for block in ocr_result.blocks:
            blocks.append({
                "bbox": block.bbox,
                "text": block.text,
                "confidence": block.confidence,
            })

        # Process vertical text
        vertical_fields = self._vertical.process_image(image, blocks)

        if vertical_fields:
            logger.info(f"Vertical text extracted: {list(vertical_fields.keys())}")

            # Merge with existing structured data
            if ocr_result.structured_data:
                original_fields = {}
                for key, value in ocr_result.structured_data.fields.items():
                    if isinstance(value, dict):
                        original_fields[key] = value
                    else:
                        original_fields[key] = {"value": value, "source": "OCR"}

                merged = merge_vertical_fields(original_fields, vertical_fields)

                # Update structured data
                for key, field_data in merged.items():
                    if isinstance(field_data, dict):
                        ocr_result.structured_data.fields[key] = field_data.get("value", "")
                        if key not in ocr_result.structured_data.field_confidences:
                            ocr_result.structured_data.field_confidences[key] = field_data.get("confidence", 0.85)

    def _analyze_layout_structure(self, message: StreamMessage, ocr_result) -> None:
        """
        Analyze document layout using PaddleStructure (if available).
        
        This provides advanced layout understanding:
        - Table structure recognition with HTML output
        - Document region segmentation (header, body, footer)
        - Reading order determination
        
        Args:
            message: Stream message with file info
            ocr_result: OCR result to enhance with layout info
        """
        from .config import settings
        
        # Check if layout analysis is enabled
        if not settings.ocr.enable_table_structure:
            logger.debug("Layout analysis disabled in config")
            return
        
        layout_engine = settings.ocr.layout_engine
        
        # Skip if using rule-based (handled by row_clustering)
        if layout_engine == "rule-based":
            return
        
        try:
            from .ocr.layout_analyzer import (
                analyze_document_layout, 
                LayoutEngine,
            )
            
            # Determine engine to use
            if layout_engine == "auto":
                engine = LayoutEngine.AUTO
            elif layout_engine == "paddle-structure":
                engine = LayoutEngine.PADDLE_STRUCTURE
            elif layout_engine == "doclayout-yolo":
                engine = LayoutEngine.DOCLAYOUT_YOLO
            else:
                engine = LayoutEngine.RULE_BASED
            
            # Run layout analysis
            layout_result = analyze_document_layout(
                image_path=message.file_path,
                image_width=ocr_result.image_width,
                image_height=ocr_result.image_height,
                engine=engine,
                use_gpu=settings.ocr.should_use_gpu,
            )
            
            # Store layout info in OCR result
            if layout_result and layout_result.regions:
                # Add layout analysis results
                layout_info = layout_result.to_dict()
                
                # Merge table HTML if available (from PaddleStructure)
                table_regions = layout_result.get_table_regions()
                if table_regions:
                    tables_html = []
                    for table in table_regions:
                        if table.table_structure:
                            tables_html.append({
                                "bbox": table.bbox,
                                "html": table.table_structure,
                                "rows": table.rows,
                                "cols": table.cols,
                                "confidence": table.confidence,
                            })
                    
                    if tables_html:
                        layout_info["tables"] = tables_html
                        logger.info(f"Extracted {len(tables_html)} table(s) with HTML structure")
                
                # Store in table_structure for frontend
                if not hasattr(ocr_result, 'layout_analysis') or ocr_result.layout_analysis is None:
                    ocr_result.layout_analysis = layout_info
                
                logger.info(
                    f"Layout analysis complete: {len(layout_result.regions)} regions, "
                    f"model={layout_result.model_used}, time={layout_result.duration_ms}ms"
                )
                
        except ImportError as e:
            logger.debug(f"Layout engine not available: {e}")
        except Exception as e:
            logger.warning(f"Layout analysis error: {e}")

    def _analyze_with_doclayout_yolo(self, message: StreamMessage, ocr_result) -> None:
        """
        Analyze document layout using DocLayout-YOLO (方案 B - MinerU 级别).
        
        This provides MinerU-level layout understanding:
        - High-accuracy table region detection
        - Multi-class document element detection (title, text, table, figure, etc.)
        - Fast inference with YOLO architecture
        
        Unlike PPStructure (方案 A), this keeps original OCR bboxes and only uses
        layout detection to guide row/column organization.
        
        Args:
            message: Stream message with file info
            ocr_result: OCR result to enhance with layout info
        """
        try:
            from .ocr.doclayout_yolo import get_doclayout_yolo_engine
            
            logger.info("Running DocLayout-YOLO layout analysis (方案 B)...")
            
            # Get or create engine
            engine = get_doclayout_yolo_engine(
                model_path=getattr(settings.ocr, 'doclayout_model_path', None),
                device="cuda" if settings.ocr.should_use_gpu else "cpu",
            )
            
            # Update confidence threshold if configured
            if hasattr(settings.ocr, 'doclayout_conf_threshold'):
                engine.conf_threshold = settings.ocr.doclayout_conf_threshold
            
            # Run layout analysis
            layout_result = engine.analyze(message.file_path)
            
            # Store layout info
            layout_info = layout_result.to_dict()
            
            # Extract table regions for enhanced processing
            table_regions = layout_result.get_table_regions()
            if table_regions:
                logger.info(f"DocLayout-YOLO detected {len(table_regions)} table region(s)")
                
                # Store table bboxes for row clustering optimization
                table_bboxes = []
                for table in table_regions:
                    table_bboxes.append({
                        "bbox": table.bbox,
                        "confidence": table.confidence,
                    })
                
                layout_info["detectedTables"] = table_bboxes
                
                # Mark blocks that are within table regions
                self._mark_blocks_in_table_regions(ocr_result, table_regions)
            
            # Store layout analysis result
            ocr_result.layout_analysis = layout_info
            
            logger.info(
                f"DocLayout-YOLO complete: {len(layout_result.regions)} regions, "
                f"tables={len(table_regions)}, time={layout_result.duration_ms}ms"
            )
            
        except FileNotFoundError as e:
            logger.warning(f"DocLayout-YOLO model not found: {e}")
            logger.warning("Download model or disable use_doclayout_yolo in config")
        except ImportError as e:
            logger.warning(f"DocLayout-YOLO dependencies not installed: {e}")
            logger.warning("Install with: pip install doclayout-yolo huggingface_hub")
        except Exception as e:
            logger.warning(f"DocLayout-YOLO analysis failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _mark_blocks_in_table_regions(self, ocr_result, table_regions) -> None:
        """
        Mark OCR blocks that are within detected table regions.
        
        This helps row clustering focus on table areas and apply
        stricter column alignment rules.
        
        Args:
            ocr_result: OCR result with blocks
            table_regions: List of detected table LayoutRegion objects
        """
        if not ocr_result.blocks or not table_regions:
            return
        
        for block in ocr_result.blocks:
            bbox = block.bbox
            if not bbox or len(bbox) < 4:
                continue
            
            # Get block center
            if isinstance(bbox[0], (list, tuple)):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                block_cx = sum(xs) / len(xs)
                block_cy = sum(ys) / len(ys)
            else:
                block_cx = (bbox[0] + bbox[2]) / 2
                block_cy = (bbox[1] + bbox[3]) / 2
            
            # Check if block center is within any table region
            for table in table_regions:
                t_bbox = table.bbox  # [x1, y1, x2, y2]
                if (t_bbox[0] <= block_cx <= t_bbox[2] and 
                    t_bbox[1] <= block_cy <= t_bbox[3]):
                    # Mark block as in table
                    if not hasattr(block, 'in_table'):
                        block.in_table = True
                    break

    def _cluster_blocks_to_rows(self, ocr_result) -> None:
        """
        Cluster OCR blocks into logical rows based on Y-coordinate.
        
        This groups scattered text blocks into rows for better table extraction,
        especially useful for invoices with many line items.
        
        Features:
        - Adaptive Y-tolerance based on line height
        - Projection-based column detection
        - Table region segmentation (header/body/total/footer)
        
        Args:
            ocr_result: OCR result to enhance with row structure
        """
        from .ocr.row_clustering import (
            cluster_ocr_blocks_to_rows, 
            extract_table_structure_advanced,
        )
        
        if not ocr_result.blocks:
            return
        
        # Convert blocks to dict format
        blocks_dict = [
            {
                "text": block.text,
                "confidence": block.confidence,
                "bbox": block.bbox,
            }
            for block in ocr_result.blocks
        ]
        
        # Cluster into rows
        rows = cluster_ocr_blocks_to_rows(blocks_dict)
        ocr_result.rows = rows
        
        # Extract table structure (with advanced column detection and segmentation)
        try:
            table_structure = extract_table_structure_advanced(
                blocks_dict,
                image_width=ocr_result.image_width,
                image_height=ocr_result.image_height,
            )
            
            # Apply template if supplier is recognized
            try:
                from .ocr.layout_analyzer import apply_template_if_available
                table_structure = apply_template_if_available(
                    ocr_result.full_text,
                    table_structure,
                )
            except Exception as e:
                logger.debug(f"Template matching skipped: {e}")
            
            ocr_result.table_structure = table_structure
            
            regions = table_structure.get("regions", {})
            logger.info(
                f"Table structure extracted: {table_structure.get('rowCount', 0)} rows, "
                f"{table_structure.get('columnCount', 0)} columns, "
                f"body={regions.get('bodyCount', 0)}, "
                f"hasTotal={regions.get('hasTotal', False)}"
            )
        except Exception as e:
            logger.warning(f"Table structure extraction failed: {e}")
            ocr_result.table_structure = None

    def _process_evidence_enrich(self, message: StreamMessage, start_time: float) -> None:
        """Process evidence enrichment task - match fields to OCR bboxes."""

        task_id = message.task_id
        doc_id = message.doc_id

        logger.info(f"Evidence enrichment: task={task_id}, doc={doc_id}")

        # 1. Parse fields to match
        if not message.fields_json:
            raise ValueError("Missing fieldsJson in EVIDENCE_ENRICH task")

        fields = json.loads(message.fields_json)
        logger.info(f"Fields to match: {list(fields.keys())}")

        # 2. Check cache first
        cached_blocks = self._get_cached_ocr(message.file_path)
        
        if cached_blocks:
            # Use cached result - convert to block objects
            from .ocr.result import OCRBlock
            blocks = [
                OCRBlock(
                    bbox=b["bbox"],
                    text=b["text"],
                    confidence=b["confidence"]
                )
                for b in cached_blocks
            ]
            logger.info(f"Using cached OCR: {len(blocks)} blocks")
        else:
            # 3. Perform OCR to get text blocks with bbox
            ocr_result = self._ocr.recognize(
                message.file_path,
                mode="text",  # Text mode is sufficient for evidence matching
                doc_type="AUTO",
            )
            blocks = ocr_result.blocks
            
            # Cache the result
            self._set_cached_ocr(message.file_path, blocks)

            logger.info(f"OCR found {len(blocks)} text blocks")

        # 4. Match fields to blocks
        match_results = self._matcher.match_all(fields, blocks)

        # 5. Build evidence map
        evidence_map = {}
        matched_count = 0

        for field_key, result in match_results.items():
            if result.matched and result.bbox:
                evidence_map[field_key] = {
                    "bbox": result.bbox,  # [x1, y1, x2, y2]
                    "textSnippet": result.text_snippet,
                    "confidence": result.confidence,
                    "matchType": result.match_type,
                }
                matched_count += 1

        duration_ms = int((time.time() - start_time) * 1000)

        # 6. Send callback
        self._callback.send_evidence_result(
            task_id=task_id,
            doc_id=doc_id,
            success=True,
            evidence_map=evidence_map,
            duration_ms=duration_ms,
        )

        # 7. ACK message
        self._stream.ack(message.message_id)

        self._processed_count += 1

        logger.info(
            f"Evidence enrichment completed: task={task_id}, "
            f"matched={matched_count}/{len(fields)}, "
            f"duration={duration_ms}ms"
        )

    def _handle_failure(self, message: StreamMessage, error: str) -> None:
        """Handle task failure with retry logic."""
        self._error_count += 1

        is_retry, new_id = self._stream.retry_or_dlq(message, error)

        if is_retry:
            logger.info(f"Task {message.task_id} scheduled for retry")
        else:
            # Sent to DLQ, notify callback
            self._callback.send_failed(message.task_id, error)
            logger.warning(f"Task {message.task_id} sent to DLQ")

    def _handle_permanent_failure(
        self,
        message: StreamMessage,
        error: str,
    ) -> None:
        """Handle permanent failure - skip retries, go to DLQ."""
        self._error_count += 1

        # Send to DLQ directly
        self._stream.send_to_dlq(message, error)
        self._stream.ack(message.message_id)

        # Notify callback
        self._callback.send_failed(message.task_id, error)

        logger.warning(f"Task {message.task_id} permanently failed: {error}")

    def stop(self) -> None:
        """Stop consumer gracefully."""
        logger.info("Stopping consumer...")
        self._running = False

    def get_stats(self) -> dict:
        """Get consumer statistics."""
        return {
            "consumerName": self._stream.consumer_name,
            "groupName": self._stream.group_name,
            "processedCount": self._processed_count,
            "errorCount": self._error_count,
            "cacheHitCount": self._cache_hit_count,
            "running": self._running,
            "streamLength": self._stream.get_stream_length(),
            "dlqLength": self._stream.get_dlq_length(),
            "pendingInfo": self._stream.get_pending_info(),
        }


def run_consumer() -> None:
    """Entry point for running consumer."""
    # Setup logging
    from .logging import setup_logging
    setup_logging()

    logger.info("=" * 60)
    logger.info("OCR Worker Starting")
    logger.info("=" * 60)

    consumer = OCRConsumer()

    try:
        consumer.start(warm_up=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        consumer.stop()
        logger.info("OCR Worker stopped")
