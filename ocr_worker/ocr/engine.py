"""
PaddleOCR Engine Wrapper
========================

Encapsulates PaddleOCR with support for text and structure recognition,
multiple languages, model hot-reloading, and timeout/cancellation.
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from PIL import Image

from ..config import settings
from .result import OCRBlock, OCRResult, StructuredData


class OCRTimeoutError(Exception):
    """Raised when OCR operation times out."""
    pass


class OCRCancelledError(Exception):
    """Raised when OCR operation is cancelled."""
    pass


class CancellationToken:
    """
    Token for cooperative cancellation of long-running operations.
    
    Usage:
        token = CancellationToken()
        
        # In worker thread:
        if token.is_cancelled:
            raise OCRCancelledError("Operation cancelled")
        
        # To cancel:
        token.cancel()
    """
    
    def __init__(self):
        self._cancelled = threading.Event()
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()
    
    def cancel(self) -> None:
        self._cancelled.set()
    
    def check(self) -> None:
        """Check if cancelled and raise if so."""
        if self.is_cancelled:
            raise OCRCancelledError("Operation cancelled")


class OCREngine:
    """
    PaddleOCR engine wrapper with lazy initialization and model management.
    
    Features:
    - Lazy model loading (only load when first used)
    - Support for text and structure recognition
    - Multi-language support
    - Model hot-reloading
    - PDF support (via pdf2image)
    """
    
    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        lang: Optional[str] = None,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ):
        """
        Initialize OCR engine (lazy loading).
        
        Args:
            use_gpu: Use GPU for inference
            lang: Language code (ch, en, etc.)
            det_model_dir: Custom detection model directory
            rec_model_dir: Custom recognition model directory
            cls_model_dir: Custom classification model directory
        """
        ocr_settings = settings.ocr
        
        # GPU detection: use explicit setting or auto-detect
        if use_gpu is not None:
            self._use_gpu = use_gpu
        else:
            self._use_gpu = ocr_settings.should_use_gpu
            
        self._lang = lang or ocr_settings.default_lang
        self._det_model_dir = det_model_dir or ocr_settings.det_model_dir
        self._rec_model_dir = rec_model_dir or ocr_settings.rec_model_dir
        self._cls_model_dir = cls_model_dir or ocr_settings.cls_model_dir
        self._use_angle_cls = ocr_settings.use_angle_cls
        
        # Lazy-loaded OCR instances
        self._ocr_text: Any = None
        self._ocr_structure: Any = None
        
        # Model version tracking
        self._model_version = "PP-OCRv4"
        
        # GPU status logging
        gpu_status = "enabled" if self._use_gpu else "disabled"
        if ocr_settings.auto_detect_gpu and not use_gpu:
            gpu_status += " (auto-detected)"
        
        logger.info(
            f"OCR Engine initialized: lang={self._lang}, gpu={gpu_status}"
        )
    
    @property
    def model_version(self) -> str:
        """Get current model version."""
        return self._model_version
    
    def _find_poppler_path(self) -> Optional[str]:
        """Find poppler bin path on Windows."""
        import platform
        import shutil
        
        # Only needed on Windows
        if platform.system() != "Windows":
            return None
        
        # Check if pdftoppm is already in PATH
        if shutil.which("pdftoppm"):
            return None
        
        # Common installation paths on Windows
        possible_paths = [
            r"C:\poppler\poppler-24.08.0\Library\bin",
            r"C:\poppler\Library\bin",
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin",
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "pdftoppm.exe")):
                logger.info(f"Found poppler at: {path}")
                return path
        
        logger.warning("Poppler not found in common paths, relying on PATH")
        return None
    
    def _get_text_ocr(self):
        """Get or create text OCR instance (lazy loading)."""
        if self._ocr_text is None:
            from paddleocr import PaddleOCR
            
            logger.info("Loading PaddleOCR text model...")
            start_time = time.time()
            
            # PaddleOCR 3.x uses simplified parameters
            kwargs = {
                "use_angle_cls": self._use_angle_cls,
                "lang": self._lang,
                # 禁用文档预处理，避免裁剪/变形导致坐标偏移
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            }
            
            # GPU setting - only add if explicitly enabled
            if self._use_gpu:
                kwargs["device"] = "gpu"
            
            # Add custom model paths if specified
            if self._det_model_dir:
                kwargs["det_model_dir"] = self._det_model_dir
            if self._rec_model_dir:
                kwargs["rec_model_dir"] = self._rec_model_dir
            if self._cls_model_dir:
                kwargs["cls_model_dir"] = self._cls_model_dir
            
            self._ocr_text = PaddleOCR(**kwargs)
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"PaddleOCR text model loaded in {load_time:.0f}ms")
        
        return self._ocr_text
    
    def _get_structure_ocr(self):
        """Get or create structure OCR instance (lazy loading)."""
        if self._ocr_structure is None:
            try:
                from paddleocr import PPStructure
                
                logger.info("Loading PPStructure model...")
                start_time = time.time()
                
                kwargs = {
                    "lang": self._lang,
                }
                if self._use_gpu:
                    kwargs["device"] = "gpu"
                
                self._ocr_structure = PPStructure(**kwargs)
                
                load_time = (time.time() - start_time) * 1000
                logger.info(f"PPStructure model loaded in {load_time:.0f}ms")
            except ImportError:
                logger.warning("PPStructure not available, falling back to text OCR")
                self._ocr_structure = self._get_text_ocr()
        
        return self._ocr_structure
    
    def recognize(
        self,
        image_path: Union[str, Path],
        mode: str = "auto",
        doc_type: str = "AUTO",
        timeout: Optional[float] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> OCRResult:
        """
        Recognize text from image or PDF.
        
        Args:
            image_path: Path to image or PDF file
            mode: Recognition mode (auto, text, structure)
            doc_type: Document type hint for structure extraction
            timeout: Maximum time in seconds for OCR operation (None = no limit)
            cancellation_token: Token for cooperative cancellation
            
        Returns:
            OCRResult with recognized text and metadata
            
        Raises:
            OCRTimeoutError: If operation exceeds timeout
            OCRCancelledError: If operation is cancelled
            FileNotFoundError: If image file doesn't exist
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Check cancellation before starting
        if cancellation_token and cancellation_token.is_cancelled:
            raise OCRCancelledError("Operation cancelled before start")
        
        logger.info(f"OCR recognize: {image_path.name}, mode={mode}, timeout={timeout}s")
        start_time = time.time()
        
        # Default timeout from config if not specified
        if timeout is None:
            timeout = getattr(settings.ocr, 'default_timeout', 300)  # 5 minutes default
        
        # Execute with timeout if specified
        if timeout and timeout > 0:
            result = self._recognize_with_timeout(
                image_path, mode, doc_type, timeout, cancellation_token
            )
        else:
            result = self._recognize_internal(image_path, mode, doc_type, cancellation_token)
        
        # Set timing info
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.model_version = self._model_version
        
        # Calculate review flag
        result.calculate_needs_review(settings.ocr.confidence_threshold)
        
        logger.info(
            f"OCR complete: {len(result.blocks)} blocks, "
            f"confidence={result.overall_confidence:.2%}, "
            f"duration={result.duration_ms}ms"
        )
        
        return result
    
    def _recognize_with_timeout(
        self,
        image_path: Path,
        mode: str,
        doc_type: str,
        timeout: float,
        cancellation_token: Optional[CancellationToken],
    ) -> OCRResult:
        """Execute recognition with timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._recognize_internal, image_path, mode, doc_type, cancellation_token
            )
            
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.error(f"OCR timeout after {timeout}s: {image_path.name}")
                raise OCRTimeoutError(f"OCR operation timed out after {timeout} seconds")
    
    def _recognize_internal(
        self,
        image_path: Path,
        mode: str,
        doc_type: str,
        cancellation_token: Optional[CancellationToken],
    ) -> OCRResult:
        """Internal recognition logic."""
        # Handle PDF files
        if image_path.suffix.lower() == ".pdf":
            return self._recognize_pdf(image_path, mode, doc_type, cancellation_token)
        
        # Check cancellation
        if cancellation_token:
            cancellation_token.check()
        
        # Get image size
        try:
            with Image.open(image_path) as img:
                image_size = img.size
        except Exception as e:
            logger.warning(f"Failed to get image size: {e}")
            image_size = (0, 0)
        
        # Perform OCR
        if mode == "structure":
            result = self._recognize_structure(str(image_path), doc_type)
        else:
            result = self._recognize_text(str(image_path))
        
        # Set image info
        result.image_width = image_size[0]
        result.image_height = image_size[1]
        
        return result
    
    def _recognize_text(self, image_path: str) -> OCRResult:
        """Perform text recognition."""
        ocr = self._get_text_ocr()
        
        try:
            # PaddleOCR 3.x / PaddleX uses predict() method
            results = ocr.predict(image_path)
            
            # 调试：打印完整的结果结构
            if isinstance(results, list) and results:
                first_result = results[0]
                if isinstance(first_result, dict):
                    logger.info(f"PaddleX result keys: {list(first_result.keys())}")
                    # 查找可能包含OCR结果的键
                    for key in first_result.keys():
                        val = first_result[key]
                        if key not in ['input_img', 'rot_img'] and val is not None:
                            val_str = str(val)[:200] if not isinstance(val, dict) else str(list(val.keys()))
                            logger.info(f"  {key}: {val_str}")
            
            # 检查文档预处理的旋转角度
            if isinstance(results, list) and results:
                first_result = results[0]
                if isinstance(first_result, dict):
                    doc_preproc = first_result.get("doc_preprocessor_res", {})
                    if isinstance(doc_preproc, dict):
                        angle = doc_preproc.get("angle", 0)
                        logger.info(f"Document preprocessor angle: {angle}")
            
            ocr_result = self._parse_paddlex_result(results)
            logger.info(f"Parsed OCR result: {len(ocr_result.blocks)} blocks, text length: {len(ocr_result.full_text)}")
            
            return ocr_result
        except Exception as e:
            logger.error(f"Text OCR failed: {e}")
            raise
    
    def _parse_paddlex_result(self, results) -> OCRResult:
        """Parse PaddleX OCR result format."""
        import numpy as np
        from .result import OCRBlock
        
        blocks = []
        all_text_parts = []
        total_confidence = 0.0
        
        if not results:
            return OCRResult()
        
        # PaddleX 返回的是一个列表，每个元素对应一张图片
        for page_result in results:
            if not isinstance(page_result, dict):
                continue
            
            # 获取 OCR 结果字段
            rec_texts = page_result.get("rec_texts", [])
            rec_scores = page_result.get("rec_scores", [])
            dt_polys = page_result.get("dt_polys", page_result.get("rec_polys", []))
            
            # 确保是列表格式
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            if isinstance(rec_scores, (int, float)):
                rec_scores = [rec_scores]
            
            # 转换为 Python 列表（处理 numpy 数组）
            if hasattr(rec_texts, 'tolist'):
                rec_texts = rec_texts.tolist()
            if hasattr(rec_scores, 'tolist'):
                rec_scores = rec_scores.tolist()
            
            logger.debug(f"Found {len(rec_texts)} texts, {len(rec_scores)} scores")
            
            for idx in range(len(rec_texts)):
                text = rec_texts[idx] if idx < len(rec_texts) else ""
                
                # 跳过空文本
                if text is None or (isinstance(text, str) and not text.strip()):
                    continue
                
                conf = rec_scores[idx] if idx < len(rec_scores) else 0.0
                
                # 处理 bbox（可能是 numpy 数组）
                bbox = [[0, 0]] * 4
                if idx < len(dt_polys):
                    poly = dt_polys[idx]
                    # 转换 numpy 数组为 Python 列表
                    if hasattr(poly, 'tolist'):
                        poly = poly.tolist()
                    
                    # 确保格式正确 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if poly and len(poly) >= 4:
                        if isinstance(poly[0], (list, tuple)) and len(poly[0]) >= 2:
                            bbox = [[float(p[0]), float(p[1])] for p in poly[:4]]
                        elif isinstance(poly[0], (int, float)):
                            # 扁平格式 [x1,y1,x2,y2,...]
                            bbox = [[float(poly[i]), float(poly[i+1])] for i in range(0, min(8, len(poly)), 2)]
                
                # 确保 conf 是 Python float
                if hasattr(conf, 'item'):
                    conf = conf.item()
                
                block = OCRBlock(
                    text=str(text),
                    confidence=float(conf) if conf else 0.0,
                    bbox=bbox,
                    line_index=idx,
                )
                blocks.append(block)
                all_text_parts.append(block.text)
                total_confidence += block.confidence
        
        overall_conf = total_confidence / len(blocks) if blocks else 0.0
        
        return OCRResult(
            full_text="\n".join(all_text_parts),
            blocks=blocks,
            overall_confidence=overall_conf,
        )
    
    def _recognize_structure(self, image_path: str, doc_type: str) -> OCRResult:
        """Perform structure recognition."""
        ocr = self._get_structure_ocr()
        
        try:
            results = ocr(image_path)
            
            # Convert structure results to OCRResult
            blocks = []
            all_text_parts = []
            total_confidence = 0.0
            
            for idx, item in enumerate(results):
                if item.get("type") == "text":
                    res = item.get("res", [])
                    for line_idx, line in enumerate(res):
                        if isinstance(line, dict):
                            text = line.get("text", "")
                            conf = line.get("confidence", 0.0)
                            bbox = line.get("text_region", [[0, 0]] * 4)
                        else:
                            # Handle tuple format
                            bbox, (text, conf) = line
                        
                        block = OCRBlock(
                            text=text,
                            confidence=conf,
                            bbox=bbox,
                            line_index=line_idx,
                        )
                        blocks.append(block)
                        all_text_parts.append(text)
                        total_confidence += conf
                elif item.get("type") == "table":
                    # Extract text from tables
                    html = item.get("res", {}).get("html", "")
                    if html:
                        all_text_parts.append(f"[TABLE]\n{html}\n[/TABLE]")
            
            overall_conf = total_confidence / len(blocks) if blocks else 0.0
            
            return OCRResult(
                full_text="\n".join(all_text_parts),
                blocks=blocks,
                overall_confidence=overall_conf,
            )
            
        except Exception as e:
            logger.error(f"Structure OCR failed: {e}")
            # Fallback to text OCR
            logger.info("Falling back to text OCR")
            return self._recognize_text(image_path)
    
    def _recognize_pdf(
        self,
        pdf_path: Path,
        mode: str,
        doc_type: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> OCRResult:
        """Recognize text from PDF file."""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image is required for PDF support")
        
        logger.info(f"Converting PDF to images: {pdf_path.name}")
        
        # Find poppler path (Windows specific)
        poppler_path = self._find_poppler_path()
        
        # Convert PDF to images - use configurable DPI (lower = faster)
        pdf_dpi = settings.ocr.pdf_dpi
        convert_kwargs = {"dpi": pdf_dpi}
        if poppler_path:
            convert_kwargs["poppler_path"] = poppler_path
        
        logger.info(f"Converting PDF with DPI={pdf_dpi}")
            
        images = convert_from_path(str(pdf_path), **convert_kwargs)
        logger.info(f"PDF has {len(images)} page(s)")
        
        # Process each page
        all_blocks = []
        all_text_parts = []
        total_confidence = 0.0
        total_blocks = 0
        page_dimensions = []
        
        import tempfile
        
        for page_idx, image in enumerate(images):
            page_no = page_idx + 1
            
            # Check cancellation between pages
            if cancellation_token:
                cancellation_token.check()
            
            # Record page dimensions
            page_dimensions.append({
                "page": page_no,
                "width": image.width,
                "height": image.height,
            })
            
            # Save to temp file - use delete=False and manual cleanup for Windows compatibility
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    image.save(tmp_path, "PNG")
                
                if mode == "structure":
                    page_result = self._recognize_structure(tmp_path, doc_type)
                else:
                    page_result = self._recognize_text(tmp_path)
                
                # Merge results with page number
                for block in page_result.blocks:
                    block.line_index += total_blocks
                    block.page = page_no
                    all_blocks.append(block)
                
                all_text_parts.append(f"--- Page {page_no} ---")
                all_text_parts.append(page_result.full_text)
                
                total_confidence += page_result.overall_confidence * len(page_result.blocks)
                total_blocks += len(page_result.blocks)
                
                logger.info(f"Processed page {page_no}/{len(images)}: {len(page_result.blocks)} blocks, size={image.width}x{image.height}")
                
            finally:
                # Clean up temp file - handle Windows file locking
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        logger.debug(f"Could not delete temp file: {tmp_path}")
        
        # Calculate overall confidence
        overall_conf = total_confidence / total_blocks if total_blocks > 0 else 0.0
        
        # Use first page dimensions as the global default
        img_width = images[0].width if images else 0
        img_height = images[0].height if images else 0
        logger.info(f"PDF OCR complete: {len(images)} pages, {total_blocks} blocks, first page: {img_width}x{img_height}")
        
        return OCRResult(
            full_text="\n".join(all_text_parts),
            blocks=all_blocks,
            overall_confidence=overall_conf,
            page_count=len(images),
            image_width=img_width,
            image_height=img_height,
            page_dimensions=page_dimensions,
        )
    
    def reload_model(
        self,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """
        Reload OCR models (hot reload support).
        
        Args:
            det_model_dir: New detection model directory
            rec_model_dir: New recognition model directory
            cls_model_dir: New classification model directory
            version: New model version string
        """
        logger.info("Reloading OCR models...")
        
        # Update model paths
        if det_model_dir:
            self._det_model_dir = det_model_dir
        if rec_model_dir:
            self._rec_model_dir = rec_model_dir
        if cls_model_dir:
            self._cls_model_dir = cls_model_dir
        if version:
            self._model_version = version
        
        # Clear cached instances
        self._ocr_text = None
        self._ocr_structure = None
        
        logger.info(f"OCR models will be reloaded on next use (version: {self._model_version})")
    
    def warm_up(self) -> None:
        """Pre-load models (warm up)."""
        logger.info("Warming up OCR engine...")
        
        # Load text OCR
        self._get_text_ocr()
        
        # Optionally load structure OCR
        # self._get_structure_ocr()
        
        logger.info("OCR engine warmed up")
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return {
            "modelVersion": self._model_version,
            "language": self._lang,
            "useGpu": self._use_gpu,
            "useAngleCls": self._use_angle_cls,
            "textModelLoaded": self._ocr_text is not None,
            "structureModelLoaded": self._ocr_structure is not None,
            "tableModelLoaded": self._ocr_table is not None,
        }
    
    # ============================================
    # PPStructure Table Recognition (方案 A)
    # ============================================
    
    def _get_table_ocr(self):
        """Get or create table OCR instance using PPStructure with table=True."""
        if not hasattr(self, '_ocr_table') or self._ocr_table is None:
            try:
                from paddleocr import PPStructure
                
                logger.info("Loading PPStructure table model...")
                start_time = time.time()
                
                kwargs = {
                    "table": True,      # Enable table recognition
                    "ocr": True,        # Also perform OCR
                    "show_log": False,
                    "lang": self._lang,
                }
                if self._use_gpu:
                    kwargs["device"] = "gpu"
                
                self._ocr_table = PPStructure(**kwargs)
                
                load_time = (time.time() - start_time) * 1000
                logger.info(f"PPStructure table model loaded in {load_time:.0f}ms")
            except Exception as e:
                logger.error(f"Failed to load PPStructure table model: {e}")
                self._ocr_table = None
                raise
        
        return self._ocr_table
    
    def recognize_with_table(
        self,
        image_path: Union[str, Path],
        timeout: Optional[float] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> OCRResult:
        """
        Recognize image using PPStructure for accurate table extraction.
        
        This method is specifically designed for invoices and documents with tables.
        It uses PPStructure to detect table regions and extract structured cell data.
        
        Args:
            image_path: Path to image file
            timeout: Maximum time in seconds
            cancellation_token: Token for cancellation
            
        Returns:
            OCRResult with parsed table structure
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"OCR recognize_with_table: {image_path.name}")
        start_time = time.time()
        
        # Get image size
        try:
            with Image.open(image_path) as img:
                image_size = img.size
        except Exception as e:
            logger.warning(f"Failed to get image size: {e}")
            image_size = (0, 0)
        
        try:
            table_ocr = self._get_table_ocr()
            results = table_ocr(str(image_path))
            
            # Parse PPStructure results
            result = self._parse_ppstructure_table_result(results, image_size)
            
        except Exception as e:
            logger.error(f"PPStructure table recognition failed: {e}")
            logger.info("Falling back to text OCR")
            result = self._recognize_text(str(image_path))
        
        # Set timing and metadata
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.model_version = f"{self._model_version}-PPStructure"
        result.image_width = image_size[0]
        result.image_height = image_size[1]
        result.calculate_needs_review(settings.ocr.confidence_threshold)
        
        logger.info(
            f"PPStructure complete: {len(result.blocks)} blocks, "
            f"tables={len(result.tables)}, "
            f"duration={result.duration_ms}ms"
        )
        
        return result
    
    def _parse_ppstructure_table_result(
        self, 
        results: List[Dict], 
        image_size: Tuple[int, int]
    ) -> OCRResult:
        """
        Parse PPStructure results into OCRResult with structured tables.
        
        PPStructure returns a list of detected regions:
        - type: "text" or "table"
        - bbox: Region bounding box
        - res: For tables, contains "html" key with HTML table string
        """
        from .result import OCRBlock
        
        blocks = []
        all_text_parts = []
        total_confidence = 0.0
        tables = []  # Parsed table structures
        
        for region_idx, region in enumerate(results):
            region_type = region.get("type", "")
            region_bbox = region.get("bbox", [0, 0, 0, 0])
            
            if region_type == "text":
                # Handle text regions
                res = region.get("res", [])
                for line_idx, line in enumerate(res):
                    if isinstance(line, dict):
                        text = line.get("text", "")
                        conf = float(line.get("confidence", 0.0))
                        bbox = line.get("text_region", [[0, 0]] * 4)
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox, text_conf = line
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            text, conf = text_conf[0], float(text_conf[1])
                        else:
                            text, conf = str(text_conf), 0.0
                    else:
                        continue
                    
                    block = OCRBlock(
                        text=text,
                        confidence=conf,
                        bbox=bbox,
                        line_index=len(blocks),
                    )
                    blocks.append(block)
                    all_text_parts.append(text)
                    total_confidence += conf
                    
            elif region_type == "table":
                # Handle table regions
                res = region.get("res", {})
                html = res.get("html", "") if isinstance(res, dict) else ""
                
                if html:
                    # Parse HTML table to structured data
                    table_data = self._parse_html_table(html)
                    table_data["bbox"] = region_bbox
                    table_data["regionIndex"] = region_idx
                    tables.append(table_data)
                    
                    # Also extract text for full_text
                    table_text = self._html_table_to_text(html)
                    all_text_parts.append(table_text)
                    
                    # Create blocks from table cells for compatibility
                    cell_blocks = self._table_cells_to_blocks(
                        table_data, 
                        region_bbox, 
                        len(blocks)
                    )
                    blocks.extend(cell_blocks)
                    total_confidence += sum(b.confidence for b in cell_blocks)
        
        overall_conf = total_confidence / len(blocks) if blocks else 0.0
        
        result = OCRResult(
            full_text="\n".join(all_text_parts),
            blocks=blocks,
            overall_confidence=overall_conf,
        )
        
        # Store parsed tables in result
        result.tables = tables
        
        return result
    
    def _parse_html_table(self, html: str) -> Dict[str, Any]:
        """
        Parse HTML table string into structured data.
        
        Args:
            html: HTML table string from PPStructure
            
        Returns:
            Dict with:
                - headers: List of column header names
                - rows: List of row data (each row is a list of cell values)
                - rowCount: Number of rows
                - colCount: Number of columns
        """
        from html.parser import HTMLParser
        
        class TableHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.current_row = []
                self.current_cell = ""
                self.in_cell = False
                self.in_header = False
                self.headers = []
                
            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.current_row = []
                elif tag in ("td", "th"):
                    self.in_cell = True
                    self.current_cell = ""
                    if tag == "th":
                        self.in_header = True
                        
            def handle_endtag(self, tag):
                if tag == "tr":
                    if self.current_row:
                        self.rows.append(self.current_row)
                elif tag in ("td", "th"):
                    cell_text = self.current_cell.strip()
                    self.current_row.append(cell_text)
                    if tag == "th":
                        self.headers.append(cell_text)
                    self.in_cell = False
                    self.in_header = False
                    
            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell += data
        
        parser = TableHTMLParser()
        try:
            parser.feed(html)
        except Exception as e:
            logger.warning(f"HTML table parsing error: {e}")
            return {"headers": [], "rows": [], "rowCount": 0, "colCount": 0}
        
        # If no explicit headers found, use first row
        headers = parser.headers
        rows = parser.rows
        
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]
        
        col_count = max(len(row) for row in rows) if rows else len(headers)
        
        return {
            "headers": headers,
            "rows": rows,
            "rowCount": len(rows),
            "colCount": col_count,
        }
    
    def _html_table_to_text(self, html: str) -> str:
        """Convert HTML table to plain text."""
        table_data = self._parse_html_table(html)
        
        lines = []
        
        # Header row
        if table_data["headers"]:
            lines.append(" | ".join(table_data["headers"]))
            lines.append("-" * 40)
        
        # Data rows
        for row in table_data["rows"]:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)
    
    def _table_cells_to_blocks(
        self, 
        table_data: Dict, 
        region_bbox: List[float],
        start_index: int
    ) -> List[OCRBlock]:
        """
        Convert table cells to OCRBlock objects for compatibility.
        
        Creates approximate bounding boxes based on table region and cell positions.
        """
        from .result import OCRBlock
        
        blocks = []
        rows = table_data.get("rows", [])
        headers = table_data.get("headers", [])
        
        if not rows:
            return blocks
        
        # Calculate cell dimensions based on region bbox
        x1, y1, x2, y2 = region_bbox if len(region_bbox) >= 4 else [0, 0, 100, 100]
        table_width = x2 - x1
        table_height = y2 - y1
        
        num_rows = len(rows) + (1 if headers else 0)
        num_cols = table_data.get("colCount", 1) or 1
        
        cell_width = table_width / num_cols
        cell_height = table_height / num_rows
        
        row_offset = 0
        
        # Header row blocks
        if headers:
            for col_idx, header in enumerate(headers):
                if header:
                    cx1 = x1 + col_idx * cell_width
                    cy1 = y1
                    cx2 = cx1 + cell_width
                    cy2 = y1 + cell_height
                    
                    bbox = [[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]]
                    
                    block = OCRBlock(
                        text=header,
                        confidence=0.95,  # Assume high confidence for table extraction
                        bbox=bbox,
                        line_index=start_index + len(blocks),
                    )
                    blocks.append(block)
            row_offset = 1
        
        # Data row blocks
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                if cell:
                    cx1 = x1 + col_idx * cell_width
                    cy1 = y1 + (row_idx + row_offset) * cell_height
                    cx2 = cx1 + cell_width
                    cy2 = cy1 + cell_height
                    
                    bbox = [[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]]
                    
                    block = OCRBlock(
                        text=str(cell),
                        confidence=0.95,
                        bbox=bbox,
                        line_index=start_index + len(blocks),
                    )
                    blocks.append(block)
        
        return blocks

