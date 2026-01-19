"""
DocLayout-YOLO Integration
==========================

Integrates DocLayout-YOLO for fast document layout detection.
Based on the YOLO architecture, optimized for document analysis.

Features:
- Fast layout detection (10+ FPS on GPU)
- Multi-class document element detection
- High accuracy on complex layouts

Installation:
    pip install doclayout-yolo
    # or from source:
    git clone https://github.com/opendatalab/DocLayout-YOLO
    cd DocLayout-YOLO && pip install -e .

Model weights:
    Download from: https://github.com/opendatalab/DocLayout-YOLO/releases
    Place in: models/doclayout_yolo_docstructbench.pt

Reference:
    https://github.com/opendatalab/DocLayout-YOLO
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from loguru import logger

from .layout_analyzer import (
    LayoutAnalysisResult,
    LayoutRegion,
    LayoutRegionType,
)


# DocLayout-YOLO class mapping (DocStructBench model)
DOCLAYOUT_CLASS_MAP = {
    0: LayoutRegionType.TITLE,
    1: LayoutRegionType.TEXT,       # plain text
    2: LayoutRegionType.TEXT,       # abandon (crossed out text)
    3: LayoutRegionType.FIGURE,
    4: LayoutRegionType.CAPTION,    # figure caption
    5: LayoutRegionType.TABLE,
    6: LayoutRegionType.CAPTION,    # table caption
    7: LayoutRegionType.FOOTER,     # table footnote
    8: LayoutRegionType.HEADER,     # section header
    9: LayoutRegionType.LIST,
    10: LayoutRegionType.TEXT,      # code
    11: LayoutRegionType.EQUATION,
}

# Class names for logging
DOCLAYOUT_CLASS_NAMES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "section_header",
    9: "list",
    10: "code",
    11: "equation",
}


class DocLayoutYOLOEngine:
    """
    DocLayout-YOLO engine for document layout detection.
    
    Usage:
        engine = DocLayoutYOLOEngine(model_path="models/doclayout_yolo.pt")
        result = engine.analyze("invoice.png")
        
        for region in result.regions:
            print(f"{region.region_type}: {region.bbox}")
    """
    
    # Default model paths to search
    DEFAULT_MODEL_PATHS = [
        "models/doclayout_yolo_docstructbench.pt",
        "models/doclayout_yolo.pt",
        "~/.cache/doclayout_yolo/doclayout_yolo_docstructbench.pt",
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",  # "cpu", "cuda", "cuda:0"
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize DocLayout-YOLO engine.
        
        Args:
            model_path: Path to model weights (.pt file)
            device: Device for inference
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self._model = None
        self._initialized = False
    
    def _find_model_path(self) -> Optional[str]:
        """Find model file from default paths, auto-download if not found."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        
        for path in self.DEFAULT_MODEL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        
        # Auto-download model if not found
        return self._auto_download_model()
    
    def _auto_download_model(self) -> Optional[str]:
        """
        Auto-download DocLayout-YOLO model from HuggingFace.
        
        Model: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
        """
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info("Auto-downloading DocLayout-YOLO model from HuggingFace...")
            
            model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                cache_dir=os.path.expanduser("~/.cache/doclayout_yolo"),
            )
            
            logger.info(f"DocLayout-YOLO model downloaded to: {model_path}")
            return model_path
            
        except ImportError:
            logger.warning("huggingface_hub not installed, cannot auto-download model")
            logger.warning("Install with: pip install huggingface_hub")
            return None
        except Exception as e:
            logger.warning(f"Failed to auto-download model: {e}")
            return None
    
    def _init_engine(self) -> None:
        """Lazy initialize DocLayout-YOLO model."""
        if self._initialized:
            return
        
        model_path = self._find_model_path()
        
        if not model_path:
            raise FileNotFoundError(
                f"DocLayout-YOLO model not found. "
                f"Please download from https://github.com/opendatalab/DocLayout-YOLO/releases "
                f"and place at one of: {self.DEFAULT_MODEL_PATHS}"
            )
        
        try:
            # Try doclayout_yolo package first
            try:
                from doclayout_yolo import YOLOv10
                
                logger.info(f"Loading DocLayout-YOLO from: {model_path}")
                start_time = time.time()
                
                self._model = YOLOv10(model_path)
                
                duration = int((time.time() - start_time) * 1000)
                logger.info(f"DocLayout-YOLO loaded in {duration}ms")
                
            except ImportError:
                # Fallback to ultralytics YOLO
                from ultralytics import YOLO
                
                logger.info(f"Loading YOLO model from: {model_path}")
                start_time = time.time()
                
                self._model = YOLO(model_path)
                
                duration = int((time.time() - start_time) * 1000)
                logger.info(f"YOLO model loaded in {duration}ms")
            
            self._initialized = True
            
        except ImportError as e:
            logger.error(f"DocLayout-YOLO dependencies not installed: {e}")
            logger.error("Install with: pip install doclayout-yolo")
            logger.error("Or fallback: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load DocLayout-YOLO: {e}")
            raise
    
    def analyze(
        self,
        image_path: Union[str, Path],
        return_image: bool = False,
    ) -> LayoutAnalysisResult:
        """
        Analyze document layout.
        
        Args:
            image_path: Path to image
            return_image: Include annotated image in result
            
        Returns:
            LayoutAnalysisResult with detected regions
        """
        self._init_engine()
        
        image_path = str(image_path)
        start_time = time.time()
        
        logger.info(f"Analyzing layout with DocLayout-YOLO: {image_path}")
        
        try:
            # Run inference
            results = self._model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
            )
            
            regions = []
            
            # Parse results
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for i, box in enumerate(boxes):
                    # Get bbox (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                    
                    # Get class and confidence
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Map to our region type
                    region_type = DOCLAYOUT_CLASS_MAP.get(cls_id, LayoutRegionType.UNKNOWN)
                    cls_name = DOCLAYOUT_CLASS_NAMES.get(cls_id, "unknown")
                    
                    region = LayoutRegion(
                        region_type=region_type,
                        bbox=bbox,
                        confidence=conf,
                        content=cls_name,  # Store original class name
                    )
                    
                    regions.append(region)
                
                # Get image dimensions
                if hasattr(result, "orig_shape"):
                    height, width = result.orig_shape[:2]
                else:
                    height, width = 0, 0
            
            # Compute reading order (top to bottom, left to right)
            reading_order = list(range(len(regions)))
            reading_order.sort(key=lambda i: (regions[i].bbox[1], regions[i].bbox[0]))
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return LayoutAnalysisResult(
                regions=regions,
                reading_order=reading_order,
                page_width=width if 'width' in dir() else 0,
                page_height=height if 'height' in dir() else 0,
                model_used="DocLayout-YOLO",
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            raise
    
    def detect_tables(
        self,
        image_path: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        Detect table regions in document.
        
        Args:
            image_path: Path to image
            
        Returns:
            List of table detections with bbox and confidence
        """
        result = self.analyze(image_path)
        
        tables = []
        for region in result.get_table_regions():
            tables.append({
                "bbox": region.bbox,
                "confidence": region.confidence,
                "x1": region.bbox[0],
                "y1": region.bbox[1],
                "x2": region.bbox[2],
                "y2": region.bbox[3],
                "width": region.bbox[2] - region.bbox[0],
                "height": region.bbox[3] - region.bbox[1],
            })
        
        return tables


# ============================================================================
# Convenience Functions
# ============================================================================

_engine: Optional[DocLayoutYOLOEngine] = None


def get_doclayout_yolo_engine(
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> DocLayoutYOLOEngine:
    """Get or create DocLayout-YOLO engine singleton."""
    global _engine
    
    if _engine is None:
        _engine = DocLayoutYOLOEngine(model_path=model_path, device=device)
    
    return _engine


def analyze_document_with_yolo(
    image_path: str,
    device: str = "cpu",
) -> LayoutAnalysisResult:
    """
    Convenience function for document layout analysis.
    
    Args:
        image_path: Path to document image
        device: Device for inference
        
    Returns:
        LayoutAnalysisResult
    """
    engine = get_doclayout_yolo_engine(device=device)
    return engine.analyze(image_path)


def detect_document_tables(
    image_path: str,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Detect tables in document.
    
    Args:
        image_path: Path to document image
        device: Device for inference
        
    Returns:
        List of table detections
    """
    engine = get_doclayout_yolo_engine(device=device)
    return engine.detect_tables(image_path)
