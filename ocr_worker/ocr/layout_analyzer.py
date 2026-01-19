"""
Layout Analyzer Module
======================

Provides layout understanding capabilities for document parsing.
Designed as an abstraction layer for integrating various layout models:
- PaddleStructure
- DocLayout-YOLO
- LayoutLM-based models

This module implements the "Advanced Intelligence" phase of OCR optimization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger
import numpy as np


class LayoutRegionType(Enum):
    """Types of layout regions in documents."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    CAPTION = "caption"
    EQUATION = "equation"
    SEAL = "seal"
    QRCODE = "qrcode"
    UNKNOWN = "unknown"


@dataclass
class LayoutRegion:
    """A detected layout region in the document."""
    
    region_type: LayoutRegionType
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    content: str = ""
    
    # For table regions
    table_structure: Optional[Dict] = None
    rows: int = 0
    cols: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.region_type.value,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "content": self.content,
            "tableStructure": self.table_structure,
            "rows": self.rows,
            "cols": self.cols,
        }


@dataclass
class LayoutAnalysisResult:
    """Complete layout analysis result."""
    
    regions: List[LayoutRegion] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    
    # Document metadata
    page_width: int = 0
    page_height: int = 0
    orientation: str = "portrait"
    
    # Analysis info
    model_used: str = "rule-based"
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "readingOrder": self.reading_order,
            "pageWidth": self.page_width,
            "pageHeight": self.page_height,
            "orientation": self.orientation,
            "modelUsed": self.model_used,
            "durationMs": self.duration_ms,
        }
    
    def get_regions_by_type(self, region_type: LayoutRegionType) -> List[LayoutRegion]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.region_type == region_type]
    
    def get_table_regions(self) -> List[LayoutRegion]:
        """Get all table regions."""
        return self.get_regions_by_type(LayoutRegionType.TABLE)


class RuleBasedLayoutAnalyzer:
    """
    Rule-based layout analyzer using OCR block positions.
    
    This is a lightweight alternative when no ML model is available.
    Works by analyzing block distributions and applying heuristics.
    """
    
    # Typical invoice structure
    INVOICE_REGIONS = {
        "header": {"y_ratio": (0, 0.15), "keywords": ["发票", "电子发票", "增值税"]},
        "buyer_seller": {"y_ratio": (0.1, 0.3), "keywords": ["购买方", "销售方", "名称", "税号"]},
        "items_table": {"y_ratio": (0.25, 0.75), "keywords": ["项目名称", "数量", "金额"]},
        "totals": {"y_ratio": (0.65, 0.85), "keywords": ["合计", "价税合计", "大写"]},
        "footer": {"y_ratio": (0.8, 1.0), "keywords": ["备注", "开票人", "收款人"]},
    }
    
    def __init__(self):
        self.regions: List[LayoutRegion] = []
    
    def analyze(
        self, 
        blocks: List[Dict], 
        image_width: int,
        image_height: int,
    ) -> LayoutAnalysisResult:
        """
        Analyze document layout from OCR blocks.
        
        Args:
            blocks: OCR blocks with text and bbox
            image_width: Image width
            image_height: Image height
            
        Returns:
            LayoutAnalysisResult
        """
        import time
        start_time = time.time()
        
        self.regions = []
        
        if not blocks or image_height == 0:
            return LayoutAnalysisResult(model_used="rule-based")
        
        # Detect table region (main content area)
        table_region = self._detect_table_region(blocks, image_width, image_height)
        if table_region:
            self.regions.append(table_region)
        
        # Detect header region
        header_region = self._detect_header_region(blocks, image_width, image_height)
        if header_region:
            self.regions.append(header_region)
        
        # Detect footer region
        footer_region = self._detect_footer_region(blocks, image_width, image_height)
        if footer_region:
            self.regions.append(footer_region)
        
        # Determine reading order
        reading_order = self._compute_reading_order()
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return LayoutAnalysisResult(
            regions=self.regions,
            reading_order=reading_order,
            page_width=image_width,
            page_height=image_height,
            orientation="portrait" if image_height > image_width else "landscape",
            model_used="rule-based",
            duration_ms=duration_ms,
        )
    
    def _detect_table_region(
        self, 
        blocks: List[Dict], 
        width: int, 
        height: int
    ) -> Optional[LayoutRegion]:
        """Detect table region using keyword matching."""
        table_keywords = ["项目名称", "规格", "单位", "数量", "单价", "金额", "税率"]
        
        table_blocks = []
        for block in blocks:
            text = block.get("text", "")
            bbox = block.get("bbox", [])
            
            if not bbox:
                continue
            
            # Check if block is in middle area (typical table position)
            ys = [p[1] for p in bbox] if isinstance(bbox[0], (list, tuple)) else [bbox[1], bbox[3]]
            y_center = sum(ys) / len(ys)
            y_ratio = y_center / height if height > 0 else 0
            
            if 0.2 <= y_ratio <= 0.85:
                table_blocks.append(block)
            
            # Check for header keywords
            for kw in table_keywords:
                if kw in text:
                    table_blocks.append(block)
                    break
        
        if not table_blocks:
            return None
        
        # Compute bounding box of table region
        all_x = []
        all_y = []
        for block in table_blocks:
            bbox = block.get("bbox", [])
            if isinstance(bbox[0], (list, tuple)):
                for p in bbox:
                    all_x.append(p[0])
                    all_y.append(p[1])
            elif len(bbox) >= 4:
                all_x.extend([bbox[0], bbox[2]])
                all_y.extend([bbox[1], bbox[3]])
        
        if not all_x or not all_y:
            return None
        
        return LayoutRegion(
            region_type=LayoutRegionType.TABLE,
            bbox=[min(all_x), min(all_y), max(all_x), max(all_y)],
            confidence=0.8,
            content="",
        )
    
    def _detect_header_region(
        self, 
        blocks: List[Dict], 
        width: int, 
        height: int
    ) -> Optional[LayoutRegion]:
        """Detect header region at top of document."""
        header_blocks = []
        
        for block in blocks:
            bbox = block.get("bbox", [])
            if not bbox:
                continue
            
            ys = [p[1] for p in bbox] if isinstance(bbox[0], (list, tuple)) else [bbox[1], bbox[3]]
            y_center = sum(ys) / len(ys)
            y_ratio = y_center / height if height > 0 else 0
            
            if y_ratio < 0.2:
                header_blocks.append(block)
        
        if not header_blocks:
            return None
        
        all_x = []
        all_y = []
        for block in header_blocks:
            bbox = block.get("bbox", [])
            if isinstance(bbox[0], (list, tuple)):
                for p in bbox:
                    all_x.append(p[0])
                    all_y.append(p[1])
        
        if not all_x:
            return None
        
        return LayoutRegion(
            region_type=LayoutRegionType.HEADER,
            bbox=[min(all_x), min(all_y), max(all_x), max(all_y)],
            confidence=0.7,
        )
    
    def _detect_footer_region(
        self, 
        blocks: List[Dict], 
        width: int, 
        height: int
    ) -> Optional[LayoutRegion]:
        """Detect footer region at bottom of document."""
        footer_blocks = []
        
        for block in blocks:
            text = block.get("text", "")
            bbox = block.get("bbox", [])
            
            if not bbox:
                continue
            
            ys = [p[1] for p in bbox] if isinstance(bbox[0], (list, tuple)) else [bbox[1], bbox[3]]
            y_center = sum(ys) / len(ys)
            y_ratio = y_center / height if height > 0 else 0
            
            # Footer keywords
            if y_ratio > 0.8 or any(kw in text for kw in ["备注", "开票人", "收款人"]):
                footer_blocks.append(block)
        
        if not footer_blocks:
            return None
        
        all_x = []
        all_y = []
        for block in footer_blocks:
            bbox = block.get("bbox", [])
            if isinstance(bbox[0], (list, tuple)):
                for p in bbox:
                    all_x.append(p[0])
                    all_y.append(p[1])
        
        if not all_x:
            return None
        
        return LayoutRegion(
            region_type=LayoutRegionType.FOOTER,
            bbox=[min(all_x), min(all_y), max(all_x), max(all_y)],
            confidence=0.7,
        )
    
    def _compute_reading_order(self) -> List[int]:
        """Compute reading order of regions (top to bottom, left to right)."""
        if not self.regions:
            return []
        
        # Sort by y then x
        indexed_regions = [(i, r) for i, r in enumerate(self.regions)]
        indexed_regions.sort(key=lambda x: (x[1].bbox[1], x[1].bbox[0]))
        
        return [i for i, _ in indexed_regions]


class LayoutModelInterface:
    """
    Interface for ML-based layout models.
    
    Implementations can wrap:
    - PaddleStructure
    - DocLayout-YOLO
    - LayoutLM
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
    
    def load_model(self) -> bool:
        """Load the layout model. Override in subclasses."""
        raise NotImplementedError
    
    def analyze(
        self, 
        image_path: str,
    ) -> LayoutAnalysisResult:
        """Analyze document layout. Override in subclasses."""
        raise NotImplementedError


# ============================================================================
# Template Learning (Supplier-specific column caching)
# ============================================================================

class TemplateCache:
    """
    Cache for learned invoice templates.
    
    Stores column layouts for known suppliers to improve
    recognition accuracy on repeat invoices.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def get_template(self, supplier_key: str) -> Optional[Dict]:
        """Get cached template for supplier."""
        return self._cache.get(supplier_key)
    
    def save_template(
        self, 
        supplier_key: str, 
        column_headers: List[str],
        column_positions: List[Tuple[float, float]],
        confidence: float = 0.9,
    ) -> None:
        """Save template for supplier."""
        self._cache[supplier_key] = {
            "headers": column_headers,
            "positions": column_positions,
            "confidence": confidence,
            "usage_count": 0,
        }
        logger.info(f"Template cached for supplier: {supplier_key}")
    
    def match_template(
        self, 
        detected_text: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Find matching template from detected text.
        
        Returns:
            (supplier_key, template) or (None, None)
        """
        # Simple keyword matching
        detected_lower = detected_text.lower()
        
        for key, template in self._cache.items():
            if key.lower() in detected_lower:
                template["usage_count"] += 1
                return key, template
        
        return None, None
    
    def export_cache(self) -> Dict:
        """Export cache for persistence."""
        return dict(self._cache)
    
    def import_cache(self, data: Dict) -> None:
        """Import cache from saved data."""
        self._cache.update(data)


# Global template cache instance
_template_cache = TemplateCache()


def get_template_cache() -> TemplateCache:
    """Get global template cache."""
    return _template_cache


# ============================================================================
# High-level API
# ============================================================================

class LayoutEngine:
    """Available layout analysis engines."""
    RULE_BASED = "rule-based"
    PADDLE_STRUCTURE = "paddle-structure"
    DOCLAYOUT_YOLO = "doclayout-yolo"
    AUTO = "auto"  # Automatically select best available


def analyze_document_layout(
    blocks: List[Dict] = None,
    image_path: str = None,
    image_width: int = 0,
    image_height: int = 0,
    engine: str = LayoutEngine.AUTO,
    use_gpu: bool = False,
) -> LayoutAnalysisResult:
    """
    Analyze document layout using the best available engine.
    
    Args:
        blocks: OCR blocks (for rule-based analysis)
        image_path: Path to image (for ML-based analysis)
        image_width: Image width
        image_height: Image height
        engine: Which engine to use (auto, paddle-structure, doclayout-yolo, rule-based)
        use_gpu: Use GPU acceleration
        
    Returns:
        LayoutAnalysisResult
    """
    # Auto-select engine
    if engine == LayoutEngine.AUTO:
        engine = _detect_best_engine()
    
    # Try ML engines first if requested
    if engine == LayoutEngine.PADDLE_STRUCTURE and image_path:
        try:
            from .paddle_structure import analyze_document_with_paddle
            return analyze_document_with_paddle(image_path, use_gpu=use_gpu)
        except ImportError:
            logger.warning("PaddleStructure not available, falling back to rule-based")
            engine = LayoutEngine.RULE_BASED
        except Exception as e:
            logger.warning(f"PaddleStructure failed: {e}, falling back to rule-based")
            engine = LayoutEngine.RULE_BASED
    
    if engine == LayoutEngine.DOCLAYOUT_YOLO and image_path:
        try:
            from .doclayout_yolo import analyze_document_with_yolo
            device = "cuda" if use_gpu else "cpu"
            return analyze_document_with_yolo(image_path, device=device)
        except ImportError:
            logger.warning("DocLayout-YOLO not available, falling back to rule-based")
            engine = LayoutEngine.RULE_BASED
        except Exception as e:
            logger.warning(f"DocLayout-YOLO failed: {e}, falling back to rule-based")
            engine = LayoutEngine.RULE_BASED
    
    # Fallback to rule-based
    if blocks:
        analyzer = RuleBasedLayoutAnalyzer()
        return analyzer.analyze(blocks, image_width, image_height)
    
    return LayoutAnalysisResult(model_used="none")


def _detect_best_engine() -> str:
    """Detect the best available layout engine."""
    # Try PaddleStructure first (more accurate for tables)
    try:
        from paddleocr import PPStructure
        logger.debug("PaddleStructure available")
        return LayoutEngine.PADDLE_STRUCTURE
    except ImportError:
        pass
    
    # Try DocLayout-YOLO
    try:
        try:
            from doclayout_yolo import YOLOv10
            logger.debug("DocLayout-YOLO available")
            return LayoutEngine.DOCLAYOUT_YOLO
        except ImportError:
            from ultralytics import YOLO
            # Check if model exists
            import os
            for path in ["models/doclayout_yolo.pt", "models/doclayout_yolo_docstructbench.pt"]:
                if os.path.exists(path):
                    logger.debug("YOLO model found")
                    return LayoutEngine.DOCLAYOUT_YOLO
    except ImportError:
        pass
    
    # Fallback
    logger.debug("Using rule-based layout analyzer")
    return LayoutEngine.RULE_BASED


def get_available_engines() -> List[str]:
    """Get list of available layout engines."""
    engines = [LayoutEngine.RULE_BASED]
    
    try:
        from paddleocr import PPStructure
        engines.append(LayoutEngine.PADDLE_STRUCTURE)
    except ImportError:
        pass
    
    try:
        from doclayout_yolo import YOLOv10
        engines.append(LayoutEngine.DOCLAYOUT_YOLO)
    except ImportError:
        try:
            from ultralytics import YOLO
            engines.append(LayoutEngine.DOCLAYOUT_YOLO)
        except ImportError:
            pass
    
    return engines


def apply_template_if_available(
    full_text: str,
    table_structure: Dict,
) -> Dict:
    """
    Apply cached template if supplier is recognized.
    
    Args:
        full_text: Full OCR text
        table_structure: Detected table structure
        
    Returns:
        Enhanced table structure with template info
    """
    cache = get_template_cache()
    supplier_key, template = cache.match_template(full_text)
    
    if template:
        logger.info(f"Template matched: {supplier_key}")
        table_structure["templateApplied"] = True
        table_structure["templateSupplier"] = supplier_key
        table_structure["templateConfidence"] = template["confidence"]
        
        # Could merge template columns with detected columns here
    
    return table_structure
