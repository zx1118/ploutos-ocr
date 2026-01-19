"""
PaddleStructure Integration
===========================

Integrates PaddleStructure (PPStructure) for advanced document layout analysis
and table structure recognition.

Features:
- Layout detection (text, table, figure, title, list)
- Table structure recognition (row/column extraction)
- Reading order determination
- Formula and seal detection

Installation:
    pip install paddlepaddle paddleocr
    # or for GPU:
    pip install paddlepaddle-gpu paddleocr

Reference:
    https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/README.md
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


# Mapping from PPStructure labels to our types
PPSTRUCTURE_LABEL_MAP = {
    "text": LayoutRegionType.TEXT,
    "title": LayoutRegionType.TITLE,
    "figure": LayoutRegionType.FIGURE,
    "figure_caption": LayoutRegionType.CAPTION,
    "table": LayoutRegionType.TABLE,
    "table_caption": LayoutRegionType.CAPTION,
    "header": LayoutRegionType.HEADER,
    "footer": LayoutRegionType.FOOTER,
    "reference": LayoutRegionType.TEXT,
    "equation": LayoutRegionType.EQUATION,
    "list": LayoutRegionType.LIST,
}


class PaddleStructureEngine:
    """
    PaddleStructure engine for document layout analysis.
    
    Usage:
        engine = PaddleStructureEngine()
        result = engine.analyze("invoice.pdf")
        
        # Get table regions
        tables = result.get_table_regions()
        for table in tables:
            print(table.table_structure)
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        table_engine: bool = True,
        layout_engine: bool = True,
        show_log: bool = False,
    ):
        """
        Initialize PaddleStructure engine.
        
        Args:
            use_gpu: Use GPU for inference
            lang: Language code (ch, en, etc.)
            table_engine: Enable table structure recognition
            layout_engine: Enable layout detection
            show_log: Show PaddleOCR logs
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.table_engine = table_engine
        self.layout_engine = layout_engine
        self.show_log = show_log
        
        self._engine = None
        self._table_engine = None
        self._initialized = False
    
    def _init_engine(self) -> None:
        """Lazy initialize PPStructure engine."""
        if self._initialized:
            return
        
        try:
            from paddleocr import PPStructure
            
            logger.info("Initializing PaddleStructure engine...")
            start_time = time.time()
            
            self._engine = PPStructure(
                use_gpu=self.use_gpu,
                lang=self.lang,
                show_log=self.show_log,
                # Layout model
                layout=self.layout_engine,
                # Table structure
                table=self.table_engine,
                # OCR for text extraction
                ocr=True,
                # Recovery (reconstruct document)
                recovery=False,
            )
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"PaddleStructure initialized in {duration}ms")
            
            self._initialized = True
            
        except ImportError as e:
            logger.error(f"PaddleStructure not installed: {e}")
            logger.error("Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleStructure: {e}")
            raise
    
    def analyze(
        self,
        image_path: Union[str, Path],
        return_ocr_result: bool = True,
    ) -> LayoutAnalysisResult:
        """
        Analyze document layout.
        
        Args:
            image_path: Path to image or PDF
            return_ocr_result: Include OCR text in result
            
        Returns:
            LayoutAnalysisResult with detected regions
        """
        self._init_engine()
        
        image_path = str(image_path)
        start_time = time.time()
        
        logger.info(f"Analyzing layout: {image_path}")
        
        try:
            # Run PPStructure
            result = self._engine(image_path, return_ocr_result_in_table=return_ocr_result)
            
            # Parse result
            regions = []
            
            for item in result:
                region_type = PPSTRUCTURE_LABEL_MAP.get(
                    item.get("type", "text"), 
                    LayoutRegionType.UNKNOWN
                )
                
                bbox = item.get("bbox", [0, 0, 0, 0])
                
                region = LayoutRegion(
                    region_type=region_type,
                    bbox=list(bbox),
                    confidence=item.get("score", 0.0),
                    content="",
                )
                
                # Handle table structure
                if region_type == LayoutRegionType.TABLE:
                    table_res = item.get("res", {})
                    if isinstance(table_res, dict):
                        region.table_structure = table_res.get("html", None)
                        # Extract cell info if available
                        cells = table_res.get("cell_bbox", [])
                        if cells:
                            # Estimate rows/cols from cells
                            region.rows = len(set(c[1] for c in cells if len(c) > 1))
                            region.cols = len(set(c[0] for c in cells if len(c) > 0))
                
                # Handle text content
                if region_type == LayoutRegionType.TEXT:
                    text_res = item.get("res", [])
                    if isinstance(text_res, list):
                        texts = []
                        for line in text_res:
                            if isinstance(line, dict):
                                texts.append(line.get("text", ""))
                            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                                texts.append(str(line[1][0]) if isinstance(line[1], tuple) else str(line[1]))
                        region.content = "\n".join(texts)
                
                regions.append(region)
            
            # Compute reading order (top to bottom, left to right)
            reading_order = list(range(len(regions)))
            reading_order.sort(key=lambda i: (regions[i].bbox[1], regions[i].bbox[0]))
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return LayoutAnalysisResult(
                regions=regions,
                reading_order=reading_order,
                model_used="PaddleStructure",
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            raise
    
    def extract_tables(
        self,
        image_path: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from document.
        
        Args:
            image_path: Path to image or PDF
            
        Returns:
            List of table structures with HTML and cell info
        """
        result = self.analyze(image_path)
        tables = []
        
        for region in result.get_table_regions():
            tables.append({
                "bbox": region.bbox,
                "confidence": region.confidence,
                "html": region.table_structure,
                "rows": region.rows,
                "cols": region.cols,
            })
        
        return tables
    
    def analyze_table_structure(
        self,
        image_path: Union[str, Path],
        table_bbox: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Detailed table structure analysis.
        
        Args:
            image_path: Path to image
            table_bbox: Optional bbox to crop table region
            
        Returns:
            Table structure with cells, rows, columns
        """
        self._init_engine()
        
        from PIL import Image
        import numpy as np
        
        # Load image
        img = Image.open(image_path)
        
        # Crop to table region if specified
        if table_bbox:
            img = img.crop(table_bbox)
        
        img_array = np.array(img)
        
        try:
            # Use table engine directly
            from paddleocr import PPStructure
            
            table_engine = PPStructure(
                use_gpu=self.use_gpu,
                lang=self.lang,
                show_log=self.show_log,
                layout=False,  # Skip layout, only table
                table=True,
                ocr=True,
            )
            
            result = table_engine(img_array)
            
            if result and len(result) > 0:
                table_res = result[0].get("res", {})
                return {
                    "html": table_res.get("html", ""),
                    "cells": table_res.get("cell_bbox", []),
                    "structure": table_res.get("structure", []),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Table structure analysis failed: {e}")
            return {}


# ============================================================================
# Convenience Functions
# ============================================================================

_engine: Optional[PaddleStructureEngine] = None


def get_paddle_structure_engine(
    use_gpu: bool = False,
    lang: str = "ch",
) -> PaddleStructureEngine:
    """Get or create PaddleStructure engine singleton."""
    global _engine
    
    if _engine is None:
        _engine = PaddleStructureEngine(use_gpu=use_gpu, lang=lang)
    
    return _engine


def analyze_document_with_paddle(
    image_path: str,
    use_gpu: bool = False,
) -> LayoutAnalysisResult:
    """
    Convenience function for document layout analysis.
    
    Args:
        image_path: Path to document image
        use_gpu: Use GPU acceleration
        
    Returns:
        LayoutAnalysisResult
    """
    engine = get_paddle_structure_engine(use_gpu=use_gpu)
    return engine.analyze(image_path)


def extract_table_html(
    image_path: str,
    use_gpu: bool = False,
) -> List[str]:
    """
    Extract tables as HTML from document.
    
    Args:
        image_path: Path to document image
        use_gpu: Use GPU acceleration
        
    Returns:
        List of HTML table strings
    """
    engine = get_paddle_structure_engine(use_gpu=use_gpu)
    tables = engine.extract_tables(image_path)
    return [t.get("html", "") for t in tables if t.get("html")]
