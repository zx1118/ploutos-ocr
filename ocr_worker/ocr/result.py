"""
OCR Result Data Structures
==========================

Structured containers for OCR results with serialization support.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import numpy as np


def to_python_type(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_type(v) for v in obj]
    return obj


@dataclass
class OCRBlock:
    """Single OCR text block with position and confidence."""
    
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    # Optional metadata
    line_index: int = 0
    word_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-safe types."""
        return {
            "text": str(self.text) if self.text else "",
            "confidence": round(float(self.confidence), 4) if self.confidence else 0.0,
            "bbox": to_python_type(self.bbox),
            "lineIndex": int(self.line_index),
            "wordIndex": int(self.word_index),
        }
    
    @classmethod
    def from_paddle_result(cls, result, line_idx: int = 0) -> "OCRBlock":
        """
        Create from PaddleOCR result.
        
        Handles multiple formats:
        - PaddleOCR 2.x: ([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence))
        - PaddleOCR 3.x: dict with 'text', 'score', 'poly' keys
        """
        if isinstance(result, dict):
            # PaddleOCR 3.x format
            text = result.get("text", result.get("rec_text", ""))
            confidence = result.get("score", result.get("rec_score", 0.0))
            bbox = result.get("poly", result.get("dt_polys", [[0, 0]] * 4))
            # Convert flat list to nested if needed
            if bbox is not None and len(bbox) > 0:
                if isinstance(bbox[0], (int, float, np.integer, np.floating)):
                    bbox = [[bbox[i], bbox[i+1]] for i in range(0, len(bbox), 2)]
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            # PaddleOCR 2.x format
            bbox, text_conf = result
            if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                text, confidence = text_conf
            else:
                text, confidence = str(text_conf), 0.0
        else:
            # Fallback
            text = str(result) if result else ""
            confidence = 0.0
            bbox = [[0, 0]] * 4
        
        # Convert numpy types to Python types
        bbox = to_python_type(bbox) if bbox else [[0, 0]] * 4
        
        return cls(
            text=str(text) if text else "",
            confidence=float(confidence) if confidence else 0.0,
            bbox=bbox,
            line_index=int(line_idx),
        )


@dataclass
class StructuredData:
    """Extracted structured data from document."""
    
    doc_type: str = "UNKNOWN"
    fields: Dict[str, Any] = field(default_factory=dict)
    field_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Common fields for invoices
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None
    amount: Optional[str] = None
    tax_amount: Optional[str] = None
    total_amount: Optional[str] = None
    seller_name: Optional[str] = None
    buyer_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "docType": self.doc_type,
            "fields": self.fields,
            "fieldConfidences": {k: round(v, 4) for k, v in self.field_confidences.items()},
        }
        
        # Add common fields if present
        for field_name in ["invoice_no", "invoice_date", "amount", "tax_amount", 
                          "total_amount", "seller_name", "buyer_name"]:
            value = getattr(self, field_name)
            if value is not None:
                # Convert snake_case to camelCase
                camel_name = "".join(
                    word.capitalize() if i > 0 else word 
                    for i, word in enumerate(field_name.split("_"))
                )
                result[camel_name] = value
        
        return result
    
    def get_low_confidence_fields(self, threshold: float = 0.8) -> List[str]:
        """Get fields with confidence below threshold."""
        return [
            field for field, conf in self.field_confidences.items()
            if conf < threshold
        ]


@dataclass
class OCRResult:
    """Complete OCR result container."""
    
    # Raw text
    full_text: str = ""
    
    # Detected blocks
    blocks: List[OCRBlock] = field(default_factory=list)
    
    # Row-level structure (blocks clustered into rows)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Table structure (columns + rows) - from row clustering
    table_structure: Optional[Dict[str, Any]] = None
    
    # PPStructure parsed tables (方案 A: PPStructure 表格识别)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    
    # Layout analysis result (from PaddleStructure/DocLayout-YOLO)
    layout_analysis: Optional[Dict[str, Any]] = None
    
    # Structured extraction
    structured_data: Optional[StructuredData] = None
    
    # Overall metrics
    overall_confidence: float = 0.0
    needs_review: bool = False
    low_confidence_fields: List[str] = field(default_factory=list)
    
    # Applied corrections
    applied_rules: List[Dict[str, str]] = field(default_factory=list)
    
    # Processing info
    duration_ms: int = 0
    model_version: str = "PP-OCRv4"
    image_width: int = 0
    image_height: int = 0
    page_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization with safe types."""
        return to_python_type({
            "fullText": str(self.full_text) if self.full_text else "",
            "blocks": [b.to_dict() for b in self.blocks],
            "rows": self.rows if self.rows else [],
            "tableStructure": self.table_structure if self.table_structure else None,
            "tables": self.tables if self.tables else [],  # PPStructure 表格
            "layoutAnalysis": self.layout_analysis if self.layout_analysis else None,
            "structuredData": self.structured_data.to_dict() if self.structured_data else None,
            "overallConfidence": round(float(self.overall_confidence), 4) if self.overall_confidence else 0.0,
            "needsReview": bool(self.needs_review),
            "lowConfidenceFields": list(self.low_confidence_fields) if self.low_confidence_fields else [],
            "appliedRules": list(self.applied_rules) if self.applied_rules else [],
            "durationMs": int(self.duration_ms) if self.duration_ms else 0,
            "modelVersion": str(self.model_version) if self.model_version else "PP-OCRv5",
            "imageWidth": int(self.image_width) if self.image_width else 0,
            "imageHeight": int(self.image_height) if self.image_height else 0,
            "pageCount": int(self.page_count) if self.page_count else 1,
        })
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_paddle_results(
        cls,
        results,
        image_size: tuple = (0, 0),
        duration_ms: int = 0,
    ) -> "OCRResult":
        """
        Create from PaddleOCR results.
        
        Handles multiple formats:
        - PaddleOCR 2.x: List[List[tuple]] - list of pages, each page is list of (bbox, (text, conf))
        - PaddleOCR 3.x: dict or list of dicts with 'rec_texts', 'rec_scores', etc.
        
        Args:
            results: PaddleOCR output
            image_size: (width, height) of image
            duration_ms: Processing time
        """
        blocks = []
        all_text_parts = []
        total_confidence = 0.0
        
        # Handle None or empty results
        if results is None:
            return cls(duration_ms=duration_ms)
        
        # Handle PaddleOCR 3.x dict format
        if isinstance(results, dict):
            rec_texts = results.get("rec_texts", results.get("rec_text", []))
            rec_scores = results.get("rec_scores", results.get("rec_score", []))
            dt_polys = results.get("dt_polys", [])
            
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            if isinstance(rec_scores, (int, float)):
                rec_scores = [rec_scores]
            
            for idx, text in enumerate(rec_texts):
                conf = rec_scores[idx] if idx < len(rec_scores) else 0.0
                bbox = dt_polys[idx] if idx < len(dt_polys) else [[0, 0]] * 4
                
                block = OCRBlock(
                    text=str(text),
                    confidence=float(conf) if conf else 0.0,
                    bbox=bbox,
                    line_index=idx,
                )
                blocks.append(block)
                all_text_parts.append(block.text)
                total_confidence += block.confidence
                
        # Handle list format (could be 2.x or 3.x list of dicts)
        elif isinstance(results, list):
            # Check if it's a list of dicts (3.x format)
            if results and isinstance(results[0], dict):
                for idx, item in enumerate(results):
                    block = OCRBlock.from_paddle_result(item, idx)
                    blocks.append(block)
                    all_text_parts.append(block.text)
                    total_confidence += block.confidence
            else:
                # PaddleOCR 2.x format: list of pages
                for page_idx, page_results in enumerate(results):
                    if page_results is None:
                        continue
                    
                    # Handle if page_results is also a dict
                    if isinstance(page_results, dict):
                        block = OCRBlock.from_paddle_result(page_results, page_idx)
                        blocks.append(block)
                        all_text_parts.append(block.text)
                        total_confidence += block.confidence
                    elif isinstance(page_results, list):
                        for line_idx, line_result in enumerate(page_results):
                            block = OCRBlock.from_paddle_result(line_result, line_idx)
                            blocks.append(block)
                            all_text_parts.append(block.text)
                            total_confidence += block.confidence
        
        # Calculate overall confidence
        overall_conf = total_confidence / len(blocks) if blocks else 0.0
        
        return cls(
            full_text="\n".join(all_text_parts),
            blocks=blocks,
            overall_confidence=overall_conf,
            duration_ms=duration_ms,
            image_width=image_size[0],
            image_height=image_size[1],
            page_count=len(results) if isinstance(results, list) else 1,
        )
    
    def calculate_needs_review(self, confidence_threshold: float = 0.8) -> None:
        """Calculate if result needs manual review."""
        # Check overall confidence
        if self.overall_confidence < confidence_threshold:
            self.needs_review = True
        
        # Check individual fields
        if self.structured_data:
            self.low_confidence_fields = self.structured_data.get_low_confidence_fields(
                confidence_threshold
            )
            if self.low_confidence_fields:
                self.needs_review = True

