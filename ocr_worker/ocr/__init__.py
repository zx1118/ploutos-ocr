"""
OCR Module
==========

PaddleOCR engine wrapper with text and structure recognition support.

Components:
- OCREngine: Main OCR engine with multiple recognition modes
- OCRResult: Structured result container
- StructureExtractor: Document structure extraction
- VerticalTextHandler: Vertical text detection and recognition
"""

from .engine import OCREngine
from .result import OCRResult, OCRBlock, StructuredData
from .structure import StructureExtractor
from .vertical import VerticalTextHandler, merge_vertical_fields

__all__ = [
    "OCREngine",
    "OCRResult",
    "OCRBlock",
    "StructuredData",
    "StructureExtractor",
    "VerticalTextHandler",
    "merge_vertical_fields",
]

