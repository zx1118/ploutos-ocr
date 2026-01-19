"""
Field Evidence Matcher
======================

Matches field values to OCR text blocks and returns bbox coordinates.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

from loguru import logger

from ..ocr.result import OCRBlock


@dataclass
class MatchResult:
    """Field match result with bbox."""
    
    field_key: str
    field_value: str
    matched: bool
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    text_snippet: Optional[str] = None
    confidence: float = 0.0
    match_type: str = "none"  # exact, contains, fuzzy, adjacent


class FieldMatcher:
    """
    Matches field values to OCR text blocks.
    
    Strategies:
    1. Exact match: field value equals block text
    2. Contains match: block text contains field value
    3. Fuzzy match: edit distance within threshold
    4. Adjacent merge: combine adjacent blocks
    """
    
    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        max_adjacent_gap: int = 50,
    ):
        """
        Initialize matcher.
        
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0-1)
            max_adjacent_gap: Max pixel gap for adjacent block merging
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.max_adjacent_gap = max_adjacent_gap
        
        # Field-specific normalizers
        self.normalizers = {
            "invoiceNo": self._normalize_invoice_no,
            "totalAmount": self._normalize_amount,
            "taxAmount": self._normalize_amount,
            "amount": self._normalize_amount,
            "invoiceDate": self._normalize_date,
            "checkCode": self._normalize_check_code,
        }
    
    def match_all(
        self,
        fields: Dict[str, str],
        blocks: List[OCRBlock],
    ) -> Dict[str, MatchResult]:
        """
        Match all fields to OCR blocks.
        
        Args:
            fields: {field_key: field_value}
            blocks: OCR text blocks with bbox
            
        Returns:
            {field_key: MatchResult}
        """
        results = {}
        
        for field_key, field_value in fields.items():
            if not field_value or not field_value.strip():
                continue
                
            result = self.match_field(field_key, field_value, blocks)
            results[field_key] = result
            
            if result.matched:
                logger.debug(
                    f"Field matched: {field_key}={field_value} -> "
                    f"bbox={result.bbox}, type={result.match_type}"
                )
            else:
                logger.debug(f"Field not matched: {field_key}={field_value}")
        
        return results
    
    def match_field(
        self,
        field_key: str,
        field_value: str,
        blocks: List[OCRBlock],
    ) -> MatchResult:
        """
        Match single field to OCR blocks.
        
        Tries multiple strategies in order:
        0. Context-aware match (for totalAmount, find near "合计"/"小写")
        1. Exact match
        2. Contains match
        3. Fuzzy match
        4. Adjacent block merge
        """
        # Normalize field value
        normalizer = self.normalizers.get(field_key, self._normalize_default)
        normalized_value = normalizer(field_value)
        
        # Strategy 0: Context-aware match for totalAmount
        if field_key == 'totalAmount':
            result = self._try_context_match(field_key, normalized_value, blocks)
            if result.matched:
                return result
        
        # Strategy 1: Exact match
        result = self._try_exact_match(field_key, normalized_value, blocks)
        if result.matched:
            return result
        
        # Strategy 2: Contains match
        result = self._try_contains_match(field_key, normalized_value, blocks)
        if result.matched:
            return result
        
        # Strategy 3: Fuzzy match
        result = self._try_fuzzy_match(field_key, normalized_value, blocks)
        if result.matched:
            return result
        
        # Strategy 4: Adjacent merge (for split values)
        result = self._try_adjacent_merge(field_key, normalized_value, blocks)
        if result.matched:
            return result
        
        # No match found
        return MatchResult(
            field_key=field_key,
            field_value=field_value,
            matched=False,
        )
    
    def _try_context_match(
        self,
        field_key: str,
        normalized_value: str,
        blocks: List[OCRBlock],
    ) -> MatchResult:
        """
        Try context-aware match for totalAmount.
        
        Look for amount values near keywords like "合计", "小写", "价税合计".
        """
        # Keywords that indicate total amount row
        total_keywords = ['合计', '小写', '价税合计', '（小写）', '(小写)']
        
        # First, find blocks containing the value
        candidate_blocks = []
        for block in blocks:
            block_amount = self._normalize_amount(block.text)
            if block_amount == normalized_value:
                candidate_blocks.append(block)
        
        if not candidate_blocks:
            return MatchResult(field_key=field_key, field_value=normalized_value, matched=False)
        
        # If only one candidate, use it
        if len(candidate_blocks) == 1:
            return MatchResult(
                field_key=field_key,
                field_value=normalized_value,
                matched=True,
                bbox=self._get_bbox_rect(candidate_blocks[0].bbox),
                text_snippet=candidate_blocks[0].text,
                confidence=0.9,
                match_type="context",
            )
        
        # Multiple candidates - prefer one near total keywords
        # Find blocks with total keywords
        keyword_blocks = []
        for block in blocks:
            block_lower = block.text.lower()
            for kw in total_keywords:
                if kw in block_lower or kw.lower() in block_lower:
                    keyword_blocks.append(block)
                    break
        
        # Find candidate closest to a keyword block (by Y coordinate)
        best_candidate = None
        best_distance = float('inf')
        
        for candidate in candidate_blocks:
            candidate_y = self._get_center_y(candidate.bbox)
            
            for kw_block in keyword_blocks:
                kw_y = self._get_center_y(kw_block.bbox)
                distance = abs(candidate_y - kw_y)
                
                # Must be on similar Y level (same row)
                if distance < 50 and distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate
        
        if best_candidate:
            return MatchResult(
                field_key=field_key,
                field_value=normalized_value,
                matched=True,
                bbox=self._get_bbox_rect(best_candidate.bbox),
                text_snippet=best_candidate.text,
                confidence=0.95,
                match_type="context",
            )
        
        # Fallback: use the last candidate (usually total row is at bottom)
        last_candidate = max(candidate_blocks, key=lambda b: self._get_center_y(b.bbox))
        return MatchResult(
            field_key=field_key,
            field_value=normalized_value,
            matched=True,
            bbox=self._get_bbox_rect(last_candidate.bbox),
            text_snippet=last_candidate.text,
            confidence=0.8,
            match_type="context",
        )
    
    def _try_exact_match(
        self,
        field_key: str,
        normalized_value: str,
        blocks: List[OCRBlock],
    ) -> MatchResult:
        """Try exact text match."""
        for block in blocks:
            block_text = self._normalize_default(block.text)
            
            if block_text == normalized_value:
                return MatchResult(
                    field_key=field_key,
                    field_value=normalized_value,
                    matched=True,
                    bbox=self._get_bbox_rect(block.bbox),
                    text_snippet=block.text,
                    confidence=1.0,
                    match_type="exact",
                )
        
        return MatchResult(field_key=field_key, field_value=normalized_value, matched=False)
    
    def _try_contains_match(
        self,
        field_key: str,
        normalized_value: str,
        blocks: List[OCRBlock],
    ) -> MatchResult:
        """Try contains match (block contains value)."""
        # For name fields, allow shorter values (Chinese names are often 2-4 characters)
        min_length = 2 if field_key in ('buyerName', 'sellerName') else 4
        
        # Skip very short values to avoid false positives
        if len(normalized_value) < min_length:
            return MatchResult(field_key=field_key, field_value=normalized_value, matched=False)
        
        # Get field-specific normalizer
        normalizer = self.normalizers.get(field_key, self._normalize_default)
        
        for block in blocks:
            # Use default normalization for general matching
            block_text = self._normalize_default(block.text)
            
            if normalized_value in block_text:
                return MatchResult(
                    field_key=field_key,
                    field_value=normalized_value,
                    matched=True,
                    bbox=self._get_bbox_rect(block.bbox),
                    text_snippet=block.text,
                    confidence=0.9,
                    match_type="contains",
                )
            
            # For date/checkCode fields, also try field-specific normalization
            if field_key in ('invoiceDate', 'checkCode'):
                block_normalized = normalizer(block.text)
                # Skip empty normalized values to avoid false positives
                if block_normalized and len(block_normalized) >= 4:
                    if normalized_value == block_normalized:
                        return MatchResult(
                            field_key=field_key,
                            field_value=normalized_value,
                            matched=True,
                            bbox=self._get_bbox_rect(block.bbox),
                            text_snippet=block.text,
                            confidence=0.9,
                            match_type="contains",
                        )
        
        return MatchResult(field_key=field_key, field_value=normalized_value, matched=False)
    
    def _try_fuzzy_match(
        self,
        field_key: str,
        normalized_value: str,
        blocks: List[OCRBlock],
    ) -> MatchResult:
        """Try fuzzy match using edit distance."""
        best_match = None
        best_ratio = 0.0
        
        for block in blocks:
            block_text = self._normalize_default(block.text)
            
            # Skip very different length strings
            len_ratio = len(normalized_value) / max(len(block_text), 1)
            if len_ratio < 0.5 or len_ratio > 2.0:
                continue
            
            ratio = SequenceMatcher(None, normalized_value, block_text).ratio()
            
            if ratio > best_ratio and ratio >= self.fuzzy_threshold:
                best_ratio = ratio
                best_match = block
        
        if best_match:
            return MatchResult(
                field_key=field_key,
                field_value=normalized_value,
                matched=True,
                bbox=self._get_bbox_rect(best_match.bbox),
                text_snippet=best_match.text,
                confidence=best_ratio,
                match_type="fuzzy",
            )
        
        return MatchResult(field_key=field_key, field_value=normalized_value, matched=False)
    
    def _try_adjacent_merge(
        self,
        field_key: str,
        normalized_value: str,
        blocks: List[OCRBlock],
    ) -> MatchResult:
        """Try merging adjacent blocks."""
        # Sort blocks by position (top-left to bottom-right)
        sorted_blocks = sorted(blocks, key=lambda b: (
            self._get_center_y(b.bbox),
            self._get_center_x(b.bbox)
        ))
        
        # Try merging 2-3 adjacent blocks
        for i in range(len(sorted_blocks)):
            for j in range(i + 1, min(i + 4, len(sorted_blocks))):
                merged_blocks = sorted_blocks[i:j + 1]
                
                # Check if blocks are adjacent
                if not self._are_blocks_adjacent(merged_blocks):
                    continue
                
                # Merge text
                merged_text = "".join(b.text for b in merged_blocks)
                merged_text_normalized = self._normalize_default(merged_text)
                
                if merged_text_normalized == normalized_value:
                    # Compute merged bbox
                    merged_bbox = self._merge_bboxes([b.bbox for b in merged_blocks])
                    
                    return MatchResult(
                        field_key=field_key,
                        field_value=normalized_value,
                        matched=True,
                        bbox=merged_bbox,
                        text_snippet=merged_text,
                        confidence=0.85,
                        match_type="adjacent",
                    )
        
        return MatchResult(field_key=field_key, field_value=normalized_value, matched=False)
    
    def _are_blocks_adjacent(self, blocks: List[OCRBlock]) -> bool:
        """Check if blocks are adjacent (close enough to be same value)."""
        if len(blocks) < 2:
            return True
        
        for i in range(len(blocks) - 1):
            b1, b2 = blocks[i], blocks[i + 1]
            
            # Get bounding boxes
            rect1 = self._get_bbox_rect(b1.bbox)
            rect2 = self._get_bbox_rect(b2.bbox)
            
            # Check horizontal or vertical adjacency
            h_gap = abs(rect2[0] - rect1[2])  # x2_of_b1 to x1_of_b2
            v_gap = abs(rect2[1] - rect1[3])  # y2_of_b1 to y1_of_b2
            
            # Check if they're on same line (similar Y)
            y_diff = abs(self._get_center_y(b1.bbox) - self._get_center_y(b2.bbox))
            
            if y_diff < 30:  # Same line
                if h_gap > self.max_adjacent_gap:
                    return False
            else:  # Different lines
                if v_gap > self.max_adjacent_gap:
                    return False
        
        return True
    
    def _merge_bboxes(self, bboxes: List[List[List[float]]]) -> List[int]:
        """Merge multiple bboxes into one bounding rectangle."""
        if not bboxes:
            return [0, 0, 0, 0]
        
        all_x = []
        all_y = []
        
        for bbox in bboxes:
            for point in bbox:
                if len(point) >= 2:
                    all_x.append(point[0])
                    all_y.append(point[1])
        
        if not all_x or not all_y:
            return [0, 0, 0, 0]
        
        return [
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x)),
            int(max(all_y)),
        ]
    
    def _get_bbox_rect(self, bbox: List[List[float]]) -> List[int]:
        """Convert polygon bbox to rectangle [x1, y1, x2, y2]."""
        if not bbox or len(bbox) < 4:
            return [0, 0, 0, 0]
        
        all_x = [p[0] for p in bbox if len(p) >= 2]
        all_y = [p[1] for p in bbox if len(p) >= 2]
        
        if not all_x or not all_y:
            return [0, 0, 0, 0]
        
        return [
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x)),
            int(max(all_y)),
        ]
    
    def _get_center_x(self, bbox: List[List[float]]) -> float:
        """Get center X of bbox."""
        if not bbox:
            return 0
        x_vals = [p[0] for p in bbox if len(p) >= 2]
        return sum(x_vals) / len(x_vals) if x_vals else 0
    
    def _get_center_y(self, bbox: List[List[float]]) -> float:
        """Get center Y of bbox."""
        if not bbox:
            return 0
        y_vals = [p[1] for p in bbox if len(p) >= 2]
        return sum(y_vals) / len(y_vals) if y_vals else 0
    
    # ========== Normalizers ==========
    
    def _normalize_default(self, text: str) -> str:
        """Default normalization: remove spaces and lowercase."""
        if not text:
            return ""
        return re.sub(r'\s+', '', text).lower()
    
    def _normalize_invoice_no(self, text: str) -> str:
        """Normalize invoice number: keep only digits."""
        if not text:
            return ""
        return re.sub(r'\D', '', text)
    
    def _normalize_amount(self, text: str) -> str:
        """Normalize amount: keep digits and decimal point."""
        if not text:
            return ""
        # Remove currency symbols, commas, etc.
        normalized = re.sub(r'[^\d.]', '', text)
        # Ensure only one decimal point
        parts = normalized.split('.')
        if len(parts) > 2:
            normalized = parts[0] + '.' + ''.join(parts[1:])
        return normalized
    
    def _normalize_date(self, text: str) -> str:
        """Normalize date: extract MMDD (month and day) for matching.
        
        This is more lenient because OCR might miss year digits.
        E.g., "2025-12-23" -> "1223", "025年12月23日" -> "1223"
        """
        if not text:
            return ""
        # Remove all non-digits
        digits = re.sub(r'\D', '', text)
        # Return last 4 digits (MMDD) if we have enough
        if len(digits) >= 4:
            return digits[-4:]  # Take MMDD part
        return digits
    
    def _normalize_check_code(self, text: str) -> str:
        """Normalize check code: keep alphanumeric characters."""
        if not text:
            return ""
        # Keep letters and digits, convert to lowercase
        return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

