"""
Structure Extractor
===================

Extract structured data from OCR results based on document type.
Uses pattern matching and heuristics for field extraction.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .result import OCRBlock, OCRResult, StructuredData


class StructureExtractor:
    """
    Extract structured fields from OCR results.
    
    Supports multiple document types with type-specific extraction patterns.
    """
    
    # Field patterns for different document types
    PATTERNS = {
        "INVOICE": {
            "invoice_no": [
                r"发票号码[：:\s]*([A-Za-z0-9]+)",
                r"No[.:\s]*([A-Za-z0-9]+)",
                r"发票代码[：:\s]*(\d+)",
            ],
            "invoice_date": [
                r"开票日期[：:\s]*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
                r"Date[：:\s]*(\d{4}[-/]\d{2}[-/]\d{2})",
            ],
            "amount": [
                r"金额[（(]?小写[)）]?[：:\s]*[¥￥]?([\d,]+\.?\d*)",
                r"合计金额[：:\s]*[¥￥]?([\d,]+\.?\d*)",
                r"Amount[：:\s]*[¥￥$]?([\d,]+\.?\d*)",
            ],
            "tax_amount": [
                r"税额[：:\s]*[¥￥]?([\d,]+\.?\d*)",
                r"Tax[：:\s]*[¥￥$]?([\d,]+\.?\d*)",
            ],
            "total_amount": [
                r"价税合计[（(]?小写[)）]?[：:\s]*[¥￥]?([\d,]+\.?\d*)",
                r"Total[：:\s]*[¥￥$]?([\d,]+\.?\d*)",
            ],
            "seller_name": [
                r"销售方[：:\s]*名称[：:\s]*(.+?)(?:\s|$)",
                r"销方[：:\s]*(.+?)(?:\s|$)",
                r"Seller[：:\s]*(.+?)(?:\s|$)",
            ],
            "buyer_name": [
                r"购买方[：:\s]*名称[：:\s]*(.+?)(?:\s|$)",
                r"购方[：:\s]*(.+?)(?:\s|$)",
                r"Buyer[：:\s]*(.+?)(?:\s|$)",
            ],
            # 备注字段 - 使用特殊提取逻辑，这里只定义简单模式
            "remark": [
                # 简单的备注提取 - 复杂逻辑在 _extract_remark 中处理
                r"备\s*注[：:]\s*([^\n]+)",
            ],
        },
        "CONTRACT": {
            "contract_no": [
                r"合同编号[：:\s]*([A-Za-z0-9-]+)",
                r"Contract\s*No[.:\s]*([A-Za-z0-9-]+)",
            ],
            "contract_date": [
                r"签订日期[：:\s]*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
                r"Date[：:\s]*(\d{4}[-/]\d{2}[-/]\d{2})",
            ],
            "party_a": [
                r"甲方[：:\s]*(.+?)(?:\s|乙方|$)",
                r"Party\s*A[：:\s]*(.+?)(?:\s|Party|$)",
            ],
            "party_b": [
                r"乙方[：:\s]*(.+?)(?:\s|甲方|$)",
                r"Party\s*B[：:\s]*(.+?)(?:\s|Party|$)",
            ],
            "amount": [
                r"合同金额[：:\s]*[¥￥]?([\d,]+\.?\d*)",
                r"Contract\s*Amount[：:\s]*[¥￥$]?([\d,]+\.?\d*)",
            ],
        },
        "EXPENSE": {
            "expense_date": [
                r"日期[：:\s]*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",
                r"Date[：:\s]*(\d{4}[-/]\d{2}[-/]\d{2})",
            ],
            "amount": [
                r"金额[：:\s]*[¥￥]?([\d,]+\.?\d*)",
                r"Amount[：:\s]*[¥￥$]?([\d,]+\.?\d*)",
                r"Total[：:\s]*[¥￥$]?([\d,]+\.?\d*)",
            ],
            "vendor": [
                r"商户[：:\s]*(.+?)(?:\s|$)",
                r"Vendor[：:\s]*(.+?)(?:\s|$)",
                r"Store[：:\s]*(.+?)(?:\s|$)",
            ],
            "category": [
                r"类别[：:\s]*(.+?)(?:\s|$)",
                r"Category[：:\s]*(.+?)(?:\s|$)",
            ],
        },
    }
    
    def __init__(self):
        """Initialize extractor."""
        pass
    
    def extract(
        self,
        ocr_result: OCRResult,
        doc_type: str = "AUTO",
    ) -> StructuredData:
        """
        Extract structured data from OCR result.
        
        Args:
            ocr_result: OCR result to process
            doc_type: Document type (AUTO for auto-detection)
            
        Returns:
            StructuredData with extracted fields
        """
        # Auto-detect document type if needed
        if doc_type == "AUTO":
            doc_type = self._detect_doc_type(ocr_result.full_text)
        
        logger.debug(f"Extracting structure for doc_type: {doc_type}")
        
        # Get patterns for document type
        patterns = self.PATTERNS.get(doc_type, {})
        
        # Extract fields
        structured = StructuredData(doc_type=doc_type)
        
        for field_name, field_patterns in patterns.items():
            # 备注字段使用特殊提取逻辑
            if field_name == "remark":
                value, confidence = self._extract_remark(
                    ocr_result.full_text,
                    ocr_result.blocks,
                )
            else:
                value, confidence = self._extract_field(
                    ocr_result.full_text,
                    ocr_result.blocks,
                    field_patterns,
                )
            
            if value:
                structured.fields[field_name] = value
                structured.field_confidences[field_name] = confidence
                
                # Set common fields
                if hasattr(structured, field_name):
                    setattr(structured, field_name, value)
        
        return structured
    
    def _detect_doc_type(self, text: str) -> str:
        """Auto-detect document type from text."""
        text_lower = text.lower()
        
        # Invoice indicators
        invoice_keywords = ["发票", "invoice", "税额", "价税合计", "销售方", "购买方"]
        if any(kw in text_lower for kw in invoice_keywords):
            return "INVOICE"
        
        # Contract indicators
        contract_keywords = ["合同", "contract", "甲方", "乙方", "party a", "party b"]
        if any(kw in text_lower for kw in contract_keywords):
            return "CONTRACT"
        
        # Expense indicators
        expense_keywords = ["费用", "expense", "报销", "receipt", "商户"]
        if any(kw in text_lower for kw in expense_keywords):
            return "EXPENSE"
        
        return "UNKNOWN"
    
    def _extract_remark(
        self,
        full_text: str,
        blocks: List[OCRBlock],
    ) -> Tuple[Optional[str], float]:
        """
        Extract remark (备注) field with enhanced logic.
        
        Handles various formats:
        - Simple: 备注：xxx
        - Structured: 号码：xxx 账期：xxx 收款人：xxx
        - Multi-line remarks
        - Scattered characters: 备 注：xxx
        
        Returns:
            Tuple of (remark_text, confidence)
        """
        remark_parts = []
        confidence = 0.7  # Default confidence
        
        # Normalize text - remove extra spaces between Chinese chars
        text_normalized = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', full_text)
        
        # Pattern 1: Standard remark format (备注：content)
        # Match until end of line or next section marker
        remark_pattern = r'备\s*注[：:]\s*(.+?)(?=\n\s*(?:收款人|复核|开票人|销售方|购买方)|$)'
        match = re.search(remark_pattern, text_normalized, re.DOTALL)
        if match:
            raw_remark = match.group(1).strip()
            # Clean up multiple spaces/newlines
            raw_remark = re.sub(r'\s+', ' ', raw_remark)
            if raw_remark and len(raw_remark) > 1:
                logger.debug(f"Found standard remark: {raw_remark[:50]}...")
        
        # Pattern 2: Extract structured remark fields individually
        # These may appear with or without "备注" label
        
        # 号码 (phone/reference number)
        phone_patterns = [
            r'号\s*码[：:]\s*(\d{5,20})',
            r'联系电话[：:]\s*(\d{5,20})',
            r'电话[：:]\s*(\d{5,20})',
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text_normalized)
            if phone_match:
                remark_parts.append(f"号码：{phone_match.group(1)}")
                logger.debug(f"备注提取 - 号码: {phone_match.group(1)}")
                break
        
        # 账期 (billing period)
        period_patterns = [
            r'账\s*期[：:]\s*([\d]{4,8}(?:[、,，\s]+[\d]{4,8})*)',
            r'期间[：:]\s*([\d]{4,8}(?:[、,，\s]+[\d]{4,8})*)',
        ]
        for pattern in period_patterns:
            period_match = re.search(pattern, text_normalized)
            if period_match:
                period = re.sub(r'[,，\s]+', '、', period_match.group(1)).strip('、')
                remark_parts.append(f"账期：{period}")
                logger.debug(f"备注提取 - 账期: {period}")
                break
        
        # 人工备注 (manual remark)
        manual_match = re.search(r'人\s*工\s*备\s*注[：:]\s*([^;；\n]{1,200})', text_normalized)
        if manual_match:
            manual = manual_match.group(1).strip()
            # Exclude if it's just punctuation
            if manual and not re.match(r'^[;；：:，,。.、\s]+$', manual):
                remark_parts.append(f"人工备注：{manual}")
                logger.debug(f"备注提取 - 人工备注: {manual}")
        
        # 收款人 (payee) - 但排除在备注中的收款人
        # 注意：很多发票的"收款人"是单独的字段，不在备注里
        # 只有当它紧跟"备注"标签时才提取
        remark_payee_match = re.search(
            r'备\s*注[：:].*?收\s*款\s*人[：:]\s*([^\s;；\n]{1,20})',
            text_normalized,
            re.DOTALL
        )
        if remark_payee_match:
            payee = remark_payee_match.group(1).strip()
            if payee and not re.match(r'^[;；：:，,。.、\s]+$', payee):
                remark_parts.append(f"收款人：{payee}")
                logger.debug(f"备注提取 - 收款人: {payee}")
        
        # 复核人 (reviewer) - 同样只在备注区域内提取
        remark_reviewer_match = re.search(
            r'备\s*注[：:].*?复\s*核\s*人?[：:]\s*([^\s;；\n]{1,20})',
            text_normalized,
            re.DOTALL
        )
        if remark_reviewer_match:
            reviewer = remark_reviewer_match.group(1).strip()
            if reviewer and not re.match(r'^[;；：:，,。.、\s]+$', reviewer):
                remark_parts.append(f"复核人：{reviewer}")
                logger.debug(f"备注提取 - 复核人: {reviewer}")
        
        # Pattern 3: Look for remark in OCR blocks (by position - bottom of invoice)
        if not remark_parts and blocks:
            # Sort blocks by Y position (top to bottom)
            sorted_blocks = sorted(blocks, key=lambda b: b.bbox[1] if b.bbox and len(b.bbox) >= 2 else 0)
            
            # Look for "备注" label in bottom half of document
            total_blocks = len(sorted_blocks)
            bottom_half = sorted_blocks[total_blocks // 2:]
            
            for i, block in enumerate(bottom_half):
                block_text = block.text.replace(" ", "")
                if "备注" in block_text or re.search(r'备\s*注', block.text):
                    # Found remark label, get content
                    # Check if content is in the same block
                    remark_content = re.sub(r'^.*?备\s*注[：:]?\s*', '', block.text)
                    if remark_content and len(remark_content) > 1:
                        remark_parts.append(remark_content.strip())
                        confidence = block.confidence
                        logger.debug(f"备注提取(block内): {remark_content[:50]}")
                        break
                    
                    # Or in adjacent blocks
                    if i + 1 < len(bottom_half):
                        next_block = bottom_half[i + 1]
                        # Check if next block is close enough (same line or next line)
                        if next_block.bbox and block.bbox:
                            y_diff = abs(next_block.bbox[1] - block.bbox[1])
                            if y_diff < 50:  # Within 50 pixels vertically
                                remark_parts.append(next_block.text.strip())
                                confidence = next_block.confidence
                                logger.debug(f"备注提取(相邻block): {next_block.text[:50]}")
                                break
        
        # Combine all remark parts
        if remark_parts:
            # Remove duplicates while preserving order
            seen = set()
            unique_parts = []
            for part in remark_parts:
                # Normalize for comparison
                normalized = re.sub(r'[：:;\s]', '', part)
                if normalized not in seen:
                    seen.add(normalized)
                    unique_parts.append(part)
            
            result = "; ".join(unique_parts)
            
            # Final cleanup
            result = re.sub(r'[;；\s]+$', '', result)  # Remove trailing semicolons
            result = re.sub(r'^[;；\s]+', '', result)  # Remove leading semicolons
            
            if result and len(result) > 1:
                logger.info(f"备注提取结果: {result}")
                return result, confidence
        
        # Fallback: Return the raw remark match if found earlier
        if match:
            raw_remark = match.group(1).strip()
            raw_remark = re.sub(r'\s+', ' ', raw_remark)
            if raw_remark and len(raw_remark) > 1:
                return raw_remark, confidence
        
        return None, 0.0
    
    def _extract_field(
        self,
        full_text: str,
        blocks: List[OCRBlock],
        patterns: List[str],
    ) -> Tuple[Optional[str], float]:
        """
        Extract field value using patterns.
        
        Returns:
            Tuple of (value, confidence)
        """
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                # Find confidence from blocks
                confidence = self._find_block_confidence(blocks, value)
                
                return value, confidence
        
        return None, 0.0
    
    def _find_block_confidence(
        self,
        blocks: List[OCRBlock],
        value: str,
    ) -> float:
        """Find confidence for extracted value from blocks."""
        if not value or not blocks:
            return 0.0
        
        # Look for block containing the value
        value_lower = value.lower()
        
        for block in blocks:
            if value_lower in block.text.lower():
                return block.confidence
        
        # If not found exactly, look for partial match
        for block in blocks:
            # Check if any word matches
            block_words = block.text.lower().split()
            value_words = value_lower.split()
            
            for vw in value_words:
                if vw in block_words:
                    return block.confidence
        
        # Default confidence for pattern-matched values
        return 0.7
    
    def add_pattern(
        self,
        doc_type: str,
        field_name: str,
        pattern: str,
    ) -> None:
        """
        Add custom extraction pattern.
        
        Args:
            doc_type: Document type
            field_name: Field name
            pattern: Regex pattern with capture group
        """
        if doc_type not in self.PATTERNS:
            self.PATTERNS[doc_type] = {}
        
        if field_name not in self.PATTERNS[doc_type]:
            self.PATTERNS[doc_type][field_name] = []
        
        self.PATTERNS[doc_type][field_name].append(pattern)
        logger.info(f"Added pattern for {doc_type}.{field_name}")

