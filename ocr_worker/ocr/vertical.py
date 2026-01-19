"""
Vertical Text Handler
=====================

Handles vertical (rotated 90 degrees) text in Chinese invoices.
Specifically targets "购买方信息" and "销售方信息" sections.
"""

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    from PIL import Image
    import cv2
    HAS_CV2 = True
except ImportError:
    logger.warning("PIL or cv2 not available, vertical text detection disabled")
    HAS_CV2 = False


@dataclass
class VerticalRegion:
    """Detected vertical text region."""
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    region_type: str = "unknown"  # buyer, seller, remark


class VerticalTextHandler:
    """
    Handles vertical text detection and recognition.
    
    Chinese invoices often have "购买方信息" and "销售方信息" 
    displayed vertically (rotated 90 degrees).
    
    Strategy:
    1. Detect vertical text regions (height >> width)
    2. Rotate regions 90 degrees
    3. Re-run OCR on rotated regions
    4. Merge results with original OCR
    """
    
    # Vertical text markers
    VERTICAL_MARKERS = {
        "buyer": ["购买方信息", "购买方", "购方信息", "购方"],
        "seller": ["销售方信息", "销售方", "销方信息", "销方"],
        "remark": ["备注", "备 注"],
    }
    
    # Aspect ratio threshold for vertical text detection
    MIN_ASPECT_RATIO = 2.0  # height / width > 2 is likely vertical
    
    def __init__(self, ocr_engine=None):
        """
        Initialize handler.
        
        Args:
            ocr_engine: OCR engine instance for re-recognition
        """
        self._ocr = ocr_engine
    
    def detect_vertical_regions(
        self,
        image: "Image.Image",
        blocks: List[dict],
    ) -> List[VerticalRegion]:
        """
        Detect vertical text regions from OCR blocks.
        
        Args:
            image: PIL Image
            blocks: OCR result blocks with bbox
            
        Returns:
            List of detected vertical regions
        """
        regions = []
        
        for block in blocks:
            bbox = block.get("bbox") or block.get("box", [])
            text = block.get("text", "")
            
            if not bbox or len(bbox) < 4:
                continue
            
            # Calculate bounding box dimensions
            if isinstance(bbox[0], list):
                # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            else:
                # [x1, y1, x2, y2] format
                x1, y1, x2, y2 = bbox[:4]
            
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                continue
            
            aspect_ratio = height / width
            
            # Check if this is a vertical text region
            if aspect_ratio >= self.MIN_ASPECT_RATIO:
                region_type = self._classify_region(text)
                if region_type:
                    regions.append(VerticalRegion(
                        x=int(x1),
                        y=int(y1),
                        width=int(width),
                        height=int(height),
                        text=text,
                        region_type=region_type,
                    ))
                    logger.debug(
                        f"Detected vertical region: type={region_type}, "
                        f"bbox=({x1}, {y1}, {x2}, {y2}), text={text[:20]}..."
                    )
        
        return regions
    
    def _classify_region(self, text: str) -> Optional[str]:
        """Classify region type based on text content."""
        text_clean = text.replace(" ", "").replace("\n", "")
        
        for region_type, markers in self.VERTICAL_MARKERS.items():
            for marker in markers:
                if marker in text_clean:
                    return region_type
                # Also check for scattered characters
                if self._match_scattered(marker, text):
                    return region_type
        
        return None
    
    def _match_scattered(self, marker: str, text: str) -> bool:
        """
        Match marker with scattered characters.
        e.g., "购买方信息" might appear as "购 买 方 信 息"
        """
        pattern = r"\s*".join(marker)
        return bool(re.search(pattern, text))
    
    def recognize_vertical_regions(
        self,
        image: "Image.Image",
        regions: List[VerticalRegion],
    ) -> Dict[str, str]:
        """
        Recognize vertical text by rotating regions and re-OCR.
        
        Args:
            image: Original image
            regions: Detected vertical regions
            
        Returns:
            Dict of recognized fields
        """
        if not self._ocr or not regions:
            return {}
        
        results = {}
        img_array = np.array(image)
        
        for region in regions:
            try:
                # Crop region with some padding
                pad = 5
                y1 = max(0, region.y - pad)
                y2 = min(img_array.shape[0], region.y + region.height + pad)
                x1 = max(0, region.x - pad)
                x2 = min(img_array.shape[1], region.x + region.width + pad)
                
                crop = img_array[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Rotate 90 degrees counterclockwise
                rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Save to temporary file for OCR (since OCREngine expects file path)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
                
                try:
                    # Run OCR on rotated image
                    ocr_result = self._ocr.recognize(tmp_path, mode="text")
                    
                    if ocr_result and ocr_result.full_text:
                        recognized_text = ocr_result.full_text.strip()
                        
                        # Extract specific fields based on region type
                        if region.region_type == "buyer":
                            fields = self._extract_party_fields(recognized_text, "buyer")
                            results.update(fields)
                        elif region.region_type == "seller":
                            fields = self._extract_party_fields(recognized_text, "seller")
                            results.update(fields)
                        elif region.region_type == "remark":
                            remark = self._extract_remark_fields(recognized_text)
                            if remark:
                                results["remark"] = remark
                        
                        logger.info(
                            f"Recognized vertical {region.region_type}: "
                            f"{recognized_text[:50]}..."
                        )
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            except Exception as e:
                logger.warning(f"Failed to recognize vertical region: {e}")
        
        return results
    
    def _extract_party_fields(
        self,
        text: str,
        party_type: str,
    ) -> Dict[str, str]:
        """
        Extract name and tax number from party text.
        
        Args:
            text: Recognized text
            party_type: "buyer" or "seller"
            
        Returns:
            Dict with extracted fields
        """
        fields = {}
        prefix = party_type  # buyerName, buyerTaxNo, sellerName, sellerTaxNo
        
        # Remove scattered spaces (common in vertical OCR)
        text_clean = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        
        # Extract name - enhanced patterns for different invoice formats
        name_patterns = [
            # Standard format: 名称：XXX公司
            r"名\s*称[：:]\s*(.+?)(?=统一社会|纳税人|识别号|地址|电话|开户|$)",
            r"名称[：:]\s*(.+?)(?=\s{2,}|$)",
            # Without label: company name followed by tax info
            r"^(.+?公司|.+?有限公司|.+?集团)(?=\s|统一社会|纳税人|$)",
            # Fallback: first line before any identifier
            r"^(.+?)(?=统一社会|纳税人|识别号)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text_clean, re.DOTALL | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Clean up name: remove internal spaces, newlines
                name = re.sub(r"[\s\n]+", "", name)
                # Remove trailing punctuation
                name = re.sub(r"[：:，,。.]+$", "", name)
                if name and len(name) >= 2:  # Valid name should have at least 2 chars
                    fields[f"{prefix}Name"] = name
                    break
        
        # Extract tax number - enhanced patterns
        tax_patterns = [
            # Full format: 统一社会信用代码/纳税人识别号：91XXXXX
            r"统一社会信用代码[/／]?纳税人识别号[：:]\s*([A-Za-z0-9]{15,20})",
            # Short format: 纳税人识别号：91XXXXX
            r"纳税人识别号[：:]\s*([A-Za-z0-9]{15,20})",
            # Even shorter: 识别号：91XXXXX
            r"识别号[：:]\s*([A-Za-z0-9]{15,20})",
            # Pattern without label: 18-digit unified social credit code
            r"(?<![A-Za-z0-9])([A-Za-z0-9]{18})(?![A-Za-z0-9])",
            # Pattern without label: 15-digit tax ID
            r"(?<![A-Za-z0-9])([A-Za-z0-9]{15})(?![A-Za-z0-9])",
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text_clean)
            if match:
                tax_no = match.group(1).strip().upper()
                # Validate: should start with number or letter
                if tax_no and re.match(r'^[A-Z0-9]', tax_no):
                    # Exclude false positives like phone numbers
                    if not re.match(r'^1[3-9]\d{9}$', tax_no):  # Not a phone number
                        fields[f"{prefix}TaxNo"] = tax_no
                        break
        
        # Extract address and phone (optional fields)
        addr_phone_pattern = r"地\s*址[,，]?\s*电\s*话[：:]\s*(.+?)(?=开户|$)"
        match = re.search(addr_phone_pattern, text_clean, re.DOTALL)
        if match:
            addr_phone = match.group(1).strip()
            fields[f"{prefix}AddressPhone"] = re.sub(r"\s+", " ", addr_phone)
        
        # Extract bank account (optional field)
        bank_pattern = r"开户[行银]?及?\s*账\s*号[：:]\s*(.+?)$"
        match = re.search(bank_pattern, text_clean, re.DOTALL)
        if match:
            bank_account = match.group(1).strip()
            fields[f"{prefix}BankAccount"] = re.sub(r"\s+", " ", bank_account)
        
        return fields
    
    def _extract_remark_fields(self, text: str) -> Optional[str]:
        """
        Extract and format remark content from recognized text.
        
        Remark section may contain:
        - 号码：15895935827
        - 账期：202504、202505、202506
        - 人工备注：xxx
        - 收款人：xxx
        - 复核人：xxx
        
        Args:
            text: Recognized remark text
            
        Returns:
            Formatted remark string
        """
        if not text:
            return None
        
        remark_parts = []
        text_clean = re.sub(r'\s+', ' ', text.strip())
        
        # Extract phone number
        phone_match = re.search(r'号\s*码[：:]\s*([0-9]{5,20})', text_clean)
        if phone_match:
            remark_parts.append(f"号码：{phone_match.group(1)}")
            logger.debug(f"备注提取 - 号码: {phone_match.group(1)}")
        
        # Extract account period
        period_match = re.search(r'账\s*期[：:]\s*([\d、,，\s]+)', text_clean)
        if period_match:
            period = re.sub(r'[,，\s]+', '、', period_match.group(1)).strip('、')
            remark_parts.append(f"账期：{period}")
            logger.debug(f"备注提取 - 账期: {period}")
        
        # Extract manual remark
        manual_match = re.search(r'人\s*工\s*备\s*注[：:]\s*([^;；\n]*)', text_clean)
        if manual_match:
            manual = manual_match.group(1).strip()
            if manual and not re.match(r'^[;；：:，,。.、\s]+$', manual):
                remark_parts.append(f"人工备注：{manual}")
                logger.debug(f"备注提取 - 人工备注: {manual}")
        
        # Extract receiver
        receiver_match = re.search(r'收\s*款\s*人[：:]\s*([^;；\n\s复]+)', text_clean)
        if receiver_match:
            receiver = receiver_match.group(1).strip()
            if receiver:
                remark_parts.append(f"收款人：{receiver}")
                logger.debug(f"备注提取 - 收款人: {receiver}")
        
        # Extract reviewer
        reviewer_match = re.search(r'复\s*核\s*人[：:]\s*([^;；\n\s开]+)', text_clean)
        if reviewer_match:
            reviewer = reviewer_match.group(1).strip()
            if reviewer:
                remark_parts.append(f"复核人：{reviewer}")
                logger.debug(f"备注提取 - 复核人: {reviewer}")
        
        if remark_parts:
            result = "; ".join(remark_parts)
            logger.info(f"备注提取结果: {result}")
            return result
        
        # If no structured content found, clean and return raw text
        # Remove scattered "备" and "注" characters
        cleaned = re.sub(r'^[备注\s]+', '', text_clean)
        cleaned = re.sub(r'[备注\s]+$', '', cleaned)
        cleaned = re.sub(r'\s+[备注]\s+', ' ', cleaned)
        
        if cleaned and not re.match(r'^[;；：:，,。.、\s]+$', cleaned):
            logger.info(f"备注提取(原文): {cleaned[:100]}")
            return cleaned
        
        return None
    
    def process_image(
        self,
        image: "Image.Image",
        ocr_blocks: List[dict],
    ) -> Dict[str, str]:
        """
        Full pipeline: detect and recognize vertical text.
        
        Args:
            image: PIL Image
            ocr_blocks: Initial OCR result blocks
            
        Returns:
            Dict of fields extracted from vertical regions
        """
        if not HAS_CV2:
            logger.debug("CV2 not available, skipping vertical text processing")
            return {}
        
        # Step 1: Detect vertical regions
        regions = self.detect_vertical_regions(image, ocr_blocks)
        
        if not regions:
            logger.debug("No vertical text regions detected")
            return {}
        
        logger.info(f"Detected {len(regions)} vertical text regions")
        
        # Step 2: Recognize rotated regions
        fields = self.recognize_vertical_regions(image, regions)
        
        return fields


def merge_vertical_fields(
    original_fields: Dict[str, dict],
    vertical_fields: Dict[str, str],
) -> Dict[str, dict]:
    """
    Merge vertical text fields with original OCR fields.
    
    Vertical text recognition takes precedence for buyer/seller fields
    if original fields are missing or low confidence.
    
    For remark fields, intelligently merge both sources.
    
    Args:
        original_fields: Original extracted fields
        vertical_fields: Fields from vertical text recognition
        
    Returns:
        Merged fields dict
    """
    result = dict(original_fields)
    
    for field_key, value in vertical_fields.items():
        if not value:
            continue
        
        # Check if original field exists and has good confidence
        original = result.get(field_key)
        
        if original is None:
            # Add new field from vertical recognition
            result[field_key] = {
                "value": value,
                "source": "VERTICAL_OCR",
                "confidence": 0.85,  # Default confidence for vertical OCR
            }
            logger.info(f"Added vertical field: {field_key}={value}")
        
        elif isinstance(original, dict):
            original_conf = original.get("confidence", 0)
            original_value = original.get("value", "")
            
            # Special handling for remark field - merge instead of replace
            if field_key == "remark":
                merged_remark = _merge_remark_values(original_value, value)
                if merged_remark != original_value:
                    result[field_key] = {
                        "value": merged_remark,
                        "source": "MERGED",
                        "confidence": max(original_conf, 0.85),
                    }
                    logger.info(f"Merged remark: '{original_value}' + '{value}' -> '{merged_remark}'")
            # Replace if original is empty or low confidence
            elif not original_value or original_conf < 0.7:
                result[field_key] = {
                    "value": value,
                    "source": "VERTICAL_OCR",
                    "confidence": 0.85,
                }
                logger.info(
                    f"Replaced field {field_key}: "
                    f"'{original_value}' -> '{value}'"
                )
    
    return result


def _merge_remark_values(original: str, vertical: str) -> str:
    """
    Intelligently merge remark values from different sources.
    
    - Extract structured parts (号码, 账期, 收款人, 复核人) from both
    - Avoid duplicates
    - Combine into a clean result
    
    Args:
        original: Original remark value (from horizontal text)
        vertical: Vertical remark value (from rotated text OCR)
        
    Returns:
        Merged remark string
    """
    if not original and not vertical:
        return ""
    if not original:
        return vertical
    if not vertical:
        return original
    
    # Parse structured fields from both sources
    def extract_fields(text: str) -> Dict[str, str]:
        fields = {}
        
        # 号码
        phone_match = re.search(r'号\s*码[：:]\s*(\d{5,20})', text)
        if phone_match:
            fields['号码'] = phone_match.group(1)
        
        # 账期
        period_match = re.search(r'账\s*期[：:]\s*([\d、,，\s]+)', text)
        if period_match:
            fields['账期'] = re.sub(r'[,，\s]+', '、', period_match.group(1)).strip('、')
        
        # 人工备注
        manual_match = re.search(r'人\s*工\s*备\s*注[：:]\s*([^;；\n]{1,200})', text)
        if manual_match:
            fields['人工备注'] = manual_match.group(1).strip()
        
        # 收款人
        payee_match = re.search(r'收\s*款\s*人[：:]\s*([^\s;；\n]{1,20})', text)
        if payee_match:
            fields['收款人'] = payee_match.group(1).strip()
        
        # 复核人
        reviewer_match = re.search(r'复\s*核\s*人?[：:]\s*([^\s;；\n]{1,20})', text)
        if reviewer_match:
            fields['复核人'] = reviewer_match.group(1).strip()
        
        return fields
    
    # Extract fields from both sources
    original_fields = extract_fields(original)
    vertical_fields = extract_fields(vertical)
    
    # Merge: vertical takes precedence for missing fields
    merged_fields = {**original_fields}
    for key, value in vertical_fields.items():
        if key not in merged_fields or not merged_fields[key]:
            merged_fields[key] = value
        elif merged_fields[key] != value:
            # If values differ, prefer longer/more complete one
            if len(value) > len(merged_fields[key]):
                merged_fields[key] = value
    
    # Build result string
    parts = []
    field_order = ['号码', '账期', '人工备注', '收款人', '复核人']
    
    for field_name in field_order:
        if field_name in merged_fields and merged_fields[field_name]:
            parts.append(f"{field_name}：{merged_fields[field_name]}")
    
    if parts:
        return "; ".join(parts)
    
    # If no structured fields found, return the longer value
    return original if len(original) >= len(vertical) else vertical

