"""
Row Clustering Algorithm (Enhanced)
====================================

Groups OCR blocks into logical rows based on Y-coordinate clustering,
then sorts by X within each row for proper reading order.

Features:
- Adaptive Y-tolerance based on line height
- Projection-based row/column line detection
- Table header/body/footer segmentation
- Dynamic column interval estimation

This enables structured table extraction from scattered OCR text blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger
import numpy as np


@dataclass
class RowItem:
    """Single item within a row with position info."""
    
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    # Column assignment (for table extraction)
    column_index: int = -1
    column_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "xCenter": round(self.x_center, 2),
            "yCenter": round(self.y_center, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "columnIndex": self.column_index,
            "columnName": self.column_name,
        }


@dataclass
class TableRow:
    """A logical row containing multiple items sorted by X position."""
    
    row_index: int
    items: List[RowItem] = field(default_factory=list)
    y_min: float = 0.0
    y_max: float = 0.0
    y_center: float = 0.0
    
    # For table extraction
    row_type: str = "data"  # header, data, subtotal, total, footer
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rowIndex": self.row_index,
            "items": [item.to_dict() for item in self.items],
            "yMin": round(self.y_min, 2),
            "yMax": round(self.y_max, 2),
            "yCenter": round(self.y_center, 2),
            "rowType": self.row_type,
            "text": " ".join(item.text for item in self.items),
        }
    
    def get_text_at_column(self, col_idx: int) -> str:
        """Get text at specific column index."""
        for item in self.items:
            if item.column_index == col_idx:
                return item.text
        return ""


@dataclass
class TableStructure:
    """Complete table structure with rows and columns."""
    
    rows: List[TableRow] = field(default_factory=list)
    column_headers: List[str] = field(default_factory=list)
    column_positions: List[Tuple[float, float]] = field(default_factory=list)  # (x_min, x_max)
    
    # Table metrics
    row_count: int = 0
    column_count: int = 0
    header_row_index: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "columnHeaders": self.column_headers,
            "columnPositions": [(round(x[0], 2), round(x[1], 2)) for x in self.column_positions],
            "rowCount": self.row_count,
            "columnCount": self.column_count,
            "headerRowIndex": self.header_row_index,
        }


class RowClusteringEngine:
    """
    Clusters OCR blocks into rows based on Y-coordinate proximity.
    
    Algorithm:
    1. Extract center coordinates and dimensions from bboxes
    2. Cluster blocks by Y-coordinate using adaptive threshold
    3. Sort blocks within each row by X-coordinate
    4. Optionally detect column structure from header row
    """
    
    def __init__(
        self,
        y_tolerance_ratio: float = 0.5,  # Tolerance as ratio of avg block height
        min_row_gap: float = 5.0,         # Minimum pixels between rows
        column_gap_ratio: float = 1.5,    # Gap ratio for column detection
    ):
        self.y_tolerance_ratio = y_tolerance_ratio
        self.min_row_gap = min_row_gap
        self.column_gap_ratio = column_gap_ratio
    
    def _split_vertical_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Split vertically-stacked text blocks using anchor-based alignment.
        
        Instead of evenly distributing split characters, we use the Y positions
        of OTHER blocks (anchors) to determine where each character should go.
        
        This solves the problem where OCR recognizes the entire "单位" column
        (件件件件...) as one block, but we need each "件" aligned to its row.
        
        Args:
            blocks: Original OCR blocks
            
        Returns:
            Processed blocks with vertical blocks split and aligned
        """
        if not blocks:
            return blocks
        
        # First pass: identify vertical blocks and collect anchor Y positions
        vertical_blocks = []
        anchor_blocks = []
        
        # Calculate statistics
        heights = []
        for block in blocks:
            bbox = block.get("bbox", [[0, 0]] * 4)
            if bbox and len(bbox) >= 4:
                ys = [p[1] for p in bbox]
                heights.append(max(ys) - min(ys))
        
        if not heights:
            return blocks
        
        heights_sorted = sorted(heights)
        avg_height = heights_sorted[len(heights_sorted) // 2] if heights_sorted else 20.0
        
        # Classify blocks
        for block in blocks:
            bbox = block.get("bbox", [[0, 0]] * 4)
            text = block.get("text", "")
            
            if not bbox or len(bbox) < 4 or not text:
                anchor_blocks.append(block)
                continue
            
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            
            # Check if vertical block with repeated chars
            is_vertical = (
                height > 2.5 * avg_height and
                height > 2 * width and
                len(text) > 1
            )
            unique_chars = set(text.replace(" ", ""))
            has_repeated_chars = len(unique_chars) <= 2 and len(text) > 2
            
            if is_vertical and has_repeated_chars:
                clean_text = text.replace(" ", "")
                logger.info(f"Detected vertical block: '{clean_text}' ({len(clean_text)} chars), height={height:.0f}, width={width:.0f}")
                vertical_blocks.append({
                    "block": block,
                    "x_center": (min(xs) + max(xs)) / 2,
                    "y_min": min(ys),
                    "y_max": max(ys),
                    "text": clean_text,
                })
            else:
                anchor_blocks.append(block)
        
        # If no vertical blocks, return original
        if not vertical_blocks:
            return blocks
        
        # For each vertical block, find anchor Y positions from nearby blocks
        # This is done per-vertical-block to ensure correct alignment
        logger.debug(f"Processing {len(vertical_blocks)} vertical block(s)")
        
        # Split vertical blocks and align to anchor rows
        result = list(anchor_blocks)
        split_count = 0
        
        for vb in vertical_blocks:
            block = vb["block"]
            bbox = block.get("bbox", [[0, 0]] * 4)
            xs = [p[0] for p in bbox]
            x_min, x_max = min(xs), max(xs)
            x_center = vb["x_center"]
            clean_text = vb["text"]
            char_count = len(clean_text)
            
            # Find anchor Y positions from blocks that are:
            # 1. Within the Y range of this vertical block
            # 2. NOT overlapping with the X range of this vertical block (i.e., different columns)
            anchor_y_centers = []
            for anchor_block in anchor_blocks:
                anchor_bbox = anchor_block.get("bbox", [[0, 0]] * 4)
                if not anchor_bbox or len(anchor_bbox) < 4:
                    continue
                
                anchor_xs = [p[0] for p in anchor_bbox]
                anchor_ys = [p[1] for p in anchor_bbox]
                anchor_y_center = sum(anchor_ys) / len(anchor_ys)
                anchor_x_center = sum(anchor_xs) / len(anchor_xs)
                
                # Check if within Y range of vertical block (with tolerance)
                if not (vb["y_min"] - avg_height <= anchor_y_center <= vb["y_max"] + avg_height):
                    continue
                
                # Check if in different column (X distance > some threshold)
                # This ensures we're using items from the same table rows
                x_distance = abs(anchor_x_center - x_center)
                if x_distance < avg_height * 2:  # Too close horizontally, might be same column
                    continue
                
                anchor_y_centers.append(anchor_y_center)
            
            # Cluster Y positions to get distinct rows
            anchor_y_centers.sort()
            rows_in_range = []
            if anchor_y_centers:
                rows_in_range.append(anchor_y_centers[0])
                for y in anchor_y_centers[1:]:
                    if y - rows_in_range[-1] > avg_height * 0.4:  # Tighter threshold
                        rows_in_range.append(y)
            
            logger.debug(f"Vertical block '{clean_text[:5]}...' ({char_count} chars): found {len(rows_in_range)} anchor rows in Y range [{vb['y_min']:.0f}, {vb['y_max']:.0f}]")
            
            if rows_in_range and len(rows_in_range) >= char_count:
                # Use anchor positions - assign one character to each row
                # Take only the first char_count rows
                rows_to_use = rows_in_range[:char_count]
                
                for i, char in enumerate(clean_text):
                    row_y = rows_to_use[i]
                    half_height = avg_height / 2
                    
                    new_bbox = [
                        [x_min, row_y - half_height],
                        [x_max, row_y - half_height],
                        [x_max, row_y + half_height],
                        [x_min, row_y + half_height],
                    ]
                    
                    result.append({
                        "text": char,
                        "confidence": block.get("confidence", 0.0),
                        "bbox": new_bbox,
                    })
                
                split_count += 1
                logger.info(f"Split vertical '{clean_text[:5]}...' ({char_count} chars) -> aligned to {len(rows_to_use)} rows")
            else:
                # Fallback: evenly distribute based on character count
                y_min, y_max = vb["y_min"], vb["y_max"]
                height = y_max - y_min
                char_height = height / char_count
                
                for i, char in enumerate(clean_text):
                    row_y_center = y_min + (i + 0.5) * char_height
                    half_height = char_height / 2
                    
                    new_bbox = [
                        [x_min, row_y_center - half_height],
                        [x_max, row_y_center - half_height],
                        [x_max, row_y_center + half_height],
                        [x_min, row_y_center + half_height],
                    ]
                    
                    result.append({
                        "text": char,
                        "confidence": block.get("confidence", 0.0),
                        "bbox": new_bbox,
                    })
                
                split_count += 1
                logger.info(f"Split vertical '{clean_text[:5]}...' ({char_count} chars) -> even distribution (only {len(rows_in_range)} anchors found)")
        
        if split_count > 0:
            logger.info(f"Vertical block split: {split_count} block(s), {len(blocks)} -> {len(result)} blocks")
        
        return result
    
    def cluster_blocks(self, blocks: List[Dict]) -> List[TableRow]:
        """
        Cluster blocks into rows.
        
        Args:
            blocks: List of OCR blocks with text, confidence, bbox
            
        Returns:
            List of TableRow sorted by Y position
        """
        if not blocks:
            return []
        
        # Preprocess: split vertical blocks that span multiple rows
        processed_blocks = self._split_vertical_blocks(blocks)
        
        # Convert blocks to RowItems with computed metrics
        items = []
        for block in processed_blocks:
            bbox = block.get("bbox", [[0, 0]] * 4)
            if not bbox or len(bbox) < 4:
                continue
            
            # Compute center and dimensions
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            
            x_center = sum(xs) / 4
            y_center = sum(ys) / 4
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            
            item = RowItem(
                text=block.get("text", ""),
                confidence=block.get("confidence", 0.0),
                bbox=bbox,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
            items.append(item)
        
        if not items:
            return []
        
        # Calculate adaptive Y tolerance
        avg_height = np.mean([item.height for item in items if item.height > 0]) or 20.0
        y_tolerance = max(avg_height * self.y_tolerance_ratio, self.min_row_gap)
        
        logger.debug(f"Row clustering: {len(items)} items, avg_height={avg_height:.1f}, y_tolerance={y_tolerance:.1f}")
        
        # Sort by Y center
        items_sorted = sorted(items, key=lambda x: x.y_center)
        
        # Cluster into rows using moving average Y position
        rows: List[TableRow] = []
        current_row_items: List[RowItem] = []
        current_y_sum = 0.0
        
        for item in items_sorted:
            if not current_row_items:
                # First item in row
                current_row_items.append(item)
                current_y_sum = item.y_center
            else:
                # Calculate current row's average Y
                current_y_avg = current_y_sum / len(current_row_items)
                
                if abs(item.y_center - current_y_avg) <= y_tolerance:
                    # Same row - add item and update running sum
                    current_row_items.append(item)
                    current_y_sum += item.y_center
                else:
                    # New row - save current row and start new one
                    rows.append(self._create_row(len(rows), current_row_items))
                    current_row_items = [item]
                    current_y_sum = item.y_center
        
        # Don't forget last row
        if current_row_items:
            rows.append(self._create_row(len(rows), current_row_items))
        
        logger.info(f"Row clustering: {len(blocks)} blocks -> {len(rows)} rows")
        
        return rows
    
    def _create_row(self, row_index: int, items: List[RowItem]) -> TableRow:
        """Create a TableRow from items, sorting by X position."""
        # Sort items by X center
        items_sorted = sorted(items, key=lambda x: x.x_center)
        
        # Calculate row bounds
        y_values = [item.y_center for item in items_sorted]
        y_min = min(y_values) if y_values else 0
        y_max = max(y_values) if y_values else 0
        y_center = np.mean(y_values) if y_values else 0
        
        return TableRow(
            row_index=row_index,
            items=items_sorted,
            y_min=y_min,
            y_max=y_max,
            y_center=y_center,
        )
    
    def detect_columns(self, rows: List[TableRow], header_keywords: List[str] = None) -> TableStructure:
        """
        Detect column structure from rows.
        
        Args:
            rows: Clustered rows
            header_keywords: Keywords to identify header row
            
        Returns:
            TableStructure with column assignments
        """
        if not rows:
            return TableStructure()
        
        # Default invoice header keywords
        if header_keywords is None:
            header_keywords = [
                "项目名称", "品名", "名称", "货物名称", "服务名称",
                "规格", "型号", "规格型号",
                "单位", "数量", "单价", "金额",
                "税率", "税额", "含税", "不含税",
            ]
        
        # Find header row
        header_row_idx = -1
        header_row = None
        for idx, row in enumerate(rows[:5]):  # Check first 5 rows
            row_text = " ".join(item.text for item in row.items)
            keyword_hits = sum(1 for kw in header_keywords if kw in row_text)
            if keyword_hits >= 2:
                header_row_idx = idx
                header_row = row
                row.row_type = "header"
                break
        
        # Determine column positions from header
        column_positions = []
        column_headers = []
        
        if header_row:
            for item in header_row.items:
                x_min = min(p[0] for p in item.bbox)
                x_max = max(p[0] for p in item.bbox)
                column_positions.append((x_min, x_max))
                column_headers.append(item.text.strip())
        else:
            # Estimate columns from first data row with most items
            max_items_row = max(rows, key=lambda r: len(r.items))
            for item in max_items_row.items:
                x_min = min(p[0] for p in item.bbox)
                x_max = max(p[0] for p in item.bbox)
                column_positions.append((x_min, x_max))
                column_headers.append("")
        
        # Assign items to columns
        for row in rows:
            if row.row_type == "header":
                continue
            
            for item in row.items:
                col_idx = self._find_best_column(item.x_center, column_positions)
                item.column_index = col_idx
                if col_idx >= 0 and col_idx < len(column_headers):
                    item.column_name = column_headers[col_idx]
            
            # Detect row type
            row_text_lower = " ".join(item.text.lower() for item in row.items)
            if "合计" in row_text_lower or "小计" in row_text_lower:
                row.row_type = "subtotal"
            elif "价税合计" in row_text_lower or "总计" in row_text_lower:
                row.row_type = "total"
            elif "备注" in row_text_lower or "开票人" in row_text_lower:
                row.row_type = "footer"
            else:
                row.row_type = "data"
        
        return TableStructure(
            rows=rows,
            column_headers=column_headers,
            column_positions=column_positions,
            row_count=len(rows),
            column_count=len(column_headers),
            header_row_index=header_row_idx,
        )
    
    def _find_best_column(self, x_center: float, column_positions: List[Tuple[float, float]]) -> int:
        """Find the best matching column for an x position."""
        if not column_positions:
            return -1
        
        best_col = -1
        best_dist = float("inf")
        
        for idx, (x_min, x_max) in enumerate(column_positions):
            col_center = (x_min + x_max) / 2
            col_width = x_max - x_min
            
            # Check if within column bounds (with tolerance)
            tolerance = col_width * 0.5
            if x_min - tolerance <= x_center <= x_max + tolerance:
                dist = abs(x_center - col_center)
                if dist < best_dist:
                    best_dist = dist
                    best_col = idx
        
        # If not within any column, find nearest
        if best_col < 0:
            for idx, (x_min, x_max) in enumerate(column_positions):
                col_center = (x_min + x_max) / 2
                dist = abs(x_center - col_center)
                if dist < best_dist:
                    best_dist = dist
                    best_col = idx
        
        return best_col


def cluster_ocr_blocks_to_rows(blocks: List[Dict]) -> List[Dict]:
    """
    Convenience function to cluster OCR blocks into rows.
    
    Args:
        blocks: List of OCR blocks with text, confidence, bbox
        
    Returns:
        List of row dictionaries
    """
    engine = RowClusteringEngine()
    rows = engine.cluster_blocks(blocks)
    return [row.to_dict() for row in rows]


def extract_table_structure(blocks: List[Dict], header_keywords: List[str] = None) -> Dict:
    """
    Extract table structure from OCR blocks.
    
    Args:
        blocks: List of OCR blocks
        header_keywords: Optional list of header keywords
        
    Returns:
        TableStructure as dictionary
    """
    engine = RowClusteringEngine()
    rows = engine.cluster_blocks(blocks)
    structure = engine.detect_columns(rows, header_keywords)
    return structure.to_dict()


# ============================================================================
# Advanced Features: Projection-based Analysis
# ============================================================================

class ProjectionAnalyzer:
    """
    Projection-based row/column line detection.
    
    Uses horizontal and vertical projection histograms to detect
    table structure lines, similar to traditional document analysis.
    """
    
    def __init__(self, image_width: int = 0, image_height: int = 0):
        self.image_width = image_width
        self.image_height = image_height
    
    def compute_projections(self, blocks: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute horizontal and vertical projection histograms.
        
        Args:
            blocks: OCR blocks with bbox
            
        Returns:
            (horizontal_projection, vertical_projection)
        """
        if not blocks:
            return np.array([]), np.array([])
        
        # Determine image bounds from blocks
        all_y = []
        all_x = []
        for block in blocks:
            bbox = block.get("bbox", [])
            if not bbox:
                continue
            for point in bbox:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    all_x.append(point[0])
                    all_y.append(point[1])
        
        if not all_x or not all_y:
            return np.array([]), np.array([])
        
        width = int(max(all_x)) + 1
        height = int(max(all_y)) + 1
        
        # Create projection arrays
        h_proj = np.zeros(height)
        v_proj = np.zeros(width)
        
        # Accumulate projections
        for block in blocks:
            bbox = block.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            
            # Horizontal projection (row detection)
            h_proj[y_min:y_max+1] += 1
            
            # Vertical projection (column detection)
            v_proj[x_min:x_max+1] += 1
        
        return h_proj, v_proj
    
    def find_row_gaps(self, h_proj: np.ndarray, min_gap: int = 5) -> List[int]:
        """Find row separator positions from horizontal projection."""
        if len(h_proj) == 0:
            return []
        
        gaps = []
        in_gap = True
        gap_start = 0
        
        for i, val in enumerate(h_proj):
            if val == 0:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap and (i - gap_start) >= min_gap:
                    gaps.append((gap_start + i) // 2)
                in_gap = False
        
        return gaps
    
    def find_column_boundaries(self, v_proj: np.ndarray, threshold_ratio: float = 0.3) -> List[Tuple[int, int]]:
        """Find column boundaries from vertical projection."""
        if len(v_proj) == 0:
            return []
        
        threshold = np.max(v_proj) * threshold_ratio
        
        columns = []
        in_column = False
        col_start = 0
        
        for i, val in enumerate(v_proj):
            if val > threshold:
                if not in_column:
                    col_start = i
                    in_column = True
            else:
                if in_column:
                    columns.append((col_start, i))
                    in_column = False
        
        # Handle last column
        if in_column:
            columns.append((col_start, len(v_proj) - 1))
        
        return columns


class AdaptiveColumnDetector:
    """
    Adaptive column detection using multiple strategies.
    
    Combines projection analysis, header detection, and gap analysis
    to determine column boundaries that adapt to different invoice formats.
    """
    
    # Common invoice column headers for matching
    INVOICE_HEADERS = {
        "项目名称": ["项目名称", "品名", "货物名称", "服务名称", "商品名称", "名称"],
        "规格型号": ["规格型号", "规格", "型号"],
        "单位": ["单位", "计量单位"],
        "数量": ["数量", "数", "件数"],
        "单价": ["单价", "不含税单价", "价格"],
        "金额": ["金额", "不含税金额", "合计金额"],
        "税率": ["税率", "征收率", "税率/征收率"],
        "税额": ["税额", "增值税额"],
    }
    
    def __init__(self):
        self.column_order = list(self.INVOICE_HEADERS.keys())
    
    def detect_columns_adaptive(
        self, 
        rows: List[TableRow], 
        image_width: int = 0
    ) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Adaptively detect columns using multiple strategies.
        
        Args:
            rows: Clustered rows
            image_width: Image width for normalization
            
        Returns:
            (column_headers, column_positions)
        """
        if not rows:
            return [], []
        
        # Strategy 1: Find header row by keyword matching
        header_row, header_idx = self._find_header_row(rows)
        
        if header_row:
            # Use header positions
            headers, positions = self._extract_from_header(header_row)
            if headers:
                return headers, positions
        
        # Strategy 2: Use projection-based column detection
        all_blocks = []
        for row in rows:
            for item in row.items:
                all_blocks.append({
                    "text": item.text,
                    "bbox": item.bbox,
                })
        
        analyzer = ProjectionAnalyzer()
        _, v_proj = analyzer.compute_projections(all_blocks)
        column_bounds = analyzer.find_column_boundaries(v_proj)
        
        if column_bounds:
            headers = [""] * len(column_bounds)
            return headers, [(float(b[0]), float(b[1])) for b in column_bounds]
        
        # Strategy 3: Estimate from row with most items
        return self._estimate_from_data_rows(rows)
    
    def _find_header_row(self, rows: List[TableRow]) -> Tuple[Optional[TableRow], int]:
        """Find the header row by keyword matching."""
        for idx, row in enumerate(rows[:5]):
            row_text = " ".join(item.text for item in row.items).lower()
            
            # Count header keyword matches
            matches = 0
            for variants in self.INVOICE_HEADERS.values():
                for v in variants:
                    if v.lower() in row_text:
                        matches += 1
                        break
            
            if matches >= 3:  # At least 3 header keywords found
                return row, idx
        
        return None, -1
    
    def _extract_from_header(self, header_row: TableRow) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Extract column info from header row with improved boundary calculation.
        
        Instead of using just the header text bbox, calculate column boundaries
        as the midpoint between adjacent headers. This gives more accurate column
        separation, especially for tables with narrow headers but wider data cells.
        """
        if not header_row.items:
            return [], []
        
        headers = []
        item_centers = []  # (x_center, x_min, x_max, header_name)
        
        for item in header_row.items:
            # Normalize header text
            header_text = item.text.strip()
            normalized = self._normalize_header(header_text)
            headers.append(normalized or header_text)
            
            # Get position
            xs = [p[0] for p in item.bbox]
            x_min, x_max = min(xs), max(xs)
            x_center = (x_min + x_max) / 2
            item_centers.append((x_center, x_min, x_max))
        
        # Calculate column boundaries as midpoints between adjacent headers
        positions = []
        for i in range(len(item_centers)):
            x_center, x_min, x_max = item_centers[i]
            
            # Left boundary: midpoint to previous header, or x_min - margin
            if i == 0:
                left = max(0, x_min - 50)  # First column extends left
            else:
                prev_center = item_centers[i - 1][0]
                left = (prev_center + x_center) / 2
            
            # Right boundary: midpoint to next header, or x_max + margin
            if i == len(item_centers) - 1:
                right = x_max + 100  # Last column extends right
            else:
                next_center = item_centers[i + 1][0]
                right = (x_center + next_center) / 2
            
            positions.append((left, right))
        
        logger.debug(f"Column positions calculated from {len(headers)} headers: {headers}")
        
        return headers, positions
    
    def _normalize_header(self, text: str) -> str:
        """Normalize header text to standard name."""
        text_lower = text.lower().replace(" ", "")
        
        for standard_name, variants in self.INVOICE_HEADERS.items():
            for v in variants:
                if v.lower().replace(" ", "") in text_lower:
                    return standard_name
        
        return text
    
    def _estimate_from_data_rows(self, rows: List[TableRow]) -> Tuple[List[str], List[Tuple[float, float]]]:
        """Estimate columns from data rows."""
        # Find row with most items
        max_items_row = max(rows, key=lambda r: len(r.items))
        
        headers = [""] * len(max_items_row.items)
        positions = []
        
        for item in max_items_row.items:
            xs = [p[0] for p in item.bbox]
            positions.append((min(xs), max(xs)))
        
        return headers, positions


class TableSegmenter:
    """
    Segments table into regions: header, body, subtotal, total, footer.
    
    Uses keyword detection and position analysis to identify
    different semantic regions within a table.
    """
    
    # Keywords for region detection
    HEADER_KEYWORDS = ["项目名称", "规格", "单位", "数量", "单价", "金额", "税率"]
    SUBTOTAL_KEYWORDS = ["小计", "合计"]
    TOTAL_KEYWORDS = ["价税合计", "总计", "大写"]
    FOOTER_KEYWORDS = ["备注", "开票人", "收款人", "复核"]
    
    def segment_table(self, rows: List[TableRow]) -> Dict[str, Any]:
        """
        Segment rows into regions.
        
        Returns:
            {
                "header": TableRow or None,
                "body": List[TableRow],
                "subtotal": List[TableRow],
                "total": TableRow or None,
                "footer": List[TableRow],
                "regions": [(start_idx, end_idx, type), ...]
            }
        """
        result = {
            "header": None,
            "body": [],
            "subtotal": [],
            "total": None,
            "footer": [],
            "regions": [],
        }
        
        if not rows:
            return result
        
        body_start = 0
        body_end = len(rows)
        
        for idx, row in enumerate(rows):
            row_text = " ".join(item.text for item in row.items)
            row_type = self._classify_row(row_text)
            row.row_type = row_type
            
            if row_type == "header":
                result["header"] = row
                body_start = idx + 1
            elif row_type == "subtotal":
                result["subtotal"].append(row)
            elif row_type == "total":
                result["total"] = row
                body_end = min(body_end, idx)
            elif row_type == "footer":
                result["footer"].append(row)
                body_end = min(body_end, idx)
        
        # Body rows
        for idx in range(body_start, body_end):
            if rows[idx].row_type == "data":
                result["body"].append(rows[idx])
        
        # Build regions list
        if result["header"]:
            result["regions"].append((0, 1, "header"))
        if result["body"]:
            result["regions"].append((body_start, body_end, "body"))
        
        return result
    
    def _classify_row(self, text: str) -> str:
        """Classify row type by keywords."""
        text_lower = text.lower()
        
        # Check header
        header_hits = sum(1 for kw in self.HEADER_KEYWORDS if kw in text_lower)
        if header_hits >= 2:
            return "header"
        
        # Check total first (more specific)
        for kw in self.TOTAL_KEYWORDS:
            if kw in text_lower:
                return "total"
        
        # Check subtotal
        for kw in self.SUBTOTAL_KEYWORDS:
            if kw in text_lower:
                return "subtotal"
        
        # Check footer
        for kw in self.FOOTER_KEYWORDS:
            if kw in text_lower:
                return "footer"
        
        return "data"


# ============================================================================
# Enhanced API Functions
# ============================================================================

def extract_table_structure_advanced(
    blocks: List[Dict],
    image_width: int = 0,
    image_height: int = 0,
) -> Dict:
    """
    Advanced table structure extraction with projection analysis.
    
    Args:
        blocks: OCR blocks
        image_width: Image width
        image_height: Image height
        
    Returns:
        Enhanced TableStructure with region info
    """
    # Basic clustering
    engine = RowClusteringEngine()
    rows = engine.cluster_blocks(blocks)
    
    if not rows:
        return {"rows": [], "regions": {}}
    
    # Adaptive column detection
    col_detector = AdaptiveColumnDetector()
    headers, positions = col_detector.detect_columns_adaptive(rows, image_width)
    
    # Table segmentation
    segmenter = TableSegmenter()
    regions = segmenter.segment_table(rows)
    
    # Assign columns to items
    for row in rows:
        for item in row.items:
            col_idx = engine._find_best_column(item.x_center, positions)
            item.column_index = col_idx
            if 0 <= col_idx < len(headers):
                item.column_name = headers[col_idx]
    
    # Build result
    structure = TableStructure(
        rows=rows,
        column_headers=headers,
        column_positions=positions,
        row_count=len(rows),
        column_count=len(headers),
        header_row_index=rows.index(regions["header"]) if regions["header"] else -1,
    )
    
    result = structure.to_dict()
    result["regions"] = {
        "header": regions["header"].to_dict() if regions["header"] else None,
        "bodyCount": len(regions["body"]),
        "subtotalCount": len(regions["subtotal"]),
        "hasTotal": regions["total"] is not None,
        "footerCount": len(regions["footer"]),
    }
    
    return result
