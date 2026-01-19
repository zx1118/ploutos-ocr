"""
Rule Definitions
================

Rule types and structures for OCR post-processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RuleType(Enum):
    """Types of correction rules."""
    
    REGEX = "regex"           # Regular expression replacement
    DICT = "dict"             # Dictionary/mapping replacement
    TEMPLATE = "template"     # Template validation
    FUZZY = "fuzzy"           # Fuzzy matching
    NORMALIZE = "normalize"   # Normalization (trim, case, etc.)
    VALIDATE = "validate"     # Validation without change


@dataclass
class Rule:
    """OCR correction rule definition."""
    
    id: str
    name: str
    rule_type: RuleType
    pattern: str
    replacement: Optional[str] = None
    
    # Scope
    doc_type: str = "ALL"     # ALL, INVOICE, CONTRACT, etc.
    field_name: Optional[str] = None  # None means apply to all fields
    
    # Priority and status
    priority: int = 100       # Lower = higher priority
    is_enabled: bool = True
    
    # Statistics
    hit_count: int = 0
    
    # Metadata
    source: str = "manual"    # manual, auto_extracted
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "ruleType": self.rule_type.value,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "docType": self.doc_type,
            "fieldName": self.field_name,
            "priority": self.priority,
            "isEnabled": self.is_enabled,
            "hitCount": self.hit_count,
            "source": self.source,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("ruleName", "")),
            rule_type=RuleType(data.get("ruleType", "regex")),
            pattern=data.get("pattern", ""),
            replacement=data.get("replacement"),
            doc_type=data.get("docType", "ALL"),
            field_name=data.get("fieldName"),
            priority=data.get("priority", 100),
            is_enabled=data.get("isEnabled", True),
            hit_count=data.get("hitCount", 0),
            source=data.get("source", "manual"),
            description=data.get("description", ""),
        )


# Built-in rules for common OCR errors
BUILTIN_RULES = [
    # Number confusion
    Rule(
        id="builtin_o_to_0",
        name="O to 0 in numbers",
        rule_type=RuleType.REGEX,
        pattern=r"(?<=\d)[oO](?=\d)",
        replacement="0",
        field_name="amount",
        priority=10,
        source="builtin",
        description="Replace O with 0 in numeric contexts",
    ),
    Rule(
        id="builtin_l_to_1",
        name="l/I to 1 in numbers",
        rule_type=RuleType.REGEX,
        pattern=r"(?<=\d)[lI](?=\d)",
        replacement="1",
        field_name="amount",
        priority=10,
        source="builtin",
        description="Replace l or I with 1 in numeric contexts",
    ),
    Rule(
        id="builtin_s_to_5",
        name="S to 5 in numbers",
        rule_type=RuleType.REGEX,
        pattern=r"(?<=\d)[sS](?=\d)",
        replacement="5",
        field_name="amount",
        priority=10,
        source="builtin",
        description="Replace S with 5 in numeric contexts",
    ),
    
    # Amount normalization
    Rule(
        id="builtin_amount_comma",
        name="Normalize amount commas",
        rule_type=RuleType.NORMALIZE,
        pattern=r"[\s,]+(?=\d{3})",
        replacement=",",
        field_name="amount",
        priority=20,
        source="builtin",
        description="Normalize thousand separators in amounts",
    ),
    
    # Date normalization
    Rule(
        id="builtin_date_format",
        name="Normalize date format",
        rule_type=RuleType.REGEX,
        pattern=r"(\d{4})[年/](\d{1,2})[月/](\d{1,2})日?",
        replacement=r"\1-\2-\3",
        field_name="invoice_date",
        priority=20,
        source="builtin",
        description="Normalize date to YYYY-MM-DD format",
    ),
    
    # Whitespace cleanup
    Rule(
        id="builtin_trim",
        name="Trim whitespace",
        rule_type=RuleType.NORMALIZE,
        pattern=r"^\s+|\s+$",
        replacement="",
        priority=50,
        source="builtin",
        description="Remove leading/trailing whitespace",
    ),
]

