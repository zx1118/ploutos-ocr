"""
Post Processor Module
=====================

Rule-based correction engine for OCR results.

Components:
- PostProcessor: Main processor that applies rules
- Rule: Rule definition and types
- RuleEngine: Rule matching and application
"""

from .processor import PostProcessor
from .rule import Rule, RuleType
from .engine import RuleEngine

__all__ = [
    "PostProcessor",
    "Rule",
    "RuleType",
    "RuleEngine",
]

