"""
Evidence Enrichment Module
==========================

Matches field values to OCR text blocks for bbox localization.
"""

from .matcher import FieldMatcher, MatchResult

__all__ = ["FieldMatcher", "MatchResult"]

