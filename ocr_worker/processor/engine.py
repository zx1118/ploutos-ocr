"""
Rule Engine
===========

Core engine for matching and applying correction rules.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .rule import Rule, RuleType, BUILTIN_RULES


class RuleEngine:
    """
    Rule matching and application engine.
    
    Features:
    - Multiple rule types (regex, dict, fuzzy, etc.)
    - Priority-based rule ordering
    - Hit count tracking
    - Rule caching
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        """
        Initialize rule engine.
        
        Args:
            rules: List of rules to use (default: builtin rules)
        """
        self._rules: List[Rule] = []
        self._rules_by_type: Dict[str, List[Rule]] = {}
        
        # Load builtin rules
        for rule in BUILTIN_RULES:
            self.add_rule(rule)
        
        # Load provided rules
        if rules:
            for rule in rules:
                self.add_rule(rule)
    
    def add_rule(self, rule: Rule) -> None:
        """Add rule to engine."""
        self._rules.append(rule)
        self._rebuild_index()
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove rule by ID."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.id != rule_id]
        
        if len(self._rules) < original_len:
            self._rebuild_index()
            return True
        return False
    
    def _rebuild_index(self) -> None:
        """Rebuild rule index for efficient lookup."""
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority)
        
        # Index by doc_type
        self._rules_by_type = {}
        for rule in self._rules:
            doc_type = rule.doc_type
            if doc_type not in self._rules_by_type:
                self._rules_by_type[doc_type] = []
            self._rules_by_type[doc_type].append(rule)
    
    def get_applicable_rules(
        self,
        doc_type: str,
        field_name: Optional[str] = None,
    ) -> List[Rule]:
        """
        Get rules applicable to document type and field.
        
        Args:
            doc_type: Document type
            field_name: Optional field name filter
            
        Returns:
            List of applicable rules sorted by priority
        """
        rules = []
        
        # Get rules for specific doc_type and ALL
        for dt in [doc_type, "ALL"]:
            rules.extend(self._rules_by_type.get(dt, []))
        
        # Filter by field_name
        if field_name:
            rules = [
                r for r in rules
                if r.field_name is None or r.field_name == field_name
            ]
        
        # Filter enabled rules and sort by priority
        rules = [r for r in rules if r.is_enabled]
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    def apply_rule(
        self,
        rule: Rule,
        value: str,
    ) -> Tuple[str, bool]:
        """
        Apply single rule to value.
        
        Args:
            rule: Rule to apply
            value: Input value
            
        Returns:
            Tuple of (result_value, was_changed)
        """
        if not value:
            return value, False
        
        original = value
        
        try:
            if rule.rule_type == RuleType.REGEX:
                value = re.sub(rule.pattern, rule.replacement or "", value)
            
            elif rule.rule_type == RuleType.DICT:
                if rule.pattern in value:
                    value = value.replace(rule.pattern, rule.replacement or "")
            
            elif rule.rule_type == RuleType.NORMALIZE:
                value = re.sub(rule.pattern, rule.replacement or "", value)
            
            elif rule.rule_type == RuleType.FUZZY:
                # Simple fuzzy: case-insensitive replacement
                pattern = re.compile(re.escape(rule.pattern), re.IGNORECASE)
                value = pattern.sub(rule.replacement or "", value)
            
            elif rule.rule_type == RuleType.TEMPLATE:
                # Validation only, no change
                if not re.match(rule.pattern, value):
                    logger.debug(f"Value '{value}' doesn't match template '{rule.pattern}'")
            
            elif rule.rule_type == RuleType.VALIDATE:
                # Validation only
                pass
        
        except Exception as e:
            logger.warning(f"Rule application failed: {rule.id}, error: {e}")
            return original, False
        
        was_changed = value != original
        
        if was_changed:
            rule.hit_count += 1
            logger.debug(f"Rule {rule.id} applied: '{original}' -> '{value}'")
        
        return value, was_changed
    
    def apply_rules(
        self,
        value: str,
        doc_type: str = "ALL",
        field_name: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Apply all applicable rules to value.
        
        Args:
            value: Input value
            doc_type: Document type
            field_name: Field name
            
        Returns:
            Tuple of (result_value, list of applied rules)
        """
        applied = []
        
        rules = self.get_applicable_rules(doc_type, field_name)
        
        for rule in rules:
            before = value
            value, was_changed = self.apply_rule(rule, value)
            
            if was_changed:
                applied.append({
                    "ruleId": rule.id,
                    "ruleName": rule.name,
                    "field": field_name,
                    "before": before,
                    "after": value,
                })
        
        return value, applied
    
    def get_rules(self) -> List[Rule]:
        """Get all rules."""
        return list(self._rules)
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get rule by ID."""
        for rule in self._rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def load_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """
        Load rules from dictionary list.
        
        Args:
            rules_data: List of rule dictionaries
            
        Returns:
            Number of rules loaded
        """
        count = 0
        for data in rules_data:
            try:
                rule = Rule.from_dict(data)
                self.add_rule(rule)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load rule: {e}")
        
        logger.info(f"Loaded {count} rules")
        return count
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules as dictionaries."""
        return [r.to_dict() for r in self._rules]

