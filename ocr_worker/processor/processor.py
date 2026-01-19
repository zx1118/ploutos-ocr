"""
Post Processor
==============

Main processor that applies rules and structure extraction to OCR results.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from ..config import settings
from ..ocr.result import OCRResult, StructuredData
from ..ocr.structure import StructureExtractor
from ..redis.client import RedisClient
from .engine import RuleEngine
from .rule import Rule


class PostProcessor:
    """
    Post-processor for OCR results.
    
    Features:
    - Rule-based text correction
    - Structure extraction
    - Rule caching from Redis
    - Dynamic rule updates
    """
    
    # Redis cache key for rules
    RULES_CACHE_KEY = "ocr:rules:cache"
    RULES_CACHE_TTL = 300  # 5 minutes
    
    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        rules_api_url: Optional[str] = None,
    ):
        """
        Initialize post processor.
        
        Args:
            redis_client: Redis client for rule caching
            rules_api_url: URL to fetch rules from JeecgBoot
        """
        self._redis = redis_client
        self._rules_api_url = rules_api_url or f"{settings.callback.base_url}/ocr/api/rules"
        
        self._rule_engine = RuleEngine()
        self._structure_extractor = StructureExtractor()
        
        # Load rules from cache/API
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules from cache or API."""
        rules_loaded = False
        
        # Try Redis cache first
        if self._redis:
            try:
                cached = self._redis.get(self.RULES_CACHE_KEY)
                if cached:
                    rules_data = json.loads(cached)
                    self._rule_engine.load_rules(rules_data)
                    logger.info(f"Loaded {len(rules_data)} rules from cache")
                    rules_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load rules from cache: {e}")
        
        # Fallback to API
        if not rules_loaded:
            self._fetch_rules_from_api()
    
    def _fetch_rules_from_api(self) -> None:
        """Fetch rules from JeecgBoot API."""
        try:
            import httpx
            
            response = httpx.get(
                self._rules_api_url,
                timeout=10.0,
                headers=self._get_auth_headers(),
            )
            
            if response.status_code == 200:
                data = response.json()
                rules_data = data.get("result", data.get("data", []))
                
                if isinstance(rules_data, list):
                    self._rule_engine.load_rules(rules_data)
                    
                    # Cache rules
                    if self._redis:
                        self._redis.setex(
                            self.RULES_CACHE_KEY,
                            self.RULES_CACHE_TTL,
                            json.dumps(rules_data),
                        )
                    
                    logger.info(f"Loaded {len(rules_data)} rules from API")
            else:
                logger.warning(f"Failed to fetch rules from API: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to fetch rules from API: {e}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API calls."""
        headers = {}
        
        token = settings.callback.api_token
        if token:
            headers["X-Access-Token"] = token
        
        return headers
    
    def process(
        self,
        ocr_result: OCRResult,
        doc_type: str = "AUTO",
    ) -> OCRResult:
        """
        Process OCR result with rules and structure extraction.
        
        Args:
            ocr_result: Raw OCR result
            doc_type: Document type hint
            
        Returns:
            Processed OCR result
        """
        logger.debug(f"Processing OCR result, doc_type={doc_type}")
        
        # 1. Extract structure if not already done
        if ocr_result.structured_data is None:
            ocr_result.structured_data = self._structure_extractor.extract(
                ocr_result, doc_type
            )
        
        actual_doc_type = ocr_result.structured_data.doc_type
        
        # 2. Apply rules to structured fields
        all_applied_rules = []
        
        if ocr_result.structured_data:
            for field_name, value in list(ocr_result.structured_data.fields.items()):
                if isinstance(value, str):
                    corrected, applied = self._rule_engine.apply_rules(
                        value,
                        doc_type=actual_doc_type,
                        field_name=field_name,
                    )
                    
                    if corrected != value:
                        ocr_result.structured_data.fields[field_name] = corrected
                        
                        # Update common fields
                        if hasattr(ocr_result.structured_data, field_name):
                            setattr(ocr_result.structured_data, field_name, corrected)
                    
                    all_applied_rules.extend(applied)
        
        # 3. Apply rules to full text (optional, for specific use cases)
        # corrected_text, text_rules = self._rule_engine.apply_rules(
        #     ocr_result.full_text,
        #     doc_type=actual_doc_type,
        # )
        # ocr_result.full_text = corrected_text
        # all_applied_rules.extend(text_rules)
        
        # 4. Update result
        ocr_result.applied_rules = all_applied_rules
        
        # 5. Recalculate review flag
        ocr_result.calculate_needs_review(settings.ocr.confidence_threshold)
        
        if all_applied_rules:
            logger.info(f"Applied {len(all_applied_rules)} correction(s)")
        
        return ocr_result
    
    def refresh_rules(self) -> int:
        """
        Refresh rules from API.
        
        Returns:
            Number of rules loaded
        """
        # Clear cache
        if self._redis:
            self._redis.delete(self.RULES_CACHE_KEY)
        
        # Create new engine and fetch rules
        self._rule_engine = RuleEngine()
        self._fetch_rules_from_api()
        
        return len(self._rule_engine.get_rules())
    
    def add_rule(self, rule: Rule) -> None:
        """Add rule to engine."""
        self._rule_engine.add_rule(rule)
    
    def get_rules(self) -> List[Rule]:
        """Get all rules."""
        return self._rule_engine.get_rules()
    
    def increment_rule_hit(self, rule_id: str) -> None:
        """
        Increment rule hit count and notify API.
        
        Args:
            rule_id: Rule ID
        """
        rule = self._rule_engine.get_rule(rule_id)
        if rule:
            rule.hit_count += 1
            
            # Async notify API (fire and forget)
            try:
                import httpx
                
                httpx.post(
                    f"{settings.callback.base_url}/ocr/api/rules/{rule_id}/hit",
                    timeout=5.0,
                    headers=self._get_auth_headers(),
                )
            except Exception:
                pass  # Ignore errors for hit tracking

