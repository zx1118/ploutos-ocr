"""
Generic Redis Client
====================

Encapsulates common Redis operations with error handling and logging.
"""

import json
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import redis
from loguru import logger

from .pool import get_redis_client


class RedisClient:
    """
    Generic Redis client wrapper with common operations.
    
    Provides a clean interface for:
    - String operations (get, set, delete)
    - Hash operations (hget, hset, hgetall)
    - List operations (lpush, rpush, lpop, rpop, lrange)
    - Set operations (sadd, srem, smembers)
    - Key operations (exists, expire, ttl, delete)
    - Pub/Sub operations (publish, subscribe)
    """
    
    def __init__(self, client: Optional[redis.Redis] = None):
        """
        Initialize Redis client.
        
        Args:
            client: Optional Redis client instance. If not provided, uses shared pool.
        """
        self._client = client or get_redis_client()
    
    @property
    def client(self) -> redis.Redis:
        """Get underlying Redis client."""
        return self._client
    
    # ==================== String Operations ====================
    
    def get(self, key: str) -> Optional[str]:
        """Get string value by key."""
        try:
            return self._client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            raise
    
    def set(
        self,
        key: str,
        value: Union[str, int, float],
        ex: Optional[Union[int, timedelta]] = None,
        px: Optional[Union[int, timedelta]] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set string value with optional expiration.
        
        Args:
            key: Key name
            value: Value to set
            ex: Expire time in seconds
            px: Expire time in milliseconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            result = self._client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            raise
    
    def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set value with expiration in seconds."""
        try:
            return self._client.setex(key, seconds, value)
        except redis.RedisError as e:
            logger.error(f"Redis SETEX error for key '{key}': {e}")
            raise
    
    def setnx(self, key: str, value: str) -> bool:
        """Set value only if key doesn't exist."""
        try:
            return self._client.setnx(key, value)
        except redis.RedisError as e:
            logger.error(f"Redis SETNX error for key '{key}': {e}")
            raise
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment value by amount."""
        try:
            return self._client.incr(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis INCR error for key '{key}': {e}")
            raise
    
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement value by amount."""
        try:
            return self._client.decr(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis DECR error for key '{key}': {e}")
            raise
    
    # ==================== JSON Operations ====================
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get and parse JSON value."""
        value = self.get(key)
        if value is not None:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for key '{key}': {e}")
                return None
        return None
    
    def set_json(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
    ) -> bool:
        """Serialize and set JSON value."""
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            return self.set(key, json_str, ex=ex)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON encode error for key '{key}': {e}")
            raise
    
    # ==================== Hash Operations ====================
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        try:
            return self._client.hget(name, key)
        except redis.RedisError as e:
            logger.error(f"Redis HGET error for hash '{name}', key '{key}': {e}")
            raise
    
    def hset(self, name: str, key: Optional[str] = None, value: Optional[str] = None,
             mapping: Optional[Dict[str, str]] = None) -> int:
        """Set hash field(s)."""
        try:
            return self._client.hset(name, key, value, mapping)
        except redis.RedisError as e:
            logger.error(f"Redis HSET error for hash '{name}': {e}")
            raise
    
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields and values."""
        try:
            return self._client.hgetall(name)
        except redis.RedisError as e:
            logger.error(f"Redis HGETALL error for hash '{name}': {e}")
            raise
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        try:
            return self._client.hdel(name, *keys)
        except redis.RedisError as e:
            logger.error(f"Redis HDEL error for hash '{name}': {e}")
            raise
    
    def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """Increment hash field by amount."""
        try:
            return self._client.hincrby(name, key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis HINCRBY error for hash '{name}', key '{key}': {e}")
            raise
    
    # ==================== List Operations ====================
    
    def lpush(self, name: str, *values: str) -> int:
        """Push values to the left of list."""
        try:
            return self._client.lpush(name, *values)
        except redis.RedisError as e:
            logger.error(f"Redis LPUSH error for list '{name}': {e}")
            raise
    
    def rpush(self, name: str, *values: str) -> int:
        """Push values to the right of list."""
        try:
            return self._client.rpush(name, *values)
        except redis.RedisError as e:
            logger.error(f"Redis RPUSH error for list '{name}': {e}")
            raise
    
    def lpop(self, name: str, count: Optional[int] = None) -> Optional[Union[str, List[str]]]:
        """Pop value(s) from the left of list."""
        try:
            return self._client.lpop(name, count)
        except redis.RedisError as e:
            logger.error(f"Redis LPOP error for list '{name}': {e}")
            raise
    
    def rpop(self, name: str, count: Optional[int] = None) -> Optional[Union[str, List[str]]]:
        """Pop value(s) from the right of list."""
        try:
            return self._client.rpop(name, count)
        except redis.RedisError as e:
            logger.error(f"Redis RPOP error for list '{name}': {e}")
            raise
    
    def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get range of elements from list."""
        try:
            return self._client.lrange(name, start, end)
        except redis.RedisError as e:
            logger.error(f"Redis LRANGE error for list '{name}': {e}")
            raise
    
    def llen(self, name: str) -> int:
        """Get list length."""
        try:
            return self._client.llen(name)
        except redis.RedisError as e:
            logger.error(f"Redis LLEN error for list '{name}': {e}")
            raise
    
    # ==================== Set Operations ====================
    
    def sadd(self, name: str, *values: str) -> int:
        """Add members to set."""
        try:
            return self._client.sadd(name, *values)
        except redis.RedisError as e:
            logger.error(f"Redis SADD error for set '{name}': {e}")
            raise
    
    def srem(self, name: str, *values: str) -> int:
        """Remove members from set."""
        try:
            return self._client.srem(name, *values)
        except redis.RedisError as e:
            logger.error(f"Redis SREM error for set '{name}': {e}")
            raise
    
    def smembers(self, name: str) -> set:
        """Get all members of set."""
        try:
            return self._client.smembers(name)
        except redis.RedisError as e:
            logger.error(f"Redis SMEMBERS error for set '{name}': {e}")
            raise
    
    def sismember(self, name: str, value: str) -> bool:
        """Check if value is member of set."""
        try:
            return self._client.sismember(name, value)
        except redis.RedisError as e:
            logger.error(f"Redis SISMEMBER error for set '{name}': {e}")
            raise
    
    # ==================== Key Operations ====================
    
    def exists(self, *names: str) -> int:
        """Check if key(s) exist."""
        try:
            return self._client.exists(*names)
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error: {e}")
            raise
    
    def delete(self, *names: str) -> int:
        """Delete key(s)."""
        try:
            return self._client.delete(*names)
        except redis.RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            raise
    
    def expire(self, name: str, time: Union[int, timedelta]) -> bool:
        """Set key expiration."""
        try:
            return self._client.expire(name, time)
        except redis.RedisError as e:
            logger.error(f"Redis EXPIRE error for key '{name}': {e}")
            raise
    
    def ttl(self, name: str) -> int:
        """Get key TTL in seconds."""
        try:
            return self._client.ttl(name)
        except redis.RedisError as e:
            logger.error(f"Redis TTL error for key '{name}': {e}")
            raise
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        try:
            return self._client.keys(pattern)
        except redis.RedisError as e:
            logger.error(f"Redis KEYS error for pattern '{pattern}': {e}")
            raise
    
    # ==================== Pub/Sub Operations ====================
    
    def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        try:
            return self._client.publish(channel, message)
        except redis.RedisError as e:
            logger.error(f"Redis PUBLISH error for channel '{channel}': {e}")
            raise
    
    def pubsub(self) -> redis.client.PubSub:
        """Get PubSub object for subscribing."""
        return self._client.pubsub()
    
    # ==================== Pipeline Operations ====================
    
    def pipeline(self, transaction: bool = True) -> redis.client.Pipeline:
        """Get pipeline for batched operations."""
        return self._client.pipeline(transaction=transaction)
    
    # ==================== Health Check ====================
    
    def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return self._client.ping()
        except redis.RedisError:
            return False
    
    def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get Redis server info."""
        try:
            return self._client.info(section)
        except redis.RedisError as e:
            logger.error(f"Redis INFO error: {e}")
            raise

