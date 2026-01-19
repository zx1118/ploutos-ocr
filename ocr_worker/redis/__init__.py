"""
Redis Module
=============

Provides Redis client with both generic operations and Stream-specific operations.

Components:
- RedisClient: Generic Redis operations (get, set, hash, list, etc.)
- StreamClient: Redis Stream operations (XADD, XREADGROUP, XACK, XAUTOCLAIM, etc.)
- ConnectionPool: Shared connection pool management
"""

from .client import RedisClient
from .stream import StreamClient, StreamMessage
from .pool import get_redis_pool, get_redis_client

__all__ = [
    "RedisClient",
    "StreamClient", 
    "StreamMessage",
    "get_redis_pool",
    "get_redis_client",
]

