"""
Redis Connection Pool Management
================================

Singleton pattern for Redis connection pool to ensure efficient connection reuse.
"""

from functools import lru_cache
from typing import Optional

import redis
from loguru import logger

from ..config import settings


class RedisPoolManager:
    """Manages Redis connection pool as singleton."""
    
    _pool: Optional[redis.ConnectionPool] = None
    _client: Optional[redis.Redis] = None
    
    @classmethod
    def get_pool(cls) -> redis.ConnectionPool:
        """Get or create connection pool."""
        if cls._pool is None:
            redis_settings = settings.redis
            
            cls._pool = redis.ConnectionPool(
                host=redis_settings.host,
                port=redis_settings.port,
                password=redis_settings.password,
                db=redis_settings.db,
                max_connections=redis_settings.max_connections,
                socket_timeout=redis_settings.socket_timeout,
                socket_connect_timeout=redis_settings.socket_connect_timeout,
                decode_responses=True,  # Return strings instead of bytes
            )
            
            logger.info(
                f"Redis connection pool created: {redis_settings.host}:{redis_settings.port}/{redis_settings.db}"
            )
        
        return cls._pool
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get Redis client with shared pool."""
        if cls._client is None:
            cls._client = redis.Redis(connection_pool=cls.get_pool())
            
            # Test connection
            try:
                cls._client.ping()
                logger.info("Redis connection verified successfully")
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return cls._client
    
    @classmethod
    def close(cls):
        """Close connection pool and client."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            logger.info("Redis client closed")
        
        if cls._pool is not None:
            cls._pool.disconnect()
            cls._pool = None
            logger.info("Redis connection pool closed")


@lru_cache()
def get_redis_pool() -> redis.ConnectionPool:
    """Get cached Redis connection pool."""
    return RedisPoolManager.get_pool()


@lru_cache()
def get_redis_client() -> redis.Redis:
    """Get cached Redis client."""
    return RedisPoolManager.get_client()

