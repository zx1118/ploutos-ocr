"""
Configuration Management
========================

Centralized configuration using pydantic-settings for environment variable parsing
with validation and type conversion.
"""

import os
import socket
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Redis connection settings."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    host: str = Field(default="127.0.0.1", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    
    # Connection pool settings
    max_connections: int = Field(default=10, description="Max connections in pool")
    socket_timeout: float = Field(default=30.0, description="Socket timeout in seconds (must be > block time)")
    socket_connect_timeout: float = Field(default=10.0, description="Connection timeout")
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class StreamSettings(BaseSettings):
    """Redis Stream settings for OCR task queue."""
    
    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    stream_key: str = Field(default="ocr:tasks", description="Main task stream key")
    result_key: str = Field(default="ocr:results", description="Result stream key for callbacks")
    dlq_key: str = Field(default="ocr:tasks:dlq", description="Dead letter queue key")
    consumer_group: str = Field(default="ocr-workers", description="Consumer group name")
    max_retry: int = Field(default=3, description="Max retry count before DLQ")
    pending_timeout_ms: int = Field(default=60000, description="Pending message timeout")
    
    # Consumer settings
    batch_size: int = Field(default=1, description="Messages to read per batch")
    block_ms: int = Field(default=5000, description="Block time when reading")
    
    @property
    def consumer_name(self) -> str:
        """Generate unique consumer name."""
        hostname = socket.gethostname()
        pid = os.getpid()
        return f"{hostname}-{pid}"


class OcrSettings(BaseSettings):
    """PaddleOCR engine settings."""
    
    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    use_gpu: bool = Field(default=False, description="Use GPU for inference")
    auto_detect_gpu: bool = Field(default=True, description="Auto-detect and use GPU if available")
    model_dir: str = Field(default="./models", description="Model directory")
    default_lang: str = Field(default="ch", description="Default language")
    confidence_threshold: float = Field(default=0.8, description="Min confidence for auto-pass")
    
    # Model versions
    det_model_dir: Optional[str] = Field(default=None, description="Detection model dir")
    rec_model_dir: Optional[str] = Field(default=None, description="Recognition model dir")
    cls_model_dir: Optional[str] = Field(default=None, description="Classification model dir")
    
    # Processing options
    use_angle_cls: bool = Field(default=True, description="Use angle classification")
    pdf_dpi: int = Field(default=150, description="DPI for PDF to image conversion (lower = faster, higher = more accurate)")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable OCR result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    cache_use_file_hash: bool = Field(default=True, description="Use file content hash for cache key")
    
    # Timeout settings
    default_timeout: float = Field(default=300.0, description="Default OCR timeout in seconds (5 minutes)")
    pdf_page_timeout: float = Field(default=60.0, description="Timeout per PDF page in seconds")
    
    # Layout analysis settings
    layout_engine: str = Field(
        default="auto", 
        description="Layout engine: auto, paddle-structure, doclayout-yolo, rule-based"
    )
    enable_table_structure: bool = Field(
        default=True, 
        description="Enable advanced table structure recognition"
    )
    
    # PPStructure table recognition (方案 A)
    # 暂时禁用：PPStructure 的 bbox 坐标与原图不对应，导致定位偏移
    use_ppstructure_table: bool = Field(
        default=False,
        description="Use PPStructure for table extraction in invoices (方案 A) - 暂时禁用"
    )
    
    # DocLayout-YOLO layout analysis (方案 B - MinerU 级别)
    use_doclayout_yolo: bool = Field(
        default=False,
        description="Use DocLayout-YOLO for MinerU-level layout analysis (方案 B)"
    )
    doclayout_model_path: Optional[str] = Field(
        default=None,
        description="Path to DocLayout-YOLO model (.pt file), auto-download if not specified"
    )
    doclayout_conf_threshold: float = Field(
        default=0.25,
        description="Confidence threshold for DocLayout-YOLO detection"
    )
    
    @property
    def should_use_gpu(self) -> bool:
        """Determine if GPU should be used (with auto-detection)."""
        if self.use_gpu:
            return True
        if not self.auto_detect_gpu:
            return False
        return self._detect_gpu_available()
    
    @staticmethod
    def _detect_gpu_available() -> bool:
        """Detect if CUDA GPU is available for PaddlePaddle."""
        try:
            import paddle
            return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            return False


class CallbackSettings(BaseSettings):
    """JeecgBoot callback settings."""
    
    model_config = SettingsConfigDict(env_prefix="JEECG_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # 本地开发和线上都使用 HTTP（避免 Python SSL 与 Undertow 的兼容性问题）
    base_url: str = Field(default="http://localhost:8080/jeecg-boot", description="JeecgBoot base URL")
    callback_path: str = Field(default="/ocr/api/callback", description="Callback endpoint path")
    api_token: Optional[str] = Field(default=None, description="API token for authentication")
    
    # HTTP settings
    timeout: float = Field(default=30.0, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts")
    
    @property
    def callback_url(self) -> str:
        """Get full callback URL."""
        return f"{self.base_url.rstrip('/')}{self.callback_path}"


class ApiSettings(BaseSettings):
    """FastAPI settings."""
    
    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8100, description="API port")
    workers: int = Field(default=1, description="Number of workers")
    debug: bool = Field(default=False, description="Debug mode")


class LogSettings(BaseSettings):
    """Logging settings."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_", env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    level: str = Field(default="INFO", description="Log level")
    dir: str = Field(default="./logs", description="Log directory")
    rotation: str = Field(default="100 MB", description="Log rotation size")
    retention: str = Field(default="30 days", description="Log retention period")


class Settings(BaseSettings):
    """Main settings aggregating all sub-settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    redis: RedisSettings = Field(default_factory=RedisSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    callback: CallbackSettings = Field(default_factory=CallbackSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    log: LogSettings = Field(default_factory=LogSettings)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings from environment variables.
    
    Clears the cache and re-reads all configuration.
    Use this for hot-reload of configuration.
    
    Returns:
        New Settings instance with updated values
    """
    get_settings.cache_clear()
    new_settings = get_settings()
    
    # Update the global settings reference
    global settings
    settings = new_settings
    
    return new_settings


class SettingsWatcher:
    """
    Watch for settings changes and trigger callbacks.
    
    Usage:
        watcher = SettingsWatcher()
        watcher.on_change("ocr.pdf_dpi", lambda old, new: print(f"DPI: {old} -> {new}"))
        watcher.check_and_notify()
    """
    
    def __init__(self):
        self._callbacks: dict = {}
        self._last_values: dict = {}
        self._initialized = False
    
    def on_change(self, key: str, callback) -> None:
        """
        Register callback for setting changes.
        
        Args:
            key: Dot-separated setting key (e.g., "ocr.pdf_dpi")
            callback: Function(old_value, new_value) to call on change
        """
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
    
    def _get_value(self, key: str):
        """Get setting value by dot-separated key."""
        parts = key.split(".")
        obj = settings
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    
    def check_and_notify(self) -> dict:
        """
        Check for setting changes and notify callbacks.
        
        Returns:
            Dict of changed settings {key: (old_value, new_value)}
        """
        # Reload settings from environment
        new_settings = reload_settings()
        
        changes = {}
        
        for key, callbacks in self._callbacks.items():
            new_value = self._get_value(key)
            old_value = self._last_values.get(key)
            
            if self._initialized and old_value != new_value:
                changes[key] = (old_value, new_value)
                
                # Notify callbacks
                for callback in callbacks:
                    try:
                        callback(old_value, new_value)
                    except Exception as e:
                        print(f"Error in settings callback for {key}: {e}")
            
            self._last_values[key] = new_value
        
        self._initialized = True
        return changes


# Global settings watcher instance
settings_watcher = SettingsWatcher()


# Convenience export
settings = get_settings()

