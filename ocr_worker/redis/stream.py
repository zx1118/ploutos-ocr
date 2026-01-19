"""
Redis Stream Client
===================

Specialized client for Redis Stream operations with consumer group support.
Handles XADD, XREADGROUP, XACK, XAUTOCLAIM, and DLQ management.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import redis
from loguru import logger

from ..config import settings
from .pool import get_redis_client


@dataclass
class StreamMessage:
    """Represents a message from Redis Stream."""
    
    message_id: str
    data: Dict[str, str]
    stream_key: str = ""
    
    # Parsed fields (populated after construction)
    task_id: str = ""
    task_type: str = "FULL_PAGE"  # FULL_PAGE, ROI, EVIDENCE_ENRICH
    doc_id: str = ""
    file_path: str = ""
    file_name: str = ""
    doc_type: str = "AUTO"
    prefer_mode: str = "auto"
    retry_count: int = 0
    created_at: int = 0
    callback_url: str = ""
    
    # EVIDENCE_ENRICH specific
    fields_json: str = ""
    
    # ROI specific
    roi_json: str = ""
    target_field: str = ""
    
    def __post_init__(self):
        """Parse common fields from data (supports both camelCase and snake_case)."""
        # 支持两种命名格式（Java 端发送的消息可能使用不同格式）
        self.task_id = self.data.get("taskId", "") or self.data.get("task_id", "")
        self.task_type = self.data.get("taskType", "") or self.data.get("task_type", "FULL_PAGE")
        self.doc_id = self.data.get("docId", "") or self.data.get("doc_id", "")
        self.file_path = self.data.get("filePath", "") or self.data.get("file_path", "") or self.data.get("image_path", "")
        self.file_name = self.data.get("fileName", "") or self.data.get("file_name", "")
        self.doc_type = self.data.get("docType", "") or self.data.get("doc_type", "AUTO")
        self.prefer_mode = self.data.get("preferMode", "") or self.data.get("prefer_mode", "") or self.data.get("mode", "auto")
        self.retry_count = int(self.data.get("retry", 0))
        self.created_at = int(self.data.get("createdAt", 0) or self.data.get("created_at", 0))
        self.callback_url = self.data.get("callbackUrl", "") or self.data.get("callback_url", "")
        
        # EVIDENCE_ENRICH specific
        self.fields_json = self.data.get("fieldsJson", "") or self.data.get("fields_json", "")
        
        # ROI specific
        self.roi_json = self.data.get("roi", "") or self.data.get("roi_json", "")
        self.target_field = self.data.get("targetField", "")
    
    def to_retry_data(self) -> Dict[str, str]:
        """Create data dict for retry with incremented count."""
        retry_data = dict(self.data)
        retry_data["retry"] = str(self.retry_count + 1)
        return retry_data


class StreamClient:
    """
    Redis Stream client with consumer group support.
    
    Features:
    - Consumer group management (create, destroy)
    - Message production (XADD)
    - Message consumption (XREADGROUP with blocking)
    - Message acknowledgment (XACK)
    - Pending message reclamation (XAUTOCLAIM)
    - Dead letter queue (DLQ) handling
    - Stream info and monitoring
    """
    
    def __init__(
        self,
        stream_key: Optional[str] = None,
        dlq_key: Optional[str] = None,
        group_name: Optional[str] = None,
        consumer_name: Optional[str] = None,
        client: Optional[redis.Redis] = None,
    ):
        """
        Initialize Stream client.
        
        Args:
            stream_key: Main stream key (default from settings)
            dlq_key: Dead letter queue key (default from settings)
            group_name: Consumer group name (default from settings)
            consumer_name: Consumer name (default auto-generated)
            client: Optional Redis client instance
        """
        stream_settings = settings.stream
        
        self._client = client or get_redis_client()
        self._stream_key = stream_key or stream_settings.stream_key
        self._dlq_key = dlq_key or stream_settings.dlq_key
        self._group_name = group_name or stream_settings.consumer_group
        self._consumer_name = consumer_name or stream_settings.consumer_name
        self._max_retry = stream_settings.max_retry
        self._pending_timeout_ms = stream_settings.pending_timeout_ms
        self._batch_size = stream_settings.batch_size
        self._block_ms = stream_settings.block_ms
    
    @property
    def stream_key(self) -> str:
        """Get main stream key."""
        return self._stream_key
    
    @property
    def dlq_key(self) -> str:
        """Get DLQ key."""
        return self._dlq_key
    
    @property
    def group_name(self) -> str:
        """Get consumer group name."""
        return self._group_name
    
    @property
    def consumer_name(self) -> str:
        """Get consumer name."""
        return self._consumer_name
    
    # ==================== Group Management ====================
    
    def ensure_group(self) -> bool:
        """
        Ensure consumer group exists, create if not.
        
        Returns:
            True if group was created, False if already exists
        """
        try:
            self._client.xgroup_create(
                self._stream_key,
                self._group_name,
                id="0",
                mkstream=True,
            )
            logger.info(
                f"Consumer group created: {self._group_name} on stream {self._stream_key}"
            )
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group already exists: {self._group_name}")
                return False
            raise
    
    def destroy_group(self) -> bool:
        """Destroy consumer group."""
        try:
            result = self._client.xgroup_destroy(self._stream_key, self._group_name)
            logger.info(f"Consumer group destroyed: {self._group_name}")
            return bool(result)
        except redis.ResponseError as e:
            logger.error(f"Failed to destroy group: {e}")
            return False
    
    # ==================== Message Production ====================
    
    def add(
        self,
        data: Dict[str, str],
        stream_key: Optional[str] = None,
        message_id: str = "*",
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        """
        Add message to stream.
        
        Args:
            data: Message data as dict
            stream_key: Optional stream key (default: main stream)
            message_id: Message ID (default: auto-generated)
            maxlen: Optional max stream length
            approximate: Use approximate maxlen (~) for performance
            
        Returns:
            Message ID
        """
        key = stream_key or self._stream_key
        
        try:
            msg_id = self._client.xadd(
                key,
                data,
                id=message_id,
                maxlen=maxlen,
                approximate=approximate,
            )
            logger.debug(f"Message added to {key}: {msg_id}")
            return msg_id
        except redis.RedisError as e:
            logger.error(f"Failed to add message to {key}: {e}")
            raise
    
    def add_task(
        self,
        task_id: str,
        file_path: str,
        file_name: str = "",
        doc_type: str = "AUTO",
        prefer_mode: str = "auto",
        callback_url: Optional[str] = None,
        extra_data: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Add OCR task to stream with standard fields.
        
        Args:
            task_id: Unique task ID
            file_path: Path to image/PDF file
            file_name: Original file name
            doc_type: Document type (INVOICE, CONTRACT, etc.)
            prefer_mode: OCR mode (auto, text, structure)
            callback_url: Callback URL for result
            extra_data: Additional fields
            
        Returns:
            Message ID
        """
        callback = callback_url or settings.callback.callback_url
        
        data = {
            "taskId": task_id,
            "filePath": file_path,
            "fileName": file_name or file_path.split("/")[-1],
            "docType": doc_type,
            "preferMode": prefer_mode,
            "retry": "0",
            "createdAt": str(int(time.time() * 1000)),
            "callbackUrl": callback,
        }
        
        if extra_data:
            data.update(extra_data)
        
        return self.add(data)
    
    # ==================== Message Consumption ====================
    
    def read(
        self,
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> List[StreamMessage]:
        """
        Read new messages from stream using consumer group.
        
        Args:
            count: Number of messages to read (default from settings)
            block: Block time in ms (default from settings)
            
        Returns:
            List of StreamMessage objects
        """
        count = count or self._batch_size
        block = block if block is not None else self._block_ms
        
        try:
            result = self._client.xreadgroup(
                self._group_name,
                self._consumer_name,
                {self._stream_key: ">"},
                count=count,
                block=block,
            )
            
            if not result:
                return []
            
            messages = []
            for stream_name, message_list in result:
                for msg_id, data in message_list:
                    messages.append(
                        StreamMessage(
                            message_id=msg_id,
                            data=data,
                            stream_key=stream_name,
                        )
                    )
            
            if messages:
                logger.debug(f"Read {len(messages)} message(s) from {self._stream_key}")
            
            return messages
            
        except redis.RedisError as e:
            logger.error(f"Failed to read from stream: {e}")
            raise
    
    def read_pending(
        self,
        count: Optional[int] = None,
    ) -> List[StreamMessage]:
        """
        Read pending messages that were not acknowledged.
        
        Args:
            count: Number of messages to read
            
        Returns:
            List of StreamMessage objects
        """
        count = count or self._batch_size
        
        try:
            result = self._client.xreadgroup(
                self._group_name,
                self._consumer_name,
                {self._stream_key: "0"},  # Read pending
                count=count,
            )
            
            if not result:
                return []
            
            messages = []
            for stream_name, message_list in result:
                for msg_id, data in message_list:
                    if data:  # Pending messages have data
                        messages.append(
                            StreamMessage(
                                message_id=msg_id,
                                data=data,
                                stream_key=stream_name,
                            )
                        )
            
            return messages
            
        except redis.RedisError as e:
            logger.error(f"Failed to read pending messages: {e}")
            raise
    
    # ==================== Message Acknowledgment ====================
    
    def ack(self, *message_ids: str) -> int:
        """
        Acknowledge message(s).
        
        Args:
            message_ids: One or more message IDs to acknowledge
            
        Returns:
            Number of messages acknowledged
        """
        if not message_ids:
            return 0
        
        try:
            result = self._client.xack(self._stream_key, self._group_name, *message_ids)
            logger.debug(f"Acknowledged {result} message(s): {message_ids}")
            return result
        except redis.RedisError as e:
            logger.error(f"Failed to acknowledge messages: {e}")
            raise
    
    # ==================== Pending Message Management ====================
    
    def claim_stale(
        self,
        min_idle_time: Optional[int] = None,
        count: int = 10,
    ) -> List[StreamMessage]:
        """
        Claim stale pending messages from other consumers.
        
        Uses XAUTOCLAIM (Redis 6.2+) to reclaim messages that have been pending
        longer than min_idle_time, transferring them to this consumer.
        Falls back gracefully if XAUTOCLAIM is not available.
        
        Args:
            min_idle_time: Minimum idle time in ms (default from settings)
            count: Maximum messages to claim
            
        Returns:
            List of claimed StreamMessage objects
        """
        idle_time = min_idle_time or self._pending_timeout_ms
        
        try:
            result = self._client.xautoclaim(
                self._stream_key,
                self._group_name,
                self._consumer_name,
                min_idle_time=idle_time,
                start_id="0-0",
                count=count,
            )
            
            # Result format: [next_id, [[msg_id, data], ...], [deleted_ids]]
            if not result or not result[1]:
                return []
            
            messages = []
            for msg_id, data in result[1]:
                if data:
                    messages.append(
                        StreamMessage(
                            message_id=msg_id,
                            data=data,
                            stream_key=self._stream_key,
                        )
                    )
            
            if messages:
                logger.info(f"Claimed {len(messages)} stale message(s)")
            
            return messages
            
        except redis.ResponseError as e:
            # XAUTOCLAIM requires Redis 6.2+
            if "unknown command" in str(e).lower():
                logger.debug("XAUTOCLAIM not supported (requires Redis 6.2+), skipping stale claim")
                return []
            logger.error(f"Failed to claim stale messages: {e}")
            raise
        except redis.RedisError as e:
            logger.error(f"Failed to claim stale messages: {e}")
            raise
    
    def get_pending_info(self) -> Dict[str, Any]:
        """
        Get pending messages info for the group.
        
        Returns:
            Dict with pending count, min/max ID, and consumer details
        """
        try:
            info = self._client.xpending(self._stream_key, self._group_name)
            return {
                "pending_count": info["pending"],
                "min_id": info["min"],
                "max_id": info["max"],
                "consumers": info.get("consumers", {}),
            }
        except redis.RedisError as e:
            logger.error(f"Failed to get pending info: {e}")
            raise
    
    # ==================== Dead Letter Queue ====================
    
    def send_to_dlq(
        self,
        message: StreamMessage,
        error: Optional[str] = None,
    ) -> str:
        """
        Send message to dead letter queue.
        
        Args:
            message: Original message
            error: Optional error description
            
        Returns:
            DLQ message ID
        """
        dlq_data = dict(message.data)
        dlq_data["dlqTime"] = str(int(time.time() * 1000))
        dlq_data["originalId"] = message.message_id
        if error:
            dlq_data["error"] = error[:1000]  # Limit error length
        
        msg_id = self.add(dlq_data, stream_key=self._dlq_key)
        logger.warning(f"Message sent to DLQ: {message.task_id} -> {msg_id}")
        return msg_id
    
    def get_dlq_messages(self, count: int = 100) -> List[StreamMessage]:
        """
        Get messages from dead letter queue.
        
        Args:
            count: Maximum messages to retrieve
            
        Returns:
            List of StreamMessage objects from DLQ
        """
        try:
            result = self._client.xrange(self._dlq_key, count=count)
            
            messages = []
            for msg_id, data in result:
                messages.append(
                    StreamMessage(
                        message_id=msg_id,
                        data=data,
                        stream_key=self._dlq_key,
                    )
                )
            
            return messages
            
        except redis.RedisError as e:
            logger.error(f"Failed to get DLQ messages: {e}")
            raise
    
    # ==================== Retry Logic ====================
    
    def retry_or_dlq(
        self,
        message: StreamMessage,
        error: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Handle message retry or send to DLQ with exponential backoff.
        
        If retry count < max_retry, creates new message with incremented retry.
        Otherwise, sends to DLQ.
        Always ACKs the original message.
        
        Args:
            message: Original message
            error: Optional error description
            
        Returns:
            Tuple of (is_retry, new_message_id)
            is_retry is True if message was retried, False if sent to DLQ
        """
        try:
            if message.retry_count < self._max_retry:
                # Retry: add new message with incremented retry count
                new_data = message.to_retry_data()
                if error:
                    new_data["lastError"] = error[:500]
                
                # Calculate exponential backoff delay
                delay_seconds = self._calculate_backoff_delay(message.retry_count)
                new_data["delayUntil"] = str(int(time.time() * 1000) + delay_seconds * 1000)
                
                new_id = self.add(new_data)
                self.ack(message.message_id)
                
                logger.info(
                    f"Retry scheduled: {message.task_id} "
                    f"(attempt {message.retry_count + 1}/{self._max_retry}, "
                    f"backoff: {delay_seconds}s)"
                )
                return True, new_id
            else:
                # Max retries exceeded: send to DLQ
                dlq_id = self.send_to_dlq(message, error)
                self.ack(message.message_id)
                
                logger.warning(
                    f"Max retries exceeded, sent to DLQ: {message.task_id}"
                )
                return False, dlq_id
                
        except redis.RedisError as e:
            logger.error(f"Failed to handle retry/DLQ: {e}")
            raise
    
    def _calculate_backoff_delay(self, retry_count: int) -> int:
        """
        Calculate exponential backoff delay in seconds.
        
        Uses formula: min(base_delay * 2^retry_count, max_delay)
        With jitter to avoid thundering herd.
        
        Args:
            retry_count: Current retry attempt (0-indexed)
            
        Returns:
            Delay in seconds
        """
        import random
        
        base_delay = 2  # Start with 2 seconds
        max_delay = 300  # Cap at 5 minutes
        
        # Exponential: 2, 4, 8, 16, 32, 64... capped at 300
        delay = min(base_delay * (2 ** retry_count), max_delay)
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        
        return int(delay + jitter)
    
    def reprocess_dlq(
        self,
        count: int = 10,
        reset_retry: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Reprocess messages from dead letter queue.
        
        Moves messages from DLQ back to main stream for reprocessing.
        Useful for manually retrying failed tasks after fixing issues.
        
        Args:
            count: Number of messages to reprocess
            reset_retry: Reset retry count to 0 (default True)
            
        Returns:
            List of tuples (dlq_id, new_stream_id) for reprocessed messages
        """
        results = []
        
        try:
            dlq_messages = self.get_dlq_messages(count)
            
            for msg in dlq_messages:
                # Prepare data for reprocessing
                new_data = dict(msg.data)
                
                # Remove DLQ-specific fields
                new_data.pop("dlqTime", None)
                new_data.pop("originalId", None)
                new_data.pop("error", None)
                new_data.pop("lastError", None)
                
                # Reset or preserve retry count
                if reset_retry:
                    new_data["retry"] = "0"
                
                # Update timestamp
                new_data["createdAt"] = str(int(time.time() * 1000))
                new_data["reprocessed"] = "true"
                
                # Add to main stream
                new_id = self.add(new_data)
                
                # Remove from DLQ
                self._client.xdel(self._dlq_key, msg.message_id)
                
                results.append((msg.message_id, new_id))
                
                logger.info(
                    f"Reprocessed DLQ message: {msg.task_id} "
                    f"({msg.message_id} -> {new_id})"
                )
            
            return results
            
        except redis.RedisError as e:
            logger.error(f"Failed to reprocess DLQ: {e}")
            raise
    
    def clear_dlq(self) -> int:
        """
        Clear all messages from dead letter queue.
        
        Use with caution - this permanently deletes all DLQ messages.
        
        Returns:
            Number of messages deleted
        """
        try:
            length = self.get_dlq_length()
            if length > 0:
                self._client.delete(self._dlq_key)
                logger.warning(f"Cleared {length} messages from DLQ")
            return length
        except redis.RedisError as e:
            logger.error(f"Failed to clear DLQ: {e}")
            raise
    
    def submit_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
    ) -> str:
        """
        Submit a task to the stream.
        
        Generic method for submitting any task type.
        
        Args:
            task_id: Unique task identifier
            task_type: Task type (FULL_PAGE, ROI, EVIDENCE_ENRICH, etc.)
            payload: Task payload data
            
        Returns:
            Message ID
        """
        data = {
            "taskId": task_id,
            "taskType": task_type,
            "retry": "0",
            "createdAt": str(int(time.time() * 1000)),
        }
        
        # Flatten payload into data
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                import json
                data[key] = json.dumps(value)
            else:
                data[key] = str(value)
        
        return self.add(data)
    
    # ==================== Stream Info ====================
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get stream information."""
        try:
            info = self._client.xinfo_stream(self._stream_key)
            return {
                "length": info["length"],
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except redis.RedisError as e:
            if "no such key" in str(e).lower():
                return {"length": 0, "first_entry": None, "last_entry": None, "groups": 0}
            raise
    
    def get_groups_info(self) -> List[Dict[str, Any]]:
        """Get consumer groups information."""
        try:
            groups = self._client.xinfo_groups(self._stream_key)
            return [
                {
                    "name": g["name"],
                    "consumers": g["consumers"],
                    "pending": g["pending"],
                    "last_delivered_id": g.get("last-delivered-id"),
                }
                for g in groups
            ]
        except redis.RedisError as e:
            if "no such key" in str(e).lower():
                return []
            raise
    
    def get_consumers_info(self) -> List[Dict[str, Any]]:
        """Get consumers information for this group."""
        try:
            consumers = self._client.xinfo_consumers(self._stream_key, self._group_name)
            return [
                {
                    "name": c["name"],
                    "pending": c["pending"],
                    "idle": c["idle"],
                }
                for c in consumers
            ]
        except redis.RedisError as e:
            if "no such key" in str(e).lower():
                return []
            raise
    
    def get_stream_length(self) -> int:
        """Get stream length."""
        try:
            return self._client.xlen(self._stream_key)
        except redis.RedisError:
            return 0
    
    def get_dlq_length(self) -> int:
        """Get DLQ length."""
        try:
            return self._client.xlen(self._dlq_key)
        except redis.RedisError:
            return 0

