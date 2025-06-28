#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Memory Management System
Advanced memory and state management for AI agents
"""

import uuid
import time
import json
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    SHARED = "shared"
    PROCEDURAL = "procedural"


class MemoryPriority(Enum):
    """Memory priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    priority: MemoryPriority = MemoryPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    expires_at: Optional[float] = None
    embedding: Optional[List[float]] = None


@dataclass
class MemoryQuery:
    """Memory query specification"""
    content: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    tags: Optional[List[str]] = None
    priority: Optional[MemoryPriority] = None
    time_range: Optional[tuple] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    include_expired: bool = False


class BaseMemoryManager(ABC):
    """Base class for memory management systems"""
    
    def __init__(
        self,
        max_capacity: int = 10000,
        cleanup_interval: float = 3600,  # 1 hour
        enable_persistence: bool = True,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize memory manager
        
        Args:
            max_capacity: Maximum number of memory items
            cleanup_interval: Interval for automatic cleanup in seconds
            enable_persistence: Whether to persist memory to disk
            persistence_path: Path for memory persistence
        """
        self.id = str(uuid.uuid4())
        self.max_capacity = max_capacity
        self.cleanup_interval = cleanup_interval
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path
        
        # Memory storage
        self.memories: Dict[str, MemoryItem] = {}
        self.memory_index: Dict[MemoryType, List[str]] = {
            memory_type: [] for memory_type in MemoryType
        }
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "total_accesses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cleanup_runs": 0,
            "last_cleanup": time.time()
        }
        
        # Callbacks
        self.on_memory_added: Optional[Callable] = None
        self.on_memory_removed: Optional[Callable] = None
        self.on_memory_accessed: Optional[Callable] = None
        
        logger.info(f"Memory manager {self.id} initialized with capacity {max_capacity}")
    
    @abstractmethod
    def store(self, content: Any, memory_type: MemoryType = MemoryType.SHORT_TERM, **kwargs) -> str:
        """Store content in memory"""
        pass
    
    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory by ID"""
        pass
    
    @abstractmethod
    def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search memories based on query"""
        pass
    
    @abstractmethod
    def forget(self, memory_id: str) -> bool:
        """Remove memory by ID"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        current_time = time.time()
        
        # Calculate memory distribution
        memory_distribution = {}
        for memory_type in MemoryType:
            memory_distribution[memory_type.value] = len(self.memory_index[memory_type])
        
        # Calculate average access frequency
        total_accesses = sum(item.access_count for item in self.memories.values())
        avg_access_frequency = total_accesses / len(self.memories) if self.memories else 0
        
        return {
            "manager_id": self.id,
            "total_memories": len(self.memories),
            "capacity_used": len(self.memories) / self.max_capacity,
            "memory_distribution": memory_distribution,
            "average_access_frequency": avg_access_frequency,
            "uptime": current_time - self.stats["last_cleanup"],
            **self.stats
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired memories"""
        current_time = time.time()
        expired_ids = []
        
        for memory_id, memory_item in self.memories.items():
            if memory_item.expires_at and memory_item.expires_at < current_time:
                expired_ids.append(memory_id)
        
        for memory_id in expired_ids:
            self.forget(memory_id)
        
        self.stats["cleanup_runs"] += 1
        self.stats["last_cleanup"] = current_time
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired memories")
        
        return len(expired_ids)
    
    def clear_all(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear all memories or memories of specific type"""
        if memory_type:
            memory_ids = self.memory_index[memory_type].copy()
            for memory_id in memory_ids:
                self.forget(memory_id)
            return len(memory_ids)
        else:
            count = len(self.memories)
            self.memories.clear()
            for memory_type in MemoryType:
                self.memory_index[memory_type].clear()
            return count


class InMemoryManager(BaseMemoryManager):
    """In-memory implementation of memory manager"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_function: Optional[Callable] = None
    
    def store(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        tags: Optional[List[str]] = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        expires_in: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store content in memory"""
        
        # Check capacity
        if len(self.memories) >= self.max_capacity:
            self._evict_least_important()
        
        # Create memory item
        memory_item = MemoryItem(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            priority=priority,
            metadata=metadata or {},
            expires_at=time.time() + expires_in if expires_in else None
        )
        
        # Generate embedding if function available
        if self.embedding_function and isinstance(content, str):
            try:
                memory_item.embedding = self.embedding_function(content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}")
        
        # Store memory
        self.memories[memory_item.id] = memory_item
        self.memory_index[memory_type].append(memory_item.id)
        
        # Update statistics
        self.stats["total_items"] += 1
        
        # Trigger callback
        if self.on_memory_added:
            self.on_memory_added(memory_item)
        
        logger.debug(f"Stored memory {memory_item.id} of type {memory_type.value}")
        return memory_item.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory by ID"""
        memory_item = self.memories.get(memory_id)
        
        if memory_item:
            # Check if expired
            if memory_item.expires_at and memory_item.expires_at < time.time():
                self.forget(memory_id)
                self.stats["cache_misses"] += 1
                return None
            
            # Update access statistics
            memory_item.accessed_at = time.time()
            memory_item.access_count += 1
            self.stats["total_accesses"] += 1
            self.stats["cache_hits"] += 1
            
            # Trigger callback
            if self.on_memory_accessed:
                self.on_memory_accessed(memory_item)
            
            return memory_item
        else:
            self.stats["cache_misses"] += 1
            return None
    
    def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search memories based on query"""
        results = []
        current_time = time.time()
        
        for memory_item in self.memories.values():
            # Skip expired memories unless explicitly requested
            if (not query.include_expired and 
                memory_item.expires_at and 
                memory_item.expires_at < current_time):
                continue
            
            # Filter by memory type
            if query.memory_type and memory_item.memory_type != query.memory_type:
                continue
            
            # Filter by priority
            if query.priority and memory_item.priority != query.priority:
                continue
            
            # Filter by time range
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= memory_item.created_at <= end_time):
                    continue
            
            # Filter by tags
            if query.tags:
                if not any(tag in memory_item.tags for tag in query.tags):
                    continue
            
            # Content similarity search
            if query.content:
                similarity_score = self._calculate_similarity(query.content, memory_item)
                if similarity_score < query.similarity_threshold:
                    continue
                memory_item.metadata["similarity_score"] = similarity_score
            
            results.append(memory_item)
        
        # Sort by relevance (similarity score, then access count, then recency)
        results.sort(key=lambda x: (
            x.metadata.get("similarity_score", 0),
            x.access_count,
            x.accessed_at
        ), reverse=True)
        
        # Limit results
        return results[:query.limit]
    
    def forget(self, memory_id: str) -> bool:
        """Remove memory by ID"""
        memory_item = self.memories.get(memory_id)
        
        if memory_item:
            # Remove from main storage
            del self.memories[memory_id]
            
            # Remove from index
            if memory_id in self.memory_index[memory_item.memory_type]:
                self.memory_index[memory_item.memory_type].remove(memory_id)
            
            # Trigger callback
            if self.on_memory_removed:
                self.on_memory_removed(memory_item)
            
            logger.debug(f"Forgot memory {memory_id}")
            return True
        
        return False
    
    def set_embedding_function(self, embedding_function: Callable[[str], List[float]]) -> None:
        """Set function for generating embeddings"""
        self.embedding_function = embedding_function
        logger.info("Embedding function set for memory manager")
    
    def _evict_least_important(self) -> None:
        """Evict least important memory to make space"""
        if not self.memories:
            return
        
        # Find least important memory (lowest priority, least accessed, oldest)
        least_important = min(
            self.memories.values(),
            key=lambda x: (x.priority.value, x.access_count, -x.created_at)
        )
        
        self.forget(least_important.id)
        logger.debug(f"Evicted memory {least_important.id} to make space")
    
    def _calculate_similarity(self, query_content: str, memory_item: MemoryItem) -> float:
        """Calculate similarity between query and memory item"""
        # Simple text similarity if no embeddings available
        if not memory_item.embedding or not isinstance(memory_item.content, str):
            # Basic text overlap similarity
            query_words = set(query_content.lower().split())
            content_words = set(str(memory_item.content).lower().split())
            
            if not query_words or not content_words:
                return 0.0
            
            intersection = query_words.intersection(content_words)
            union = query_words.union(content_words)
            
            return len(intersection) / len(union) if union else 0.0
        
        # Embedding-based similarity
        if self.embedding_function:
            try:
                query_embedding = self.embedding_function(query_content)
                return self._cosine_similarity(query_embedding, memory_item.embedding)
            except Exception as e:
                logger.warning(f"Failed to calculate embedding similarity: {str(e)}")
                return 0.0
        
        return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class SharedMemory:
    """Shared memory system for crew collaboration"""
    
    def __init__(self, crew_id: str):
        """
        Initialize shared memory for a crew
        
        Args:
            crew_id: ID of the crew this memory belongs to
        """
        self.crew_id = crew_id
        self.memory_manager = InMemoryManager(max_capacity=5000)
        self.access_log: List[Dict[str, Any]] = []
        self.locks: Dict[str, bool] = {}
        
        logger.info(f"Shared memory initialized for crew {crew_id}")
    
    def share(
        self,
        content: Any,
        shared_by: str,
        tags: Optional[List[str]] = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM
    ) -> str:
        """Share content with the crew"""
        
        memory_id = self.memory_manager.store(
            content=content,
            memory_type=MemoryType.SHARED,
            tags=tags,
            priority=priority,
            metadata={
                "shared_by": shared_by,
                "crew_id": self.crew_id,
                "shared_at": time.time()
            }
        )
        
        # Log access
        self.access_log.append({
            "action": "share",
            "memory_id": memory_id,
            "agent_id": shared_by,
            "timestamp": time.time()
        })
        
        logger.info(f"Agent {shared_by} shared memory {memory_id} with crew {self.crew_id}")
        return memory_id
    
    def access(self, memory_id: str, accessed_by: str) -> Optional[MemoryItem]:
        """Access shared memory"""
        
        # Check if memory is locked
        if self.locks.get(memory_id, False):
            logger.warning(f"Memory {memory_id} is locked, access denied for {accessed_by}")
            return None
        
        memory_item = self.memory_manager.retrieve(memory_id)
        
        if memory_item:
            # Log access
            self.access_log.append({
                "action": "access",
                "memory_id": memory_id,
                "agent_id": accessed_by,
                "timestamp": time.time()
            })
            
            logger.debug(f"Agent {accessed_by} accessed memory {memory_id}")
        
        return memory_item
    
    def search_shared(self, query: MemoryQuery, accessed_by: str) -> List[MemoryItem]:
        """Search shared memories"""
        
        # Ensure we only search shared memories
        query.memory_type = MemoryType.SHARED
        
        results = self.memory_manager.search(query)
        
        # Log search
        self.access_log.append({
            "action": "search",
            "query": query.__dict__,
            "agent_id": accessed_by,
            "results_count": len(results),
            "timestamp": time.time()
        })
        
        return results
    
    def lock_memory(self, memory_id: str, locked_by: str) -> bool:
        """Lock memory for exclusive access"""
        if memory_id not in self.locks or not self.locks[memory_id]:
            self.locks[memory_id] = True
            
            self.access_log.append({
                "action": "lock",
                "memory_id": memory_id,
                "agent_id": locked_by,
                "timestamp": time.time()
            })
            
            logger.info(f"Memory {memory_id} locked by {locked_by}")
            return True
        
        return False
    
    def unlock_memory(self, memory_id: str, unlocked_by: str) -> bool:
        """Unlock memory"""
        if memory_id in self.locks and self.locks[memory_id]:
            self.locks[memory_id] = False
            
            self.access_log.append({
                "action": "unlock",
                "memory_id": memory_id,
                "agent_id": unlocked_by,
                "timestamp": time.time()
            })
            
            logger.info(f"Memory {memory_id} unlocked by {unlocked_by}")
            return True
        
        return False
    
    def get_access_log(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access log, optionally filtered by agent"""
        
        if agent_id:
            filtered_log = [entry for entry in self.access_log if entry["agent_id"] == agent_id]
        else:
            filtered_log = self.access_log
        
        return filtered_log[-limit:]
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        
        # Analyze access patterns
        agent_activity = {}
        action_counts = {}
        
        for entry in self.access_log:
            agent_id = entry["agent_id"]
            action = entry["action"]
            
            if agent_id not in agent_activity:
                agent_activity[agent_id] = {"shares": 0, "accesses": 0, "searches": 0}
            
            if action == "share":
                agent_activity[agent_id]["shares"] += 1
            elif action == "access":
                agent_activity[agent_id]["accesses"] += 1
            elif action == "search":
                agent_activity[agent_id]["searches"] += 1
            
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "crew_id": self.crew_id,
            "total_memories": len(self.memory_manager.memories),
            "total_actions": len(self.access_log),
            "agent_activity": agent_activity,
            "action_distribution": action_counts,
            "active_locks": sum(1 for locked in self.locks.values() if locked),
            "memory_stats": self.memory_manager.get_stats()
        }
