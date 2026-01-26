"""
Vector Memory Adapter
=====================

Adapter to translate memory interface operations to vector-memory-mcp tools.
Provides a similar interface to GraphitiMemory for seamless integration.

This adapter works in two modes:
1. HTTP mode (Docker): Makes direct HTTP calls to the vector-memory server
2. npx mode: Agents use MCP tools directly - adapter provides utility formatting

The adapter follows the same async patterns as graphiti_helpers.py for consistency.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from core.sentry import capture_exception

from .config import VectorMemoryConfig

logger = logging.getLogger(__name__)


class VectorMemoryAdapter:
    """
    Adapter for vector-memory-mcp memory operations.

    Provides a GraphitiMemory-compatible interface for vector-memory backend.
    Supports both HTTP mode (Docker) and npx mode (MCP tools via agent).

    Usage:
        adapter = VectorMemoryAdapter(spec_dir, project_dir)
        await adapter.initialize()

        # Store session insights
        await adapter.save_session_insights(session_num, insights)

        # Retrieve relevant context
        context = await adapter.get_relevant_context(query)

        # Always close when done
        await adapter.close()
    """

    def __init__(
        self,
        spec_dir: Path,
        project_dir: Path,
    ):
        """
        Initialize the vector memory adapter.

        Args:
            spec_dir: Spec directory for namespacing memories
            project_dir: Project root directory
        """
        self.spec_dir = spec_dir
        self.project_dir = project_dir
        self.config = VectorMemoryConfig.from_env()

        # HTTP client for Docker mode
        self._http_session: Any = None
        self._initialized = False
        self._available = self.config.is_valid()

        # Namespace for memories (scoped to project)
        self._namespace = f"project_{project_dir.name}"

    @property
    def is_enabled(self) -> bool:
        """Check if vector memory integration is enabled and configured."""
        return self._available

    @property
    def is_initialized(self) -> bool:
        """Check if adapter has been initialized."""
        return self._initialized

    async def initialize(self) -> bool:
        """
        Initialize the adapter.

        In HTTP mode, this creates an HTTP client session.
        In npx mode, this is a no-op (MCP tools are used directly by agents).

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        if not self._available:
            logger.info("Vector memory not available - skipping initialization")
            return False

        try:
            if self.config.is_docker_mode():
                # Import aiohttp only when needed (lazy import)
                try:
                    import aiohttp

                    self._http_session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=30)
                    )
                except ImportError:
                    logger.warning(
                        "aiohttp not installed - HTTP mode unavailable. "
                        "Install with: pip install aiohttp"
                    )
                    return False

            self._initialized = True
            logger.info(
                f"Vector memory adapter initialized "
                f"(mode: {'http' if self.config.is_docker_mode() else 'npx'}, "
                f"namespace: {self._namespace})"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize vector memory adapter: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="initialize",
            )
            return False

    async def close(self) -> None:
        """Close the adapter and clean up resources."""
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

        self._initialized = False

    async def save_session_insights(
        self,
        session_num: int,
        insights: dict,
    ) -> bool:
        """
        Save session insights to vector memory.

        Args:
            session_num: Session number
            insights: Session insights dictionary

        Returns:
            True if save succeeded
        """
        if not await self._ensure_initialized():
            return False

        try:
            # Format insights as a memory entry
            memory_text = self._format_session_insights(session_num, insights)
            memory_id = f"session_{self._namespace}_{session_num}"

            metadata = {
                "type": "session_insight",
                "session_num": session_num,
                "namespace": self._namespace,
                "spec": self.spec_dir.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return await self._store_memory(memory_id, memory_text, metadata)

        except Exception as e:
            logger.warning(f"Failed to save session insights: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="save_session_insights",
                session_num=session_num,
            )
            return False

    async def save_pattern(self, pattern: str) -> bool:
        """
        Save a code pattern to vector memory.

        Args:
            pattern: Pattern description to save

        Returns:
            True if save succeeded
        """
        if not await self._ensure_initialized():
            return False

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            memory_id = f"pattern_{self._namespace}_{hash(pattern) & 0xFFFFFFFF}"

            metadata = {
                "type": "pattern",
                "namespace": self._namespace,
                "spec": self.spec_dir.name,
                "timestamp": timestamp,
            }

            return await self._store_memory(
                memory_id, f"Pattern: {pattern}", metadata
            )

        except Exception as e:
            logger.warning(f"Failed to save pattern: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="save_pattern",
            )
            return False

    async def save_gotcha(self, gotcha: str) -> bool:
        """
        Save a gotcha (pitfall) to vector memory.

        Args:
            gotcha: Gotcha description to save

        Returns:
            True if save succeeded
        """
        if not await self._ensure_initialized():
            return False

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            memory_id = f"gotcha_{self._namespace}_{hash(gotcha) & 0xFFFFFFFF}"

            metadata = {
                "type": "gotcha",
                "namespace": self._namespace,
                "spec": self.spec_dir.name,
                "timestamp": timestamp,
            }

            return await self._store_memory(
                memory_id, f"Gotcha: {gotcha}", metadata
            )

        except Exception as e:
            logger.warning(f"Failed to save gotcha: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="save_gotcha",
            )
            return False

    async def get_relevant_context(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[dict]:
        """
        Search for relevant context based on a query.

        Args:
            query: Search query
            num_results: Maximum number of results

        Returns:
            List of relevant memory entries
        """
        if not await self._ensure_initialized():
            return []

        try:
            return await self._search_memories(query, num_results)

        except Exception as e:
            logger.warning(f"Failed to get relevant context: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="get_relevant_context",
            )
            return []

    async def get_patterns_and_gotchas(
        self,
        query: str,
        num_results: int = 5,
        min_score: float = 0.5,
    ) -> tuple[list[dict], list[dict]]:
        """
        Get patterns and gotchas relevant to the query.

        Args:
            query: Search query
            num_results: Max results per type
            min_score: Minimum relevance score (0.0-1.0)

        Returns:
            Tuple of (patterns, gotchas) lists
        """
        if not await self._ensure_initialized():
            return [], []

        try:
            # Search for all relevant memories
            all_results = await self._search_memories(
                query, num_results * 2  # Get more to filter by type
            )

            patterns = []
            gotchas = []

            for result in all_results:
                result_type = result.get("type", "")
                score = result.get("score", 1.0)

                if score < min_score:
                    continue

                if result_type == "pattern" and len(patterns) < num_results:
                    patterns.append({
                        "pattern": result.get("content", "").replace("Pattern: ", ""),
                        "applies_to": result.get("spec", ""),
                    })
                elif result_type == "gotcha" and len(gotchas) < num_results:
                    gotchas.append({
                        "gotcha": result.get("content", "").replace("Gotcha: ", ""),
                        "solution": "",  # Vector memory doesn't store solutions separately
                    })

            return patterns, gotchas

        except Exception as e:
            logger.warning(f"Failed to get patterns and gotchas: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="get_patterns_and_gotchas",
            )
            return [], []

    async def get_session_history(
        self,
        limit: int = 5,
    ) -> list[dict]:
        """
        Get recent session insights.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session insight dictionaries
        """
        if not await self._ensure_initialized():
            return []

        try:
            # Search for session insights
            results = await self._search_memories(
                "session insight recommendations",
                limit * 2  # Get more to filter
            )

            sessions = []
            for result in results:
                if result.get("type") == "session_insight" and len(sessions) < limit:
                    sessions.append({
                        "session_number": result.get("session_num", 0),
                        "recommendations_for_next_session": [],
                        "content": result.get("content", ""),
                    })

            return sessions

        except Exception as e:
            logger.warning(f"Failed to get session history: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="get_session_history",
            )
            return []

    def get_status_summary(self) -> dict:
        """
        Get a summary of vector memory status.

        Returns:
            Dict with status information
        """
        return {
            "enabled": self.is_enabled,
            "initialized": self.is_initialized,
            "mode": "http" if self.config.is_docker_mode() else "npx",
            "url": self.config.url if self.config.is_docker_mode() else None,
            "db_path": str(self.config.get_db_path()),
            "model": self.config.model,
            "namespace": self._namespace,
        }

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    async def _ensure_initialized(self) -> bool:
        """
        Ensure adapter is initialized, attempting initialization if needed.

        Returns:
            True if initialized and ready
        """
        if self._initialized:
            return True

        if not self._available:
            return False

        return await self.initialize()

    def _format_session_insights(self, session_num: int, insights: dict) -> str:
        """Format session insights as a text memory entry."""
        parts = [f"Session {session_num} Insights:"]

        # What worked
        what_worked = insights.get("what_worked", [])
        if what_worked:
            parts.append("What worked:")
            for item in what_worked[:5]:
                parts.append(f"  - {item}")

        # What failed
        what_failed = insights.get("what_failed", [])
        if what_failed:
            parts.append("What failed:")
            for item in what_failed[:5]:
                parts.append(f"  - {item}")

        # Recommendations
        recommendations = insights.get("recommendations_for_next_session", [])
        if recommendations:
            parts.append("Recommendations:")
            for rec in recommendations[:5]:
                parts.append(f"  - {rec}")

        # Subtasks completed
        subtasks = insights.get("subtasks_completed", [])
        if subtasks:
            parts.append(f"Subtasks completed: {', '.join(subtasks)}")

        return "\n".join(parts)

    async def _store_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict,
    ) -> bool:
        """
        Store a memory entry.

        In HTTP mode, makes an API call to the vector-memory server.
        In npx mode, this is a no-op (MCP tools are used by agents).

        Args:
            memory_id: Unique identifier for the memory
            text: Memory text content
            metadata: Additional metadata

        Returns:
            True if store succeeded
        """
        if not self.config.is_docker_mode():
            # In npx mode, memories are stored via MCP tools by agents
            logger.debug(
                f"Vector memory (npx mode): Memory would be stored via MCP tools: {memory_id}"
            )
            return True

        if self._http_session is None:
            logger.warning("HTTP session not initialized for vector memory")
            return False

        try:
            url = urljoin(self.config.url.rstrip("/") + "/", "store")
            payload = {
                "id": memory_id,
                "text": text,
                "metadata": metadata,
            }

            async with self._http_session.post(url, json=payload) as response:
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Vector memory store failed: {response.status} - {error_text}"
                    )
                    return False

        except Exception as e:
            logger.warning(f"Vector memory HTTP store failed: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="_store_memory",
                memory_id=memory_id,
            )
            return False

    async def _search_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for memories by query.

        In HTTP mode, makes an API call to the vector-memory server.
        In npx mode, returns empty (MCP tools are used by agents).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching memory entries
        """
        if not self.config.is_docker_mode():
            # In npx mode, searches are done via MCP tools by agents
            logger.debug(
                f"Vector memory (npx mode): Search would be done via MCP tools: {query[:50]}"
            )
            return []

        if self._http_session is None:
            logger.warning("HTTP session not initialized for vector memory")
            return []

        try:
            url = urljoin(self.config.url.rstrip("/") + "/", "search")
            payload = {
                "query": query,
                "limit": limit,
                "namespace": self._namespace,
            }

            async with self._http_session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._normalize_search_results(data)
                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Vector memory search failed: {response.status} - {error_text}"
                    )
                    return []

        except Exception as e:
            logger.warning(f"Vector memory HTTP search failed: {e}")
            capture_exception(
                e,
                component="vector_memory",
                operation="_search_memories",
            )
            return []

    def _normalize_search_results(self, data: Any) -> list[dict]:
        """
        Normalize search results to a consistent format.

        Args:
            data: Raw response data from vector-memory server

        Returns:
            Normalized list of memory entries
        """
        results = []

        # Handle different response formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("results", data.get("memories", []))
        else:
            return []

        for item in items:
            if not isinstance(item, dict):
                continue

            # Extract metadata
            metadata = item.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            results.append({
                "content": item.get("text", item.get("content", "")),
                "type": metadata.get("type", "unknown"),
                "score": item.get("score", item.get("similarity", 1.0)),
                "session_num": metadata.get("session_num"),
                "spec": metadata.get("spec", ""),
                "timestamp": metadata.get("timestamp", ""),
            })

        return results


# Convenience function for getting a vector memory adapter
def get_vector_memory_adapter(
    spec_dir: Path,
    project_dir: Path,
) -> VectorMemoryAdapter:
    """
    Get a VectorMemoryAdapter instance for the given spec.

    This is the main entry point for other modules.

    Args:
        spec_dir: Spec directory
        project_dir: Project root directory

    Returns:
        VectorMemoryAdapter instance
    """
    return VectorMemoryAdapter(spec_dir, project_dir)


async def get_initialized_adapter(
    spec_dir: Path,
    project_dir: Path,
) -> VectorMemoryAdapter | None:
    """
    Get an initialized VectorMemoryAdapter instance if available.

    Args:
        spec_dir: Spec directory
        project_dir: Project root directory

    Returns:
        Initialized VectorMemoryAdapter or None if not available
    """
    config = VectorMemoryConfig.from_env()
    if not config.is_valid():
        return None

    adapter = VectorMemoryAdapter(spec_dir, project_dir)
    if await adapter.initialize():
        return adapter

    return None


def is_vector_memory_enabled() -> bool:
    """
    Check if vector memory integration is available.

    Returns:
        True if VECTOR_MEMORY_ENABLED is set and configuration is valid
    """
    config = VectorMemoryConfig.from_env()
    return config.is_valid()
