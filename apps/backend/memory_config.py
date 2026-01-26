"""
Memory Backend Configuration
============================

Unified configuration for selecting between memory backends in Auto Claude.
Provides backend selection logic and status utilities.

Memory Strategy:
- PRIMARY: Selected via MEMORY_BACKEND env var (default: graphiti)
- SECONDARY: The non-primary backend (if enabled)
- FALLBACK: File-based (always available)

Supported Backends:
- graphiti: Graph-based memory using Graphiti + LadybugDB (requires API keys)
- vector_memory: Vector-based memory using vector-memory-mcp (local, no API keys)

Environment Variables:
    MEMORY_BACKEND: Primary memory backend to use (graphiti|vector_memory)
                   Default: "graphiti" for backward compatibility
    GRAPHITI_ENABLED: Set to "true" to enable Graphiti backend
    VECTOR_MEMORY_ENABLED: Set to "true" to enable vector-memory backend
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MemoryBackend(str, Enum):
    """Supported memory backends."""

    GRAPHITI = "graphiti"
    VECTOR_MEMORY = "vector_memory"


# Default backend for backward compatibility
DEFAULT_BACKEND = MemoryBackend.GRAPHITI


@dataclass
class MemoryBackendStatus:
    """Status information for a memory backend."""

    backend: MemoryBackend
    enabled: bool
    is_primary: bool
    available: bool
    reason: str = ""


def get_memory_backend() -> str:
    """
    Get the primary memory backend from environment configuration.

    Returns:
        Backend identifier string: "graphiti" or "vector_memory"

    Environment:
        MEMORY_BACKEND: Set to "vector_memory" or "graphiti" (default)

    Note:
        Returns "graphiti" by default for backward compatibility with existing
        configurations that don't set MEMORY_BACKEND explicitly.
    """
    backend_str = os.environ.get("MEMORY_BACKEND", "").lower().strip()

    if backend_str == "vector_memory":
        return MemoryBackend.VECTOR_MEMORY.value
    elif backend_str == "graphiti":
        return MemoryBackend.GRAPHITI.value
    else:
        # Default to graphiti for backward compatibility
        return DEFAULT_BACKEND.value


def get_memory_backend_enum() -> MemoryBackend:
    """
    Get the primary memory backend as enum value.

    Returns:
        MemoryBackend enum value
    """
    backend_str = get_memory_backend()
    return MemoryBackend(backend_str)


def is_graphiti_primary() -> bool:
    """
    Check if Graphiti is the primary memory backend.

    Returns:
        True if MEMORY_BACKEND is "graphiti" or not set (default)
    """
    return get_memory_backend() == MemoryBackend.GRAPHITI.value


def is_vector_memory_primary() -> bool:
    """
    Check if vector-memory is the primary memory backend.

    Returns:
        True if MEMORY_BACKEND is "vector_memory"
    """
    return get_memory_backend() == MemoryBackend.VECTOR_MEMORY.value


def is_backend_enabled(backend: MemoryBackend) -> bool:
    """
    Check if a specific backend is enabled via environment variables.

    Args:
        backend: The backend to check

    Returns:
        True if the backend is enabled
    """
    if backend == MemoryBackend.GRAPHITI:
        enabled_str = os.environ.get("GRAPHITI_ENABLED", "").lower()
        return enabled_str in ("true", "1", "yes")
    elif backend == MemoryBackend.VECTOR_MEMORY:
        enabled_str = os.environ.get("VECTOR_MEMORY_ENABLED", "").lower()
        return enabled_str in ("true", "1", "yes")
    return False


def get_enabled_backends() -> list[MemoryBackend]:
    """
    Get list of all enabled memory backends.

    Returns:
        List of enabled MemoryBackend enum values
    """
    enabled = []
    for backend in MemoryBackend:
        if is_backend_enabled(backend):
            enabled.append(backend)
    return enabled


def get_fallback_order() -> list[str]:
    """
    Get the memory backend fallback order based on configuration.

    The fallback order is:
    1. Primary backend (from MEMORY_BACKEND)
    2. Secondary backend (the other one, if enabled)
    3. File-based (always available, represented as "file")

    Returns:
        List of backend identifiers in fallback order
    """
    primary = get_memory_backend()
    fallback_order = [primary]

    # Add the secondary backend if enabled
    secondary = (
        MemoryBackend.VECTOR_MEMORY.value
        if primary == MemoryBackend.GRAPHITI.value
        else MemoryBackend.GRAPHITI.value
    )

    if is_backend_enabled(MemoryBackend(secondary)):
        fallback_order.append(secondary)

    # File-based is always the ultimate fallback
    fallback_order.append("file")

    return fallback_order


def get_backend_status(backend: MemoryBackend) -> MemoryBackendStatus:
    """
    Get detailed status for a specific backend.

    Args:
        backend: The backend to check

    Returns:
        MemoryBackendStatus with availability information
    """
    enabled = is_backend_enabled(backend)
    is_primary = get_memory_backend() == backend.value
    available = False
    reason = ""

    if not enabled:
        reason = f"{backend.value.upper()}_ENABLED is not set to true"
    else:
        # Check if backend is actually available (imports work, etc.)
        if backend == MemoryBackend.GRAPHITI:
            try:
                from integrations.graphiti.config import GraphitiConfig

                config = GraphitiConfig.from_env()
                if config.is_valid():
                    available = True
                else:
                    errors = config.get_validation_errors()
                    reason = errors[0] if errors else "Graphiti configuration invalid"
            except ImportError as e:
                reason = f"Graphiti packages not available: {e}"
        elif backend == MemoryBackend.VECTOR_MEMORY:
            try:
                from integrations.vector_memory.config import VectorMemoryConfig

                config = VectorMemoryConfig.from_env()
                if config.is_valid():
                    available = True
                else:
                    errors = config.get_validation_errors()
                    reason = (
                        errors[0] if errors else "Vector-memory configuration invalid"
                    )
            except ImportError as e:
                reason = f"Vector-memory module not available: {e}"

    return MemoryBackendStatus(
        backend=backend,
        enabled=enabled,
        is_primary=is_primary,
        available=available,
        reason=reason,
    )


def get_all_backends_status() -> dict[str, MemoryBackendStatus]:
    """
    Get status for all memory backends.

    Returns:
        Dict mapping backend names to their status
    """
    return {backend.value: get_backend_status(backend) for backend in MemoryBackend}


def get_available_primary_backend() -> Optional[str]:
    """
    Get the first available backend in the fallback order.

    This function checks the fallback order and returns the first backend
    that is both enabled and available (can be imported and configured).

    Returns:
        Backend identifier string, or None if no backends available
    """
    for backend_str in get_fallback_order():
        if backend_str == "file":
            # File-based fallback is always available but not a "primary" backend
            return None

        try:
            backend = MemoryBackend(backend_str)
            status = get_backend_status(backend)
            if status.available:
                return backend_str
        except ValueError:
            continue

    return None


def get_memory_config_summary() -> dict:
    """
    Get a summary of memory configuration for logging/debugging.

    Returns:
        Dict with configuration summary
    """
    primary = get_memory_backend()
    enabled = get_enabled_backends()
    fallback = get_fallback_order()
    available_primary = get_available_primary_backend()

    return {
        "primary_backend": primary,
        "enabled_backends": [b.value for b in enabled],
        "fallback_order": fallback,
        "available_primary": available_primary,
        "status": {
            backend.value: {
                "enabled": is_backend_enabled(backend),
                "is_primary": primary == backend.value,
            }
            for backend in MemoryBackend
        },
    }


def validate_memory_config() -> tuple[bool, list[str]]:
    """
    Validate the overall memory configuration.

    Returns:
        Tuple of (is_valid, warnings)
        - is_valid: True if at least one backend is available or file fallback works
        - warnings: List of configuration warnings
    """
    warnings = []
    primary = get_memory_backend()
    primary_status = get_backend_status(MemoryBackend(primary))

    if not primary_status.enabled:
        warnings.append(
            f"Primary backend '{primary}' is not enabled. "
            f"Set {primary.upper()}_ENABLED=true to enable it."
        )
    elif not primary_status.available:
        warnings.append(
            f"Primary backend '{primary}' is enabled but not available: "
            f"{primary_status.reason}"
        )

    # Check if any backend is available
    available_backend = get_available_primary_backend()
    if not available_backend:
        enabled_count = len(get_enabled_backends())
        if enabled_count == 0:
            warnings.append(
                "No memory backends enabled. Memory will use file-based fallback. "
                "Set GRAPHITI_ENABLED=true or VECTOR_MEMORY_ENABLED=true to enable a backend."
            )
        else:
            warnings.append(
                f"{enabled_count} backend(s) enabled but none available. "
                "Memory will use file-based fallback."
            )

    # Memory system is always valid (file fallback always works)
    return True, warnings
