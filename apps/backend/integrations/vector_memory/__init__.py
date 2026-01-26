"""
Vector Memory Integration
=========================

Integration with vector-memory-mcp for local semantic memory without cloud API keys.
Uses sentence-transformers for embedding generation, enabling fully offline operation.
"""

# Config imports don't require external packages
from .config import VectorMemoryConfig, validate_vector_memory_config

# Lazy imports for components that may require additional setup
__all__ = [
    "VectorMemoryConfig",
    "validate_vector_memory_config",
    "VectorMemoryAdapter",
]


def __getattr__(name):
    """Lazy import to avoid requiring adapter setup for config-only imports."""
    if name == "VectorMemoryAdapter":
        from .adapter import VectorMemoryAdapter

        return VectorMemoryAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
