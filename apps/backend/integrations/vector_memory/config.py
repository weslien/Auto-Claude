"""
Vector Memory Configuration
===========================

Configuration dataclass for vector-memory-mcp integration.
Enables local embedding generation via sentence-transformers without cloud API keys.

Environment Variables:
    VECTOR_MEMORY_ENABLED: Set to "true" to enable vector-memory integration
    MEMORY_BACKEND: Set to "vector_memory" to use as primary backend (default: graphiti)
    VECTOR_MEMORY_DB_PATH: SQLite database path (default: .vector-memory/memories.db)
    VECTOR_MEMORY_MODEL: Embedding model (default: Xenova/all-MiniLM-L6-v2)
    VECTOR_MEMORY_URL: Optional HTTP URL for Docker mode (overrides npx execution)
"""

import os
from dataclasses import dataclass
from pathlib import Path


# Default configuration values
DEFAULT_DB_PATH = ".vector-memory/memories.db"
DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2"


@dataclass
class VectorMemoryConfig:
    """Configuration for vector-memory-mcp integration.

    Enables fully offline operation without any cloud API keys.
    Uses sentence-transformers for local embedding generation.
    """

    # Core settings
    enabled: bool = False

    # Database settings
    db_path: str = DEFAULT_DB_PATH

    # Model settings
    model: str = DEFAULT_MODEL

    # Connection mode: npx (default) or http URL (Docker)
    url: str = ""  # If set, use HTTP mode instead of npx

    @classmethod
    def from_env(cls) -> "VectorMemoryConfig":
        """Create config from environment variables."""
        # Check if vector-memory is explicitly enabled
        enabled_str = os.environ.get("VECTOR_MEMORY_ENABLED", "").lower()
        enabled = enabled_str in ("true", "1", "yes")

        # Database path
        db_path = os.environ.get("VECTOR_MEMORY_DB_PATH", DEFAULT_DB_PATH)

        # Model for embeddings
        model = os.environ.get("VECTOR_MEMORY_MODEL", DEFAULT_MODEL)

        # Optional URL for Docker mode
        url = os.environ.get("VECTOR_MEMORY_URL", "")

        return cls(
            enabled=enabled,
            db_path=db_path,
            model=model,
            url=url,
        )

    def is_valid(self) -> bool:
        """Check if config has minimum required values for operation."""
        return self.enabled

    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors for current configuration."""
        errors = []

        if not self.enabled:
            errors.append("VECTOR_MEMORY_ENABLED must be set to true")

        return errors

    def get_db_path(self) -> Path:
        """Get the resolved database path."""
        return Path(self.db_path).expanduser()

    def is_docker_mode(self) -> bool:
        """Check if running in Docker HTTP mode."""
        return bool(self.url)


def is_vector_memory_enabled() -> bool:
    """Quick check if vector-memory integration is enabled."""
    config = VectorMemoryConfig.from_env()
    return config.is_valid()


def validate_vector_memory_config() -> tuple[bool, list[str]]:
    """
    Validate vector-memory configuration from environment.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    config = VectorMemoryConfig.from_env()

    if not config.is_valid():
        errors = config.get_validation_errors()
        return False, errors

    return True, []
