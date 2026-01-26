"""
End-to-End Tests for Vector Memory Integration
==============================================

These tests verify the complete vector-memory flow:
1. Configuration loading from environment
2. Backend selection logic
3. Memory adapter initialization
4. Memory operations (save/retrieve)
5. Fallback chain behavior

To run with vector-memory enabled:
    VECTOR_MEMORY_ENABLED=true MEMORY_BACKEND=vector_memory python -m pytest tests/test_vector_memory_e2e.py -v

Note: Full E2E testing with the actual MCP server requires:
    - npx -y @weslien/vector-memory-mcp --working-dir /path/to/project
    - Or Docker: docker run -p 3000:3000 ghcr.io/weslien/vector-memory-mcp
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure apps/backend is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "backend"))


class TestVectorMemoryConfig:
    """Test vector memory configuration loading."""

    def test_config_from_env_disabled(self):
        """Test config loading when vector-memory is disabled."""
        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "false"}, clear=False):
            from integrations.vector_memory.config import VectorMemoryConfig

            config = VectorMemoryConfig.from_env()
            assert config.enabled is False
            assert config.is_valid() is False

    def test_config_from_env_enabled(self):
        """Test config loading when vector-memory is enabled."""
        with patch.dict(
            os.environ,
            {
                "VECTOR_MEMORY_ENABLED": "true",
                "VECTOR_MEMORY_DB_PATH": ".test-memory/memories.db",
                "VECTOR_MEMORY_MODEL": "Xenova/all-MiniLM-L6-v2",
            },
            clear=False,
        ):
            from integrations.vector_memory.config import VectorMemoryConfig

            config = VectorMemoryConfig.from_env()
            assert config.enabled is True
            assert config.is_valid() is True
            assert config.db_path == ".test-memory/memories.db"
            assert config.model == "Xenova/all-MiniLM-L6-v2"

    def test_config_docker_mode(self):
        """Test config in Docker mode with URL."""
        with patch.dict(
            os.environ,
            {
                "VECTOR_MEMORY_ENABLED": "true",
                "VECTOR_MEMORY_URL": "http://localhost:3000",
            },
            clear=False,
        ):
            from integrations.vector_memory.config import VectorMemoryConfig

            config = VectorMemoryConfig.from_env()
            assert config.is_docker_mode() is True
            assert config.url == "http://localhost:3000"

    def test_config_npx_mode(self):
        """Test config in npx mode (no URL)."""
        with patch.dict(
            os.environ,
            {
                "VECTOR_MEMORY_ENABLED": "true",
                "VECTOR_MEMORY_URL": "",
            },
            clear=False,
        ):
            from integrations.vector_memory.config import VectorMemoryConfig

            config = VectorMemoryConfig.from_env()
            assert config.is_docker_mode() is False

    def test_validation_errors(self):
        """Test validation error messages."""
        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "false"}, clear=False):
            from integrations.vector_memory.config import VectorMemoryConfig

            config = VectorMemoryConfig.from_env()
            errors = config.get_validation_errors()
            assert len(errors) > 0
            assert "VECTOR_MEMORY_ENABLED" in errors[0]


class TestMemoryBackendSelection:
    """Test memory backend selection logic."""

    def test_default_backend_is_graphiti(self):
        """Test that default backend is graphiti for backward compatibility."""
        with patch.dict(os.environ, {"MEMORY_BACKEND": ""}, clear=False):
            from memory_config import get_memory_backend

            assert get_memory_backend() == "graphiti"

    def test_select_vector_memory_backend(self):
        """Test selecting vector_memory as primary backend."""
        with patch.dict(
            os.environ, {"MEMORY_BACKEND": "vector_memory"}, clear=False
        ):
            from memory_config import get_memory_backend, is_vector_memory_primary

            assert get_memory_backend() == "vector_memory"
            assert is_vector_memory_primary() is True

    def test_select_graphiti_backend(self):
        """Test selecting graphiti as primary backend."""
        with patch.dict(os.environ, {"MEMORY_BACKEND": "graphiti"}, clear=False):
            from memory_config import get_memory_backend, is_graphiti_primary

            assert get_memory_backend() == "graphiti"
            assert is_graphiti_primary() is True

    def test_fallback_order_vector_memory_primary(self):
        """Test fallback order when vector_memory is primary."""
        with patch.dict(
            os.environ,
            {
                "MEMORY_BACKEND": "vector_memory",
                "VECTOR_MEMORY_ENABLED": "true",
                "GRAPHITI_ENABLED": "true",
            },
            clear=False,
        ):
            from memory_config import get_fallback_order

            order = get_fallback_order()
            # vector_memory first, then graphiti, then file
            assert order[0] == "vector_memory"
            assert "graphiti" in order
            assert "file" in order

    def test_fallback_order_graphiti_primary(self):
        """Test fallback order when graphiti is primary."""
        with patch.dict(
            os.environ,
            {
                "MEMORY_BACKEND": "graphiti",
                "VECTOR_MEMORY_ENABLED": "true",
                "GRAPHITI_ENABLED": "true",
            },
            clear=False,
        ):
            from memory_config import get_fallback_order

            order = get_fallback_order()
            # graphiti first, then vector_memory, then file
            assert order[0] == "graphiti"
            assert "vector_memory" in order
            assert "file" in order

    def test_backend_enabled_check(self):
        """Test is_backend_enabled function."""
        with patch.dict(
            os.environ,
            {
                "VECTOR_MEMORY_ENABLED": "true",
                "GRAPHITI_ENABLED": "false",
            },
            clear=False,
        ):
            from memory_config import MemoryBackend, is_backend_enabled

            assert is_backend_enabled(MemoryBackend.VECTOR_MEMORY) is True
            assert is_backend_enabled(MemoryBackend.GRAPHITI) is False


class TestVectorMemoryAdapter:
    """Test the vector memory adapter."""

    def test_adapter_creation_disabled(self):
        """Test adapter creation when disabled."""
        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "false"}, clear=False):
            from integrations.vector_memory.adapter import VectorMemoryAdapter

            adapter = VectorMemoryAdapter(
                spec_dir=Path("/tmp/test-spec"),
                project_dir=Path("/tmp/test-project"),
            )
            assert adapter.is_enabled is False

    def test_adapter_creation_enabled(self):
        """Test adapter creation when enabled."""
        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "true"}, clear=False):
            from integrations.vector_memory.adapter import VectorMemoryAdapter

            adapter = VectorMemoryAdapter(
                spec_dir=Path("/tmp/test-spec"),
                project_dir=Path("/tmp/test-project"),
            )
            assert adapter.is_enabled is True
            assert adapter.is_initialized is False  # Not initialized yet

    def test_adapter_status_summary(self):
        """Test adapter status summary."""
        with patch.dict(
            os.environ,
            {
                "VECTOR_MEMORY_ENABLED": "true",
                "VECTOR_MEMORY_DB_PATH": ".test-memory/memories.db",
            },
            clear=False,
        ):
            from integrations.vector_memory.adapter import VectorMemoryAdapter

            adapter = VectorMemoryAdapter(
                spec_dir=Path("/tmp/test-spec"),
                project_dir=Path("/tmp/test-project"),
            )
            status = adapter.get_status_summary()
            assert status["enabled"] is True
            assert status["mode"] == "npx"
            assert ".test-memory/memories.db" in status["db_path"]
            assert "project_test-project" in status["namespace"]

    @pytest.mark.asyncio
    async def test_adapter_initialize_disabled(self):
        """Test adapter initialization when disabled."""
        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "false"}, clear=False):
            from integrations.vector_memory.adapter import VectorMemoryAdapter

            adapter = VectorMemoryAdapter(
                spec_dir=Path("/tmp/test-spec"),
                project_dir=Path("/tmp/test-project"),
            )
            result = await adapter.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_adapter_initialize_npx_mode(self):
        """Test adapter initialization in npx mode."""
        with patch.dict(
            os.environ,
            {"VECTOR_MEMORY_ENABLED": "true", "VECTOR_MEMORY_URL": ""},
            clear=False,
        ):
            from integrations.vector_memory.adapter import VectorMemoryAdapter

            adapter = VectorMemoryAdapter(
                spec_dir=Path("/tmp/test-spec"),
                project_dir=Path("/tmp/test-project"),
            )
            result = await adapter.initialize()
            # npx mode doesn't need HTTP session, so should succeed
            assert result is True
            assert adapter.is_initialized is True
            await adapter.close()


class TestMCPServerConfiguration:
    """Test MCP server configuration in client.py."""

    def test_is_vector_memory_enabled_helper(self):
        """Test the helper function in client.py."""
        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "true"}, clear=False):
            from core.client import is_vector_memory_enabled

            assert is_vector_memory_enabled() is True

        with patch.dict(os.environ, {"VECTOR_MEMORY_ENABLED": "false"}, clear=False):
            from core.client import is_vector_memory_enabled

            assert is_vector_memory_enabled() is False

    def test_is_vector_memory_docker_mode(self):
        """Test Docker mode detection."""
        with patch.dict(
            os.environ, {"VECTOR_MEMORY_URL": "http://localhost:3000"}, clear=False
        ):
            from core.client import is_vector_memory_docker_mode

            assert is_vector_memory_docker_mode() is True

        with patch.dict(os.environ, {"VECTOR_MEMORY_URL": ""}, clear=False):
            from core.client import is_vector_memory_docker_mode

            assert is_vector_memory_docker_mode() is False

    def test_get_vector_memory_url(self):
        """Test getting the vector memory URL."""
        with patch.dict(
            os.environ, {"VECTOR_MEMORY_URL": "http://localhost:3000"}, clear=False
        ):
            from core.client import get_vector_memory_url

            assert get_vector_memory_url() == "http://localhost:3000"


class TestVectorMemoryTools:
    """Test vector memory tool definitions."""

    def test_vector_memory_tools_defined(self):
        """Test that VECTOR_MEMORY_TOOLS constant is defined."""
        from agents.tools_pkg.models import VECTOR_MEMORY_TOOLS

        assert isinstance(VECTOR_MEMORY_TOOLS, (list, tuple, set))
        assert len(VECTOR_MEMORY_TOOLS) >= 5  # store, retrieve, search, list, delete

    def test_vector_memory_tools_exported(self):
        """Test that VECTOR_MEMORY_TOOLS is exported from package."""
        from agents.tools_pkg import VECTOR_MEMORY_TOOLS

        assert VECTOR_MEMORY_TOOLS is not None

    def test_tool_names(self):
        """Test expected tool names are present."""
        from agents.tools_pkg.models import VECTOR_MEMORY_TOOLS

        expected_tools = ["store", "retrieve", "search", "list", "delete"]
        for tool in expected_tools:
            assert any(tool in t for t in VECTOR_MEMORY_TOOLS), f"Missing tool: {tool}"


class TestBackwardCompatibility:
    """Test backward compatibility with existing Graphiti integration."""

    def test_graphiti_still_default(self):
        """Test that Graphiti is still the default backend."""
        # Clear MEMORY_BACKEND to test default
        env = {k: v for k, v in os.environ.items() if k != "MEMORY_BACKEND"}
        with patch.dict(os.environ, env, clear=True):
            from memory_config import get_memory_backend

            assert get_memory_backend() == "graphiti"

    def test_graphiti_imports_unchanged(self):
        """Test that Graphiti imports still work."""
        from graphiti_config import is_graphiti_enabled, get_graphiti_status
        from memory.graphiti_helpers import get_graphiti_memory

        # Functions should exist
        assert callable(is_graphiti_enabled)
        assert callable(get_graphiti_status)
        assert callable(get_graphiti_memory)

    def test_memory_manager_imports(self):
        """Test that memory_manager imports work correctly."""
        from agents.memory_manager import (
            save_session_memory,
            get_graphiti_context,
            debug_memory_system_status,
        )

        # Functions should exist
        assert callable(save_session_memory)
        assert callable(get_graphiti_context)
        assert callable(debug_memory_system_status)

    def test_backward_compat_alias(self):
        """Test that backward compatibility alias exists."""
        from agents.memory_manager import save_session_to_graphiti

        assert callable(save_session_to_graphiti)


class TestEnvExample:
    """Test that .env.example is properly documented."""

    def test_env_example_has_vector_memory_section(self):
        """Test that .env.example documents vector-memory configuration."""
        env_example_path = Path(__file__).parent.parent / "apps" / "backend" / ".env.example"
        if env_example_path.exists():
            content = env_example_path.read_text()
            assert "VECTOR_MEMORY_ENABLED" in content
            assert "MEMORY_BACKEND" in content
            assert "VECTOR_MEMORY_DB_PATH" in content
            assert "VECTOR_MEMORY_MODEL" in content
            assert "VECTOR_MEMORY_URL" in content


class TestE2EVerificationSteps:
    """
    Document E2E verification steps that require manual testing with
    a running vector-memory-mcp server.

    These tests document what manual verification should check.
    """

    @pytest.mark.skip(reason="Manual verification - requires running MCP server")
    def test_e2e_step1_enable_vector_memory(self):
        """
        Step 1: Enable vector-memory backend

        Set environment variables:
            VECTOR_MEMORY_ENABLED=true
            MEMORY_BACKEND=vector_memory

        Verify:
            - is_vector_memory_enabled() returns True
            - get_memory_backend() returns "vector_memory"
        """
        pass

    @pytest.mark.skip(reason="Manual verification - requires running MCP server")
    def test_e2e_step2_start_mcp_server(self):
        """
        Step 2: Start vector-memory-mcp server

        Option A - npx (recommended):
            npx -y @weslien/vector-memory-mcp --working-dir /path/to/project

        Option B - Docker:
            docker run -p 3000:3000 -v /path/to/data:/data ghcr.io/weslien/vector-memory-mcp

        Verify:
            - Server starts without errors
            - Server responds to requests
        """
        pass

    @pytest.mark.skip(reason="Manual verification - requires running MCP server")
    def test_e2e_step3_run_spec_build(self):
        """
        Step 3: Run a spec build

        Command:
            cd apps/backend
            VECTOR_MEMORY_ENABLED=true MEMORY_BACKEND=vector_memory DEBUG=true python run.py --spec 001

        Verify:
            - "MCP servers: vector-memory (local embeddings)" in output
            - No MCP connection errors
            - Build completes successfully
        """
        pass

    @pytest.mark.skip(reason="Manual verification - requires running MCP server")
    def test_e2e_step4_verify_memory_operations(self):
        """
        Step 4: Verify memory operations use vector-memory backend

        Check DEBUG output for:
            - "Attempting SECONDARY storage: vector-memory"
            - "Session X saved to vector-memory (SECONDARY)"
            - OR if primary: "Attempting PRIMARY storage: vector-memory"

        Verify:
            - Memory save operations succeed
            - Context retrieval works
        """
        pass

    @pytest.mark.skip(reason="Manual verification - requires running MCP server")
    def test_e2e_step5_check_database(self):
        """
        Step 5: Check .vector-memory/memories.db exists with data

        Commands:
            ls -la .vector-memory/
            sqlite3 .vector-memory/memories.db "SELECT COUNT(*) FROM memories"

        Verify:
            - memories.db file exists
            - Database has data (count > 0)
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
