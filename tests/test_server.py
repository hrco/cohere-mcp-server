"""
Tests for MCP server tools and resources.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

from cohere_mcp import server
from cohere_mcp.server import (
    cohere_chat,
    cohere_chat_stream,
    cohere_embed,
    cohere_rerank,
    cohere_aya_chat,
    cohere_summarize,
    cohere_classify,
    get_models,
    get_config_info,
    create_server,
)


class TestCohereChat:
    """Test cohere_chat tool."""

    @pytest.mark.asyncio
    async def test_chat_success(self, mock_cohere_client):
        """Test successful chat completion."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_chat(message="Hello, world!")

            assert "text" in response
            assert response["text"] == "Test response"
            assert response["model"] == "command-a-03-2025"
            assert "tokens" in response

    @pytest.mark.asyncio
    async def test_chat_with_parameters(self, mock_cohere_client):
        """Test chat with custom parameters."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_chat(
                message="Test",
                model="command-r",
                temperature=0.5,
                max_tokens=2000,
                system_prompt="You are a coding assistant."
            )

            assert "text" in response
            # Verify the client was called with correct parameters
            mock_cohere_client.chat.assert_called_once()
            call_args = mock_cohere_client.chat.call_args
            assert call_args[1]["model"] == "command-r"
            assert call_args[1]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_chat_no_client(self):
        """Test chat when client is not initialized."""
        with patch.object(server, 'client', None):
            with pytest.raises(ValueError, match="not initialized"):
                await cohere_chat(message="test")

    @pytest.mark.asyncio
    async def test_chat_error_handling(self, mock_cohere_client):
        """Test chat error handling."""
        from cohere_mcp.client import CohereAPIError

        mock_cohere_client.chat.side_effect = CohereAPIError("API Error")

        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_chat(message="test")

            assert "error" in response
            assert "API Error" in response["error"]


class TestCohereChatStream:
    """Test cohere_chat_stream tool."""

    @pytest.mark.asyncio
    async def test_chat_stream_success(self, mock_cohere_client):
        """Test successful streaming chat."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_chat_stream(message="Hello!")

            assert isinstance(response, str)
            assert len(response) > 0


class TestCohereEmbed:
    """Test cohere_embed tool."""

    @pytest.mark.asyncio
    async def test_embed_success(self, mock_cohere_client, sample_texts):
        """Test successful embedding generation."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_embed(texts=sample_texts)

            assert "embeddings" in response
            assert "model" in response
            assert "dimensions" in response
            assert response["model"] == "embed-english-v3.0"
            assert response["num_texts"] == 1

    @pytest.mark.asyncio
    async def test_embed_with_custom_params(self, mock_cohere_client):
        """Test embed with custom parameters."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_embed(
                texts=["test"],
                model="embed-multilingual-v3.0",
                input_type="search_query"
            )

            # Verify client was called with correct params
            mock_cohere_client.embed.assert_called_once()
            call_args = mock_cohere_client.embed.call_args
            assert call_args[1]["model"] == "embed-multilingual-v3.0"
            assert call_args[1]["input_type"] == "search_query"


class TestCohereRerank:
    """Test cohere_rerank tool."""

    @pytest.mark.asyncio
    async def test_rerank_success(self, mock_cohere_client, sample_documents):
        """Test successful reranking."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_rerank(
                query="machine learning",
                documents=sample_documents
            )

            assert "results" in response
            assert "model" in response
            assert "query" in response
            assert len(response["results"]) > 0
            assert "relevance_score" in response["results"][0]

    @pytest.mark.asyncio
    async def test_rerank_with_top_n(self, mock_cohere_client):
        """Test rerank with top_n parameter."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_rerank(
                query="test",
                documents=["doc1", "doc2", "doc3"],
                top_n=2
            )

            assert "results" in response


class TestCohereAyaChat:
    """Test cohere_aya_chat tool."""

    @pytest.mark.asyncio
    async def test_aya_chat_success(self, mock_cohere_client):
        """Test successful Aya chat."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_aya_chat(message="Bonjour")

            assert "text" in response
            assert response["text"] == "Test multilingual response"

    @pytest.mark.asyncio
    async def test_aya_chat_with_language(self, mock_cohere_client):
        """Test Aya chat with language specification."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_aya_chat(
                message="Hello",
                language="Spanish"
            )

            assert "text" in response
            assert response["language"] == "Spanish"


class TestCohereSummarize:
    """Test cohere_summarize tool."""

    @pytest.mark.asyncio
    async def test_summarize_success(self, mock_cohere_client):
        """Test successful summarization."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_summarize(
                text="Long text to summarize..."
            )

            assert "summary" in response
            assert response["summary"] == "This is a summary"

    @pytest.mark.asyncio
    async def test_summarize_with_params(self, mock_cohere_client):
        """Test summarize with custom parameters."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_summarize(
                text="Text to summarize",
                length="short",
                format="bullets"
            )

            # Verify params were passed
            mock_cohere_client.summarize.assert_called_once()
            call_args = mock_cohere_client.summarize.call_args
            assert call_args[1]["length"] == "short"
            assert call_args[1]["format"] == "bullets"


class TestCohereClassify:
    """Test cohere_classify tool."""

    @pytest.mark.asyncio
    async def test_classify_success(self, mock_cohere_client, sample_classification_examples):
        """Test successful classification."""
        with patch.object(server, 'client', mock_cohere_client):
            response = await cohere_classify(
                inputs=["Great product!"],
                examples=sample_classification_examples
            )

            assert "classifications" in response
            assert len(response["classifications"]) > 0


class TestResources:
    """Test MCP resources."""

    @pytest.mark.asyncio
    async def test_get_models(self):
        """Test get_models resource."""
        response = await get_models()

        # Parse JSON response
        models_info = json.loads(response)

        assert "chat_models" in models_info
        assert "embed_models" in models_info
        assert "rerank_models" in models_info
        assert "aya_models" in models_info

    @pytest.mark.asyncio
    async def test_get_config_info(self):
        """Test get_config_info resource."""
        response = await get_config_info()

        # Parse JSON response
        config_info = json.loads(response)

        assert "default_models" in config_info
        assert "api_settings" in config_info
        assert "defaults" in config_info
        assert "chat" in config_info["default_models"]


class TestServerFunctions:
    """Test server creation and running functions."""

    def test_create_server(self):
        """Test create_server function."""
        from mcp.server.fastmcp import FastMCP

        mcp_server = create_server()

        assert mcp_server is not None
        assert isinstance(mcp_server, FastMCP)

    def test_run_server_no_client(self, capsys):
        """Test run_server with no client initialized."""
        with patch.object(server, 'client', None):
            # Run server should log error and return
            server.run_server()

            # Since server returns early, we should see logs but not start
            # This is a basic test - actual server start would block


class TestToolDocstrings:
    """Test that all tools have proper documentation."""

    def test_all_tools_have_docstrings(self):
        """Verify all MCP tools have docstrings."""
        tools = [
            cohere_chat,
            cohere_chat_stream,
            cohere_embed,
            cohere_rerank,
            cohere_aya_chat,
            cohere_summarize,
            cohere_classify,
        ]

        for tool in tools:
            assert tool.__doc__ is not None
            assert len(tool.__doc__.strip()) > 0

    def test_resources_have_docstrings(self):
        """Verify all resources have docstrings."""
        resources = [get_models, get_config_info]

        for resource in resources:
            assert resource.__doc__ is not None
            assert len(resource.__doc__.strip()) > 0
