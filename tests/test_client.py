"""
Tests for Cohere API client wrapper.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from cohere.core.api_error import ApiError

from cohere_mcp.client import (
    CohereClient,
    CohereAPIError,
    ChatResponse,
    EmbedResponse,
    RerankResponse
)
from cohere_mcp.config import CohereConfig


class TestCohereClient:
    """Test CohereClient initialization."""

    def test_init_with_config(self, mock_api_key):
        """Test client initialization with config."""
        config = CohereConfig(api_key=mock_api_key)
        client = CohereClient(config=config)

        assert client.config == config
        assert client._client is not None

    def test_init_from_env(self, mock_api_key):
        """Test client initialization from environment."""
        client = CohereClient()

        assert client.config.api_key == mock_api_key
        assert client._client is not None


class TestChatMethod:
    """Test chat method."""

    @pytest.mark.asyncio
    async def test_chat_success(self, mock_api_key, mock_chat_response):
        """Test successful chat completion."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(return_value=mock_chat_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            messages = [{"role": "user", "content": "Hello"}]
            response = await client.chat(messages=messages)

            assert isinstance(response, ChatResponse)
            assert response.text == "This is a test response"
            assert response.model == "command-a-03-2025"
            assert response.tokens_input == 10
            assert response.tokens_output == 5

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, mock_api_key, mock_chat_response):
        """Test chat with system prompt."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(return_value=mock_chat_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            messages = [{"role": "user", "content": "Hello"}]
            response = await client.chat(
                messages=messages,
                system_prompt="You are a helpful assistant."
            )

            assert isinstance(response, ChatResponse)
            # Verify system prompt was added to messages
            call_args = mock_client.chat.call_args
            assert any(
                msg.get("role") == "system"
                for msg in call_args[1]["messages"]
            )

    @pytest.mark.asyncio
    async def test_chat_api_error(self, mock_api_key):
        """Test chat with API error."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(side_effect=ApiError(body={}, status_code=500))
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(CohereAPIError):
                await client.chat(messages=messages)


class TestEmbedMethod:
    """Test embed method."""

    @pytest.mark.asyncio
    async def test_embed_success(self, mock_api_key, mock_embed_response, sample_texts):
        """Test successful embedding generation."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.embed = Mock(return_value=mock_embed_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.embed(texts=sample_texts)

            assert isinstance(response, EmbedResponse)
            assert len(response.embeddings) > 0
            assert response.model == "embed-english-v3.0"
            assert response.input_type == "search_document"

    @pytest.mark.asyncio
    async def test_embed_custom_input_type(self, mock_api_key, mock_embed_response):
        """Test embed with custom input type."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.embed = Mock(return_value=mock_embed_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.embed(
                texts=["test"],
                input_type="search_query"
            )

            call_args = mock_client.embed.call_args
            assert call_args[1]["input_type"] == "search_query"


class TestRerankMethod:
    """Test rerank method."""

    @pytest.mark.asyncio
    async def test_rerank_success(self, mock_api_key, mock_rerank_response, sample_documents):
        """Test successful reranking."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.rerank = Mock(return_value=mock_rerank_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.rerank(
                query="machine learning",
                documents=sample_documents
            )

            assert isinstance(response, RerankResponse)
            assert len(response.results) == 2
            assert response.results[0].relevance_score > response.results[1].relevance_score
            assert response.model == "rerank-english-v3.0"

    @pytest.mark.asyncio
    async def test_rerank_with_top_n(self, mock_api_key, mock_rerank_response):
        """Test rerank with custom top_n."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.rerank = Mock(return_value=mock_rerank_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.rerank(
                query="test",
                documents=["doc1", "doc2", "doc3"],
                top_n=2
            )

            call_args = mock_client.rerank.call_args
            assert call_args[1]["top_n"] == 2


class TestAyaChatMethod:
    """Test aya_chat method."""

    @pytest.mark.asyncio
    async def test_aya_chat_success(self, mock_api_key, mock_chat_response):
        """Test successful Aya multilingual chat."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(return_value=mock_chat_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.aya_chat(
                message="Bonjour",
                language="French"
            )

            assert isinstance(response, ChatResponse)
            assert response.text == "This is a test response"

    @pytest.mark.asyncio
    async def test_aya_chat_with_language(self, mock_api_key, mock_chat_response):
        """Test Aya chat with language specification."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(return_value=mock_chat_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.aya_chat(
                message="Hello",
                language="Spanish"
            )

            # Verify system prompt includes language
            call_args = mock_client.chat.call_args
            system_prompt = call_args[1].get("system_prompt")
            assert system_prompt and "Spanish" in system_prompt


class TestClassifyMethod:
    """Test classify method."""

    @pytest.mark.asyncio
    async def test_classify_success(
        self,
        mock_api_key,
        mock_classify_response,
        sample_classification_examples
    ):
        """Test successful classification."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.classify = Mock(return_value=mock_classify_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.classify(
                inputs=["Great product!"],
                examples=sample_classification_examples
            )

            assert "classifications" in response
            assert len(response["classifications"]) > 0
            assert response["classifications"][0]["prediction"] == "positive"


class TestSummarizeMethod:
    """Test summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_success(self, mock_api_key, mock_chat_response):
        """Test successful summarization."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(return_value=mock_chat_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.summarize(
                text="Long text to summarize...",
                length="short"
            )

            assert "summary" in response
            assert "model" in response
            assert response["length"] == "short"

    @pytest.mark.asyncio
    async def test_summarize_custom_format(self, mock_api_key, mock_chat_response):
        """Test summarize with custom format."""
        with patch('cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client.chat = Mock(return_value=mock_chat_response)
            mock_client_class.return_value = mock_client

            config = CohereConfig(api_key=mock_api_key)
            client = CohereClient(config=config)

            response = await client.summarize(
                text="Text to summarize",
                format="bullets"
            )

            assert response["format"] == "bullets"
