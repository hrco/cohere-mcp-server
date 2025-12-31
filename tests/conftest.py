"""
Pytest configuration and fixtures for Cohere MCP Server tests.
"""

import os
import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict

# Set test environment variable
os.environ["COHERE_API_KEY"] = "test-api-key-12345"


@pytest.fixture
def mock_api_key():
    """Provide a test API key."""
    return "test-api-key-12345"


@pytest.fixture
def mock_chat_response():
    """Mock Cohere chat API response."""
    mock_response = Mock()
    mock_response.message = Mock()
    mock_response.message.content = [Mock(text="This is a test response")]
    mock_response.finish_reason = "COMPLETE"
    mock_response.usage = Mock()
    mock_response.usage.billed_units = Mock()
    mock_response.usage.billed_units.input_tokens = 10
    mock_response.usage.billed_units.output_tokens = 5
    return mock_response


@pytest.fixture
def mock_embed_response():
    """Mock Cohere embed API response."""
    mock_response = Mock()
    mock_response.embeddings = Mock()
    mock_response.embeddings.float_ = [[0.1, 0.2, 0.3, 0.4] * 256]  # 1024 dimensions
    return mock_response


@pytest.fixture
def mock_rerank_response():
    """Mock Cohere rerank API response."""
    mock_result_1 = Mock()
    mock_result_1.index = 0
    mock_result_1.relevance_score = 0.95
    mock_result_1.document = Mock()
    mock_result_1.document.text = "Most relevant document"

    mock_result_2 = Mock()
    mock_result_2.index = 2
    mock_result_2.relevance_score = 0.75
    mock_result_2.document = Mock()
    mock_result_2.document.text = "Second most relevant"

    mock_response = Mock()
    mock_response.results = [mock_result_1, mock_result_2]
    return mock_response


@pytest.fixture
def mock_classify_response():
    """Mock Cohere classify API response."""
    mock_classification = Mock()
    mock_classification.input = "test input"
    mock_classification.prediction = "positive"
    mock_classification.confidence = 0.87
    mock_classification.labels = []

    mock_response = Mock()
    mock_response.classifications = [mock_classification]
    return mock_response


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a test document.",
        "Another test document here.",
        "Third document for testing."
    ]


@pytest.fixture
def sample_documents():
    """Sample documents for reranking tests."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks with multiple layers.",
        "JavaScript is used for web development."
    ]


@pytest.fixture
def sample_classification_examples():
    """Sample classification training examples."""
    return [
        {"text": "I love this product!", "label": "positive"},
        {"text": "This is terrible.", "label": "negative"},
        {"text": "Great experience overall.", "label": "positive"},
        {"text": "Very disappointed.", "label": "negative"}
    ]


@pytest.fixture
async def mock_cohere_client(
    mock_chat_response,
    mock_embed_response,
    mock_rerank_response,
    mock_classify_response
):
    """Mock CohereClient for testing."""
    from cohere_mcp.client import CohereClient

    client = Mock(spec=CohereClient)

    # Mock async methods
    client.chat = AsyncMock(return_value=Mock(
        text="Test response",
        model="command-a-03-2025",
        finish_reason="COMPLETE",
        tokens_input=10,
        tokens_output=5
    ))

    client.embed = AsyncMock(return_value=Mock(
        embeddings=[[0.1] * 1024],
        model="embed-english-v3.0",
        input_type="search_document",
        texts=["test"]
    ))

    client.rerank = AsyncMock(return_value=Mock(
        results=[
            Mock(index=0, relevance_score=0.95, document="doc1"),
            Mock(index=1, relevance_score=0.75, document="doc2")
        ],
        model="rerank-english-v3.0",
        query="test query"
    ))

    client.aya_chat = AsyncMock(return_value=Mock(
        text="Test multilingual response",
        model="aya-expanse-32b",
        finish_reason="COMPLETE",
        tokens_input=10,
        tokens_output=5
    ))

    client.classify = AsyncMock(return_value={
        "classifications": [{
            "input": "test",
            "prediction": "positive",
            "confidence": 0.87,
            "labels": {}
        }],
        "model": "embed-english-v3.0"
    })

    client.summarize = AsyncMock(return_value={
        "summary": "This is a summary",
        "model": "command-r",
        "length": "medium",
        "format": "paragraph",
        "tokens_used": {"input": 100, "output": 20}
    })

    return client
