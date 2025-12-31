"""
Tests for configuration module.
"""

import os
import pytest
from cohere_mcp.config import (
    CohereConfig,
    CohereModel,
    EmbedInputType,
    get_config,
    MODEL_INFO
)


class TestCohereModel:
    """Test CohereModel enum."""

    def test_command_models(self):
        """Test Command model values."""
        assert CohereModel.COMMAND_A_03_2025.value == "command-a-03-2025"
        assert CohereModel.COMMAND_R_PLUS.value == "command-r-plus"
        assert CohereModel.COMMAND_R.value == "command-r"

    def test_embed_models(self):
        """Test Embed model values."""
        assert CohereModel.EMBED_ENGLISH_V3.value == "embed-english-v3.0"
        assert CohereModel.EMBED_MULTILINGUAL_V3.value == "embed-multilingual-v3.0"

    def test_rerank_models(self):
        """Test Rerank model values."""
        assert CohereModel.RERANK_ENGLISH_V3.value == "rerank-english-v3.0"
        assert CohereModel.RERANK_MULTILINGUAL_V3.value == "rerank-multilingual-v3.0"

    def test_aya_models(self):
        """Test Aya model values."""
        assert CohereModel.AYA_EXPANSE_8B.value == "aya-expanse-8b"
        assert CohereModel.AYA_EXPANSE_32B.value == "aya-expanse-32b"


class TestEmbedInputType:
    """Test EmbedInputType enum."""

    def test_input_types(self):
        """Test all embed input type values."""
        assert EmbedInputType.SEARCH_DOCUMENT.value == "search_document"
        assert EmbedInputType.SEARCH_QUERY.value == "search_query"
        assert EmbedInputType.CLASSIFICATION.value == "classification"
        assert EmbedInputType.CLUSTERING.value == "clustering"


class TestCohereConfig:
    """Test CohereConfig dataclass."""

    def test_default_config(self, mock_api_key):
        """Test config with default values."""
        config = CohereConfig(api_key=mock_api_key)

        assert config.api_key == mock_api_key
        assert config.default_chat_model == CohereModel.COMMAND_A_03_2025.value
        assert config.default_embed_model == CohereModel.EMBED_ENGLISH_V3.value
        assert config.default_rerank_model == CohereModel.RERANK_ENGLISH_V3.value
        assert config.default_aya_model == CohereModel.AYA_EXPANSE_32B.value
        assert config.timeout == 60
        assert config.max_retries == 3

    def test_custom_config(self, mock_api_key):
        """Test config with custom values."""
        config = CohereConfig(
            api_key=mock_api_key,
            default_chat_model="command-r",
            timeout=120,
            max_retries=5
        )

        assert config.api_key == mock_api_key
        assert config.default_chat_model == "command-r"
        assert config.timeout == 120
        assert config.max_retries == 5

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            CohereConfig(api_key="")

    def test_from_env(self, mock_api_key):
        """Test config creation from environment variables."""
        os.environ["COHERE_API_KEY"] = mock_api_key

        config = CohereConfig.from_env()

        assert config.api_key == mock_api_key
        assert isinstance(config, CohereConfig)

    def test_from_env_with_custom_values(self, mock_api_key):
        """Test config from env with custom environment variables."""
        os.environ["COHERE_API_KEY"] = mock_api_key
        os.environ["COHERE_DEFAULT_CHAT_MODEL"] = "command-r-plus"
        os.environ["COHERE_TIMEOUT"] = "90"
        os.environ["COHERE_MAX_RETRIES"] = "5"

        config = CohereConfig.from_env()

        assert config.api_key == mock_api_key
        assert config.default_chat_model == "command-r-plus"
        assert config.timeout == 90
        assert config.max_retries == 5

        # Cleanup
        del os.environ["COHERE_DEFAULT_CHAT_MODEL"]
        del os.environ["COHERE_TIMEOUT"]
        del os.environ["COHERE_MAX_RETRIES"]

    def test_get_config(self, mock_api_key):
        """Test get_config() function."""
        os.environ["COHERE_API_KEY"] = mock_api_key

        config = get_config()

        assert isinstance(config, CohereConfig)
        assert config.api_key == mock_api_key


class TestModelInfo:
    """Test MODEL_INFO dictionary."""

    def test_model_info_structure(self):
        """Test that MODEL_INFO has expected structure."""
        assert "chat_models" in MODEL_INFO
        assert "embed_models" in MODEL_INFO
        assert "rerank_models" in MODEL_INFO
        assert "aya_models" in MODEL_INFO

    def test_chat_models_info(self):
        """Test chat models information."""
        chat_models = MODEL_INFO["chat_models"]

        assert CohereModel.COMMAND_A_03_2025.value in chat_models
        command_a_info = chat_models[CohereModel.COMMAND_A_03_2025.value]

        assert "description" in command_a_info
        assert "context_length" in command_a_info
        assert "recommended_for" in command_a_info
        assert command_a_info["context_length"] == 128000

    def test_embed_models_info(self):
        """Test embed models information."""
        embed_models = MODEL_INFO["embed_models"]

        assert CohereModel.EMBED_ENGLISH_V3.value in embed_models
        embed_info = embed_models[CohereModel.EMBED_ENGLISH_V3.value]

        assert "description" in embed_info
        assert "dimensions" in embed_info
        assert "max_tokens" in embed_info
        assert embed_info["dimensions"] == 1024

    def test_rerank_models_info(self):
        """Test rerank models information."""
        rerank_models = MODEL_INFO["rerank_models"]

        assert CohereModel.RERANK_ENGLISH_V3.value in rerank_models
        rerank_info = rerank_models[CohereModel.RERANK_ENGLISH_V3.value]

        assert "description" in rerank_info
        assert "max_documents" in rerank_info
        assert rerank_info["max_documents"] == 1000

    def test_aya_models_info(self):
        """Test Aya models information."""
        aya_models = MODEL_INFO["aya_models"]

        assert CohereModel.AYA_EXPANSE_32B.value in aya_models
        aya_info = aya_models[CohereModel.AYA_EXPANSE_32B.value]

        assert "description" in aya_info
        assert "languages" in aya_info
        assert "context_length" in aya_info
