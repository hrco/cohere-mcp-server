"""
Configuration management for Cohere MCP Server.

Handles API key management, model configurations, and default settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class CohereModel(str, Enum):
    """Available Cohere models."""

    # Command models - Chat/Completion
    COMMAND_A_03_2025 = "command-a-03-2025"
    COMMAND_R_PLUS = "command-r-plus"
    COMMAND_R = "command-r"
    COMMAND_LIGHT = "command-light"
    COMMAND = "command"

    # Embed models
    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    # Rerank models
    RERANK_ENGLISH_V3 = "rerank-english-v3.0"
    RERANK_MULTILINGUAL_V3 = "rerank-multilingual-v3.0"
    RERANK_ENGLISH_V2 = "rerank-english-v2.0"
    RERANK_MULTILINGUAL_V2 = "rerank-multilingual-v2.0"

    # Aya models (Multilingual)
    AYA_EXPANSE_8B = "aya-expanse-8b"
    AYA_EXPANSE_32B = "aya-expanse-32b"


class EmbedInputType(str, Enum):
    """Input types for embedding generation."""
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


@dataclass
class CohereConfig:
    """Configuration settings for Cohere API integration."""

    api_key: str = field(default_factory=lambda: os.environ.get("COHERE_API_KEY", ""))

    # Default models
    default_chat_model: str = CohereModel.COMMAND_A_03_2025.value
    default_embed_model: str = CohereModel.EMBED_ENGLISH_V3.value
    default_rerank_model: str = CohereModel.RERANK_ENGLISH_V3.value
    default_aya_model: str = CohereModel.AYA_EXPANSE_32B.value

    # API settings
    api_base_url: str = "https://api.cohere.com/v2"
    timeout: int = 60
    max_retries: int = 3

    # Chat defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    # Embed defaults
    default_embed_input_type: str = EmbedInputType.SEARCH_DOCUMENT.value

    # Rerank defaults
    default_top_n: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError(
                "COHERE_API_KEY environment variable is not set. "
                "Please set it with your Cohere API key."
            )

    @classmethod
    def from_env(cls) -> "CohereConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.environ.get("COHERE_API_KEY", ""),
            default_chat_model=os.environ.get(
                "COHERE_DEFAULT_CHAT_MODEL",
                CohereModel.COMMAND_A_03_2025.value
            ),
            default_embed_model=os.environ.get(
                "COHERE_DEFAULT_EMBED_MODEL",
                CohereModel.EMBED_ENGLISH_V3.value
            ),
            default_rerank_model=os.environ.get(
                "COHERE_DEFAULT_RERANK_MODEL",
                CohereModel.RERANK_ENGLISH_V3.value
            ),
            timeout=int(os.environ.get("COHERE_TIMEOUT", "60")),
            max_retries=int(os.environ.get("COHERE_MAX_RETRIES", "3")),
        )


def get_config() -> CohereConfig:
    """Get the current configuration instance."""
    return CohereConfig.from_env()


# Model information for documentation
MODEL_INFO = {
    "chat_models": {
        CohereModel.COMMAND_A_03_2025.value: {
            "description": "Latest Command model - best for complex reasoning and instruction following",
            "context_length": 128000,
            "recommended_for": ["complex tasks", "reasoning", "code generation"]
        },
        CohereModel.COMMAND_R_PLUS.value: {
            "description": "Command R+ - excellent for RAG and tool use",
            "context_length": 128000,
            "recommended_for": ["RAG", "tool use", "agents"]
        },
        CohereModel.COMMAND_R.value: {
            "description": "Command R - balanced performance and cost",
            "context_length": 128000,
            "recommended_for": ["general chat", "summarization"]
        },
    },
    "embed_models": {
        CohereModel.EMBED_ENGLISH_V3.value: {
            "description": "Best English embedding model",
            "dimensions": 1024,
            "max_tokens": 512
        },
        CohereModel.EMBED_MULTILINGUAL_V3.value: {
            "description": "Best multilingual embedding model (100+ languages)",
            "dimensions": 1024,
            "max_tokens": 512
        },
    },
    "rerank_models": {
        CohereModel.RERANK_ENGLISH_V3.value: {
            "description": "Best English reranking model",
            "max_documents": 1000
        },
        CohereModel.RERANK_MULTILINGUAL_V3.value: {
            "description": "Best multilingual reranking model",
            "max_documents": 1000
        },
    },
    "aya_models": {
        CohereModel.AYA_EXPANSE_8B.value: {
            "description": "Aya Expanse 8B - efficient multilingual model",
            "languages": "23+ languages",
            "context_length": 8192
        },
        CohereModel.AYA_EXPANSE_32B.value: {
            "description": "Aya Expanse 32B - powerful multilingual model",
            "languages": "23+ languages",
            "context_length": 8192
        },
    }
}
