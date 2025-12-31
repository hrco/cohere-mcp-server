"""
Cohere API Client wrapper for MCP Server.

Provides a unified interface for all Cohere API operations with
proper error handling, retries, and response formatting.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import cohere
from cohere.core.api_error import ApiError

from .config import CohereConfig, EmbedInputType

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    """Structured response from chat completion."""
    text: str
    model: str
    finish_reason: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None


@dataclass
class EmbedResponse:
    """Structured response from embedding generation."""
    embeddings: List[List[float]]
    model: str
    input_type: str
    texts: List[str]


@dataclass
class RerankResult:
    """Single reranking result."""
    index: int
    relevance_score: float
    document: str


@dataclass
class RerankResponse:
    """Structured response from reranking."""
    results: List[RerankResult]
    model: str
    query: str


class CohereClient:
    """
    Async wrapper for Cohere API operations.

    Provides methods for:
    - Chat/Completion
    - Embeddings
    - Reranking
    - Aya multilingual operations
    """

    def __init__(self, config: Optional[CohereConfig] = None):
        """Initialize the Cohere client with configuration."""
        self.config = config or CohereConfig.from_env()
        self._client = cohere.ClientV2(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        logger.info("Cohere client initialized successfully")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        preamble: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a chat completion request to Cohere.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (defaults to config default)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            preamble: Optional preamble/context

        Returns:
            ChatResponse with the generated text and metadata
        """
        model = model or self.config.default_chat_model
        temperature = temperature if temperature is not None else self.config.default_temperature
        max_tokens = max_tokens or self.config.default_max_tokens

        try:
            # Build message list
            formatted_messages = []

            # Add system message if provided
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add conversation messages
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

            # Run synchronous client in executor for async compatibility
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.chat(
                    model=model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

            # Extract response text
            text = ""
            if hasattr(response, 'message') and response.message:
                if hasattr(response.message, 'content') and response.message.content:
                    # Handle list of content items
                    if isinstance(response.message.content, list):
                        text = "".join(
                            item.text for item in response.message.content
                            if hasattr(item, 'text')
                        )
                    else:
                        text = str(response.message.content)

            # Extract token usage
            tokens_input = None
            tokens_output = None
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'billed_units'):
                    tokens_input = getattr(response.usage.billed_units, 'input_tokens', None)
                    tokens_output = getattr(response.usage.billed_units, 'output_tokens', None)

            return ChatResponse(
                text=text,
                model=model,
                finish_reason=getattr(response, 'finish_reason', None),
                tokens_input=tokens_input,
                tokens_output=tokens_output
            )

        except ApiError as e:
            logger.error(f"Cohere API error in chat: {e}")
            raise CohereAPIError(f"Chat API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}")
            raise CohereAPIError(f"Chat error: {str(e)}") from e

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: Optional[str] = None,
        truncate: str = "END",
    ) -> EmbedResponse:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            input_type: Type of input (search_document, search_query, etc.)
            truncate: How to handle long texts (END, START, NONE)

        Returns:
            EmbedResponse with embeddings and metadata
        """
        model = model or self.config.default_embed_model
        input_type = input_type or self.config.default_embed_input_type

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.embed(
                    texts=texts,
                    model=model,
                    input_type=input_type,
                    truncate=truncate,
                    embedding_types=["float"]
                )
            )

            # Extract embeddings
            embeddings = []
            if hasattr(response, 'embeddings'):
                if hasattr(response.embeddings, 'float_'):
                    embeddings = response.embeddings.float_
                elif hasattr(response.embeddings, 'float'):
                    embeddings = response.embeddings.float
                elif isinstance(response.embeddings, list):
                    embeddings = response.embeddings

            return EmbedResponse(
                embeddings=embeddings,
                model=model,
                input_type=input_type,
                texts=texts
            )

        except ApiError as e:
            logger.error(f"Cohere API error in embed: {e}")
            raise CohereAPIError(f"Embed API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in embed: {e}")
            raise CohereAPIError(f"Embed error: {str(e)}") from e

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> RerankResponse:
        """
        Rerank documents based on relevance to a query.

        Args:
            query: The search query
            documents: List of documents to rerank
            model: Reranking model to use
            top_n: Number of top results to return

        Returns:
            RerankResponse with ranked results
        """
        model = model or self.config.default_rerank_model
        top_n = top_n or min(self.config.default_top_n, len(documents))

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.rerank(
                    query=query,
                    documents=documents,
                    model=model,
                    top_n=top_n,
                )
            )

            results = []
            if hasattr(response, 'results'):
                for result in response.results:
                    results.append(RerankResult(
                        index=result.index,
                        relevance_score=result.relevance_score,
                        document=documents[result.index]
                    ))

            return RerankResponse(
                results=results,
                model=model,
                query=query
            )

        except ApiError as e:
            logger.error(f"Cohere API error in rerank: {e}")
            raise CohereAPIError(f"Rerank API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in rerank: {e}")
            raise CohereAPIError(f"Rerank error: {str(e)}") from e

    async def aya_chat(
        self,
        message: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Chat using Aya multilingual models.

        Args:
            message: The user message
            model: Aya model to use (aya-expanse-8b or aya-expanse-32b)
            language: Target language for response (optional hint)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ChatResponse with the generated text
        """
        model = model or self.config.default_aya_model

        # Build system prompt for language preference if specified
        system_prompt = None
        if language:
            system_prompt = f"Please respond in {language}."

        messages = [{"role": "user", "content": message}]

        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )

    async def classify(
        self,
        inputs: List[str],
        examples: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify texts into categories based on examples.

        Args:
            inputs: List of texts to classify
            examples: List of dicts with 'text' and 'label' keys
            model: Model to use for classification

        Returns:
            Classification results with predictions and confidence
        """
        model = model or self.config.default_chat_model

        try:
            # Use embed model for classification
            embed_model = self.config.default_embed_model

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.classify(
                    inputs=inputs,
                    examples=examples,
                    model=embed_model,
                )
            )

            classifications = []
            if hasattr(response, 'classifications'):
                for cls in response.classifications:
                    classifications.append({
                        "input": cls.input,
                        "prediction": cls.prediction,
                        "confidence": cls.confidence,
                        "labels": {
                            label.name: label.confidence
                            for label in (cls.labels or [])
                        } if hasattr(cls, 'labels') else {}
                    })

            return {
                "classifications": classifications,
                "model": embed_model
            }

        except ApiError as e:
            logger.error(f"Cohere API error in classify: {e}")
            raise CohereAPIError(f"Classify API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in classify: {e}")
            raise CohereAPIError(f"Classify error: {str(e)}") from e

    async def summarize(
        self,
        text: str,
        model: Optional[str] = None,
        length: str = "medium",
        format: str = "paragraph",
        extractiveness: str = "medium",
    ) -> Dict[str, Any]:
        """
        Summarize a piece of text.

        Args:
            text: Text to summarize
            model: Model to use
            length: Summary length (short, medium, long)
            format: Output format (paragraph, bullets)
            extractiveness: How extractive vs. abstractive (low, medium, high)

        Returns:
            Summary result with text and metadata
        """
        model = model or self.config.default_chat_model

        # Use chat endpoint with summarization prompt
        system_prompt = f"""You are a summarization assistant. Create a {length} {format} summary.
Extractiveness level: {extractiveness} (low=more abstractive, high=more extractive)."""

        messages = [{
            "role": "user",
            "content": f"Please summarize the following text:\n\n{text}"
        }]

        response = await self.chat(
            messages=messages,
            model=model,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more focused summaries
        )

        return {
            "summary": response.text,
            "model": model,
            "length": length,
            "format": format,
            "tokens_used": {
                "input": response.tokens_input,
                "output": response.tokens_output
            }
        }


class CohereAPIError(Exception):
    """Custom exception for Cohere API errors."""
    pass
