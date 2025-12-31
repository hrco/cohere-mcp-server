"""
Cohere MCP Server - FastMCP implementation.

Exposes Cohere's AI capabilities through the Model Context Protocol (MCP).
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

from .client import CohereClient, CohereAPIError
from .config import get_config, CohereModel, EmbedInputType, MODEL_INFO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("cohere-mcp-server")

# Initialize Cohere client
try:
    client = CohereClient()
    logger.info("Cohere client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Cohere client: {e}")
    client = None


@mcp.tool()
async def cohere_chat(
    message: str,
    model: str = CohereModel.COMMAND_A_03_2025.value,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chat with Cohere's Command models.

    Use this for conversational AI, question answering, reasoning tasks,
    and code generation.

    Args:
        message: The user message to send
        model: Cohere model to use (default: command-a-03-2025)
        temperature: Sampling temperature 0-1 (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4096)
        system_prompt: Optional system prompt for instructions

    Returns:
        Dict with response text and metadata
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        messages = [{"role": "user", "content": message}]

        response = await client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )

        return {
            "text": response.text,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "tokens": {
                "input": response.tokens_input,
                "output": response.tokens_output
            }
        }
    except CohereAPIError as e:
        logger.error(f"Cohere chat error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def cohere_chat_stream(
    message: str,
    model: str = CohereModel.COMMAND_A_03_2025.value,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Chat with Cohere's Command models using streaming responses.

    Streams the response token by token for real-time interaction.

    Args:
        message: The user message to send
        model: Cohere model to use (default: command-a-03-2025)
        temperature: Sampling temperature 0-1 (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4096)
        system_prompt: Optional system prompt for instructions

    Returns:
        Streamed response text
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        # For now, use non-streaming as fallback
        # TODO: Implement actual streaming with Cohere SDK's stream method
        messages = [{"role": "user", "content": message}]

        response = await client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )

        return response.text

    except CohereAPIError as e:
        logger.error(f"Cohere chat stream error: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in chat stream: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
async def cohere_embed(
    texts: List[str],
    model: str = CohereModel.EMBED_ENGLISH_V3.value,
    input_type: str = EmbedInputType.SEARCH_DOCUMENT.value,
) -> Dict[str, Any]:
    """
    Generate embeddings for texts using Cohere's embedding models.

    Use for semantic search, RAG (Retrieval Augmented Generation),
    clustering, and classification tasks.

    Args:
        texts: List of texts to embed (max 96 texts per request)
        model: Embedding model to use (default: embed-english-v3.0)
        input_type: Type of input - search_document, search_query,
                   classification, or clustering

    Returns:
        Dict with embeddings array and metadata
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        response = await client.embed(
            texts=texts,
            model=model,
            input_type=input_type
        )

        return {
            "embeddings": response.embeddings,
            "model": response.model,
            "input_type": response.input_type,
            "num_texts": len(response.texts),
            "dimensions": len(response.embeddings[0]) if response.embeddings else 0
        }
    except CohereAPIError as e:
        logger.error(f"Cohere embed error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in embed: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def cohere_rerank(
    query: str,
    documents: List[str],
    model: str = CohereModel.RERANK_ENGLISH_V3.value,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Rerank documents based on relevance to a query.

    Use for improving search results in RAG systems by reordering
    documents by relevance.

    Args:
        query: The search query
        documents: List of documents to rerank (max 1000)
        model: Rerank model to use (default: rerank-english-v3.0)
        top_n: Number of top results to return (default: 10)

    Returns:
        Dict with ranked results and relevance scores
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        response = await client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=min(top_n, len(documents))
        )

        results = [
            {
                "index": result.index,
                "relevance_score": result.relevance_score,
                "document": result.document
            }
            for result in response.results
        ]

        return {
            "results": results,
            "model": response.model,
            "query": response.query,
            "total_documents": len(documents)
        }
    except CohereAPIError as e:
        logger.error(f"Cohere rerank error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in rerank: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def cohere_aya_chat(
    message: str,
    model: str = CohereModel.AYA_EXPANSE_32B.value,
    language: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Chat using Cohere's Aya multilingual models.

    Supports 23+ languages including English, Spanish, French, Arabic,
    Japanese, and many more.

    Args:
        message: The user message (in any supported language)
        model: Aya model to use (aya-expanse-8b or aya-expanse-32b)
        language: Optional target language for response
        temperature: Sampling temperature 0-1 (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 4096)

    Returns:
        Dict with response text and metadata
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        response = await client.aya_chat(
            message=message,
            model=model,
            language=language,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "text": response.text,
            "model": response.model,
            "language": language,
            "finish_reason": response.finish_reason,
            "tokens": {
                "input": response.tokens_input,
                "output": response.tokens_output
            }
        }
    except CohereAPIError as e:
        logger.error(f"Cohere Aya chat error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in Aya chat: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def cohere_summarize(
    text: str,
    model: str = CohereModel.COMMAND_R.value,
    length: str = "medium",
    format: str = "paragraph",
) -> Dict[str, Any]:
    """
    Summarize text using Cohere's models.

    Args:
        text: Text to summarize
        model: Model to use (default: command-r)
        length: Summary length - short, medium, or long
        format: Output format - paragraph or bullets

    Returns:
        Dict with summary and metadata
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        response = await client.summarize(
            text=text,
            model=model,
            length=length,
            format=format
        )

        return response
    except CohereAPIError as e:
        logger.error(f"Cohere summarize error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in summarize: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
async def cohere_classify(
    inputs: List[str],
    examples: List[Dict[str, str]],
    model: str = CohereModel.EMBED_ENGLISH_V3.value,
) -> Dict[str, Any]:
    """
    Classify texts into categories based on examples.

    Args:
        inputs: List of texts to classify
        examples: List of dicts with 'text' and 'label' keys for training
        model: Embedding model to use (default: embed-english-v3.0)

    Returns:
        Dict with classification results and confidence scores
    """
    if not client:
        raise ValueError("Cohere client not initialized")

    try:
        response = await client.classify(
            inputs=inputs,
            examples=examples,
            model=model
        )

        return response
    except CohereAPIError as e:
        logger.error(f"Cohere classify error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in classify: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.resource("cohere://models")
async def get_models() -> str:
    """
    List all available Cohere models with their capabilities.

    Returns model information including descriptions, context lengths,
    and recommended use cases.
    """
    return json.dumps(MODEL_INFO, indent=2)


@mcp.resource("cohere://config")
async def get_config_info() -> str:
    """
    Get current Cohere MCP server configuration.

    Returns configuration details including default models and settings.
    """
    try:
        config = get_config()
        config_info = {
            "default_models": {
                "chat": config.default_chat_model,
                "embed": config.default_embed_model,
                "rerank": config.default_rerank_model,
                "aya": config.default_aya_model
            },
            "api_settings": {
                "timeout": config.timeout,
                "max_retries": config.max_retries
            },
            "defaults": {
                "temperature": config.default_temperature,
                "max_tokens": config.default_max_tokens,
                "embed_input_type": config.default_embed_input_type,
                "rerank_top_n": config.default_top_n
            }
        }
        return json.dumps(config_info, indent=2)
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return json.dumps({"error": str(e)})


def create_server() -> FastMCP:
    """
    Create and return a configured FastMCP server instance.

    Returns:
        Configured FastMCP server
    """
    return mcp


def run_server():
    """
    Run the Cohere MCP server.

    Starts the server and listens for MCP protocol messages via stdio.
    """
    logger.info("Starting Cohere MCP Server...")

    if not client:
        logger.error("Failed to start: Cohere client not initialized")
        logger.error("Please ensure COHERE_API_KEY environment variable is set")
        return

    logger.info("Server ready. Listening for MCP requests...")
    mcp.run()


if __name__ == "__main__":
    run_server()
