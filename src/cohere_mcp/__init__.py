"""
Cohere MCP Server - Model Context Protocol integration for Cohere's AI platform.

This package provides MCP tools for:
- Chat/Completion with Command models
- Embeddings generation for semantic search
- Reranking for improved search results
- Multilingual capabilities with Aya models
"""

__version__ = "1.0.0"
__author__ = "Cohere MCP Project"

from .server import create_server, run_server

__all__ = ["create_server", "run_server", "__version__"]
