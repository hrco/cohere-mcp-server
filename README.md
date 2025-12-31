# Cohere MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides seamless integration with [Cohere's AI platform](https://cohere.com/). This server exposes Cohere's powerful language models, embeddings, and reranking capabilities through the MCP protocol, enabling AI assistants like Claude to leverage Cohere's tools.

## Features

- **Chat & Completion** - Conversational AI with Command models (including command-a-03-2025)
- **Embeddings** - Generate semantic embeddings for search, RAG, and clustering
- **Reranking** - Improve search relevance for RAG systems
- **Multilingual Chat** - 23+ language support with Aya models
- **Text Summarization** - Condense long documents
- **Classification** - Few-shot text classification
- **Streaming Support** - Real-time response streaming for chat

## Available Models

### Command Models (Chat/Completion)
- `command-a-03-2025` - Latest Command model for complex reasoning
- `command-r-plus` - Excellent for RAG and tool use
- `command-r` - Balanced performance and cost
- `command-light` - Fast, lightweight model

### Embedding Models
- `embed-english-v3.0` - Best English embedding model (1024 dimensions)
- `embed-multilingual-v3.0` - 100+ language support
- `embed-english-light-v3.0` - Lightweight English embeddings
- `embed-multilingual-light-v3.0` - Lightweight multilingual

### Rerank Models
- `rerank-english-v3.0` - Best English reranking
- `rerank-multilingual-v3.0` - Multilingual reranking support

### Aya Models (Multilingual)
- `aya-expanse-32b` - Powerful multilingual model (23+ languages)
- `aya-expanse-8b` - Efficient multilingual model

## Installation

### Prerequisites

- Python 3.10 or higher
- A Cohere API key ([get one here](https://dashboard.cohere.com/api-keys))

### Install from Source

1. Clone or download this repository:
```bash
cd /home/diablo/Projects/COHERE/cohere-mcp-server
```

2. Install the package:
```bash
pip install -e .
```

3. Set up your API key:
```bash
# Create a .env file in the project root
echo "COHERE_API_KEY=your-api-key-here" > .env
```

Or set it as an environment variable:
```bash
export COHERE_API_KEY="your-api-key-here"
```

## Usage

### Running the Server

Run the MCP server directly:
```bash
cohere-mcp
```

Or using Python:
```bash
python -m cohere_mcp.server
```

The server communicates via stdio and follows the MCP protocol specification.

### Configuring with Claude Desktop

Add the following to your Claude Desktop configuration file:

**MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "cohere": {
      "command": "cohere-mcp",
      "env": {
        "COHERE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Or if using Python directly:
```json
{
  "mcpServers": {
    "cohere": {
      "command": "python",
      "args": ["-m", "cohere_mcp.server"],
      "env": {
        "COHERE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Using with Other MCP Clients

Any MCP-compatible client can connect to this server. The server uses stdio transport and follows the MCP specification.

## Available Tools

### cohere_chat

Chat with Cohere's Command models for conversational AI and reasoning tasks.

**Parameters**:
- `message` (string, required) - The user message
- `model` (string) - Model to use (default: "command-a-03-2025")
- `temperature` (number) - Sampling temperature 0-1 (default: 0.7)
- `max_tokens` (number) - Maximum tokens to generate (default: 4096)
- `system_prompt` (string) - Optional system instructions

**Example**:
```json
{
  "message": "Explain quantum computing in simple terms",
  "model": "command-a-03-2025",
  "temperature": 0.7
}
```

### cohere_chat_stream

Streaming version of chat for real-time responses.

**Parameters**: Same as `cohere_chat`

### cohere_embed

Generate embeddings for semantic search, RAG, and clustering.

**Parameters**:
- `texts` (array of strings, required) - Texts to embed (max 96 per request)
- `model` (string) - Embedding model (default: "embed-english-v3.0")
- `input_type` (string) - Type: "search_document", "search_query", "classification", or "clustering"

**Example**:
```json
{
  "texts": ["Document 1", "Document 2"],
  "model": "embed-english-v3.0",
  "input_type": "search_document"
}
```

### cohere_rerank

Rerank documents based on relevance to a query (ideal for RAG systems).

**Parameters**:
- `query` (string, required) - The search query
- `documents` (array of strings, required) - Documents to rerank (max 1000)
- `model` (string) - Rerank model (default: "rerank-english-v3.0")
- `top_n` (number) - Number of results to return (default: 10)

**Example**:
```json
{
  "query": "What is machine learning?",
  "documents": ["Doc about ML", "Doc about cooking", "Doc about AI"],
  "top_n": 5
}
```

### cohere_aya_chat

Chat using multilingual Aya models (supports 23+ languages).

**Parameters**:
- `message` (string, required) - User message in any supported language
- `model` (string) - Aya model (default: "aya-expanse-32b")
- `language` (string) - Target response language (optional)
- `temperature` (number) - Sampling temperature (default: 0.7)
- `max_tokens` (number) - Maximum tokens (default: 4096)

### cohere_summarize

Summarize text content.

**Parameters**:
- `text` (string, required) - Text to summarize
- `model` (string) - Model to use (default: "command-r")
- `length` (string) - "short", "medium", or "long"
- `format` (string) - "paragraph" or "bullets"

### cohere_classify

Classify texts based on example training data.

**Parameters**:
- `inputs` (array of strings, required) - Texts to classify
- `examples` (array of objects, required) - Training examples with "text" and "label" keys
- `model` (string) - Model to use (default: "embed-english-v3.0")

## Available Resources

### cohere://models

Lists all available Cohere models with their capabilities, context lengths, and recommended use cases.

### cohere://config

Shows current server configuration including default models and settings.

## Configuration

The server can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `COHERE_API_KEY` | Your Cohere API key | *Required* |
| `COHERE_DEFAULT_CHAT_MODEL` | Default chat model | command-a-03-2025 |
| `COHERE_DEFAULT_EMBED_MODEL` | Default embedding model | embed-english-v3.0 |
| `COHERE_DEFAULT_RERANK_MODEL` | Default rerank model | rerank-english-v3.0 |
| `COHERE_TIMEOUT` | API request timeout (seconds) | 60 |
| `COHERE_MAX_RETRIES` | Maximum API retry attempts | 3 |

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs testing and linting tools:
- pytest - Testing framework
- black - Code formatter
- ruff - Linter
- mypy - Type checker

### Running Tests

```bash
pytest
```

Run with coverage:
```bash
pytest --cov=cohere_mcp --cov-report=html
```

### Code Quality

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
ruff check src/ tests/
```

Type check:
```bash
mypy src/
```

## Project Structure

```
cohere-mcp-server/
├── src/
│   └── cohere_mcp/
│       ├── __init__.py      # Package initialization
│       ├── config.py        # Configuration management
│       ├── client.py        # Cohere API client wrapper
│       └── server.py        # MCP server implementation
├── tests/                   # Test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_client.py
│   └── test_server.py
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Use Cases

### Retrieval Augmented Generation (RAG)

1. **Embed documents**: Use `cohere_embed` with `input_type="search_document"`
2. **Embed query**: Use `cohere_embed` with `input_type="search_query"`
3. **Rerank results**: Use `cohere_rerank` to improve relevance
4. **Generate response**: Use `cohere_chat` with retrieved context

### Semantic Search

1. Index documents using `cohere_embed`
2. Search with query embeddings
3. Optionally rerank with `cohere_rerank`

### Multilingual Applications

Use `cohere_aya_chat` for conversations in:
- English, Spanish, French, German, Italian, Portuguese
- Arabic, Hebrew, Turkish
- Chinese, Japanese, Korean
- Hindi, Bengali, and many more

## Troubleshooting

### "Cohere client not initialized" error

Make sure you've set the `COHERE_API_KEY` environment variable:
```bash
export COHERE_API_KEY="your-key-here"
```

### API Key Issues

- Verify your API key at https://dashboard.cohere.com/api-keys
- Ensure the key has proper permissions
- Check for any whitespace in the key value

### Connection Issues

- Check your internet connection
- Verify Cohere API status
- Increase timeout with `COHERE_TIMEOUT` environment variable

## API Pricing

Refer to [Cohere's pricing page](https://cohere.com/pricing) for current API costs.

## Resources

- [Cohere Documentation](https://docs.cohere.com/)
- [Cohere API Reference](https://docs.cohere.com/reference/about)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Cohere Dashboard](https://dashboard.cohere.com/)

## License

MIT License - See LICENSE file for details.

## Support

For issues and questions:
- Cohere API issues: [Cohere Support](https://cohere.com/support)
- MCP Server issues: Open an issue in this repository
- MCP Protocol: [MCP Documentation](https://modelcontextprotocol.io/)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

---

Built with [Cohere](https://cohere.com/) and [Model Context Protocol](https://modelcontextprotocol.io/)
