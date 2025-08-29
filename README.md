# llm-wrapper

A Python wrapper around LLMs (Gemini, Claude) with rate limiting and a unified interface.

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **Rate Limiting**: Automatic rate limiting to prevent API quota exhaustion
- **Async Support**: Full async/await support for concurrent operations
- **File Support**: Handle text, images, and PDFs as input
- **Image Generation**: Support for Gemini's Imagen models
- **Singleton Pattern**: Efficient resource management with automatic client reuse

## Installation

### From GitHub (latest)

```bash
uv add git+https://github.com/uncharted-research/llm-wrapper.git
```

### From PyPI (when published)

```bash
uv add llm-wrapper
```

## Quick Start

### Basic Text Generation

```python
import asyncio
from llm_wrapper import get_llm_manager

# API key is automatically loaded from .env file
async def main():
    llm = get_llm_manager()
    
    # Simple text generation
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.0-flash",
        prompt="Explain quantum computing in simple terms"
    )
    
    if success:
        print(result["text"])
    else:
        print(f"Error: {result}")

asyncio.run(main())
```

### With File Input

```python
import asyncio
from llm_wrapper import get_llm_manager

async def analyze_document():
    llm = get_llm_manager()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-pro",
        prompt="Summarize this document",
        file_path="document.pdf"
    )
    
    if success:
        print(result["text"])

asyncio.run(analyze_document())
```

### Image Generation

```python
import asyncio
from llm_wrapper import get_llm_manager

async def generate_image():
    llm = get_llm_manager()
    
    success, image = await llm.generate_image(
        model="imagen-3.0-generate-002",
        prompt="A futuristic city at sunset"
    )
    
    if success:
        image.save("generated_image.png")
        print("Image saved!")

asyncio.run(generate_image())
```

### Rate Limit Monitoring

```python
from llm_wrapper import get_llm_manager

llm = get_llm_manager()
status = llm.get_rate_limit_status("gemini", "gemini-2.0-flash")
print(f"API calls remaining: {status['calls_remaining']}")
print(f"Tokens remaining: {status['tokens_remaining']}")
```

## Supported Models

### Gemini Models

- `gemini-2.5-pro`: 50 calls/min, 800K tokens/min
- `gemini-2.5-flash`: 400 calls/min, 500K tokens/min  
- `gemini-2.0-flash`: 800 calls/min, 1M tokens/min
- `imagen-3.0-generate-002`: 5 calls/min (image generation)

### Claude Models

*Coming soon*

## Configuration

### Environment Variables (.env file)

Create a `.env` file in your project root with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The package automatically loads environment variables from the `.env` file using `python-dotenv`.

**Quick setup**: Copy `env.example` to `.env` and fill in your API keys:
```bash
cp env.example .env
# Then edit .env with your actual API keys
```

### API Keys Setup

1. **Gemini**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Claude**: *Coming soon*

### Example .env file

```env
# Google AI Studio API Key
GEMINI_API_KEY=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890

# Future: Claude API Key (when implemented)
# CLAUDE_API_KEY=your_claude_api_key_here
```

## Development

### Local Installation

```bash
git clone https://github.com/christophmayer/llm-wrapper.git
cd llm-wrapper
uv sync --dev
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run isort .
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
