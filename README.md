# llm-wrapper

A Python wrapper around LLMs (Gemini, Claude) with rate limiting and a unified interface.

## Features

- **Unified Interface**: Single API for multiple LLM providers (Gemini and Claude)
- **Rate Limiting**: Automatic rate limiting to prevent API quota exhaustion
- **Async Support**: Full async/await support for concurrent operations
- **File Support**: Handle text, images, and PDFs as input (Gemini only)
- **Image Data Support**: Send image bytes directly without file paths (Gemini only)
- **Image Generation**: Support for Gemini's Imagen models
- **Claude Integration**: Full support for Claude Opus and Sonnet models
- **Singleton Pattern**: Efficient resource management with automatic client reuse

## Installation

### From GitHub (latest)

```bash
uv add git+https://github.com/uncharted-research/llm-wrapper.git
```

```bash
pip install --upgrade --force-reinstall git+https://github.com/uncharted-research/llm-wrapper.git
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

### With Image Data (Gemini Only)

```python
import asyncio
from llm_wrapper import get_llm_manager

async def analyze_image_bytes():
    llm = get_llm_manager()
    
    # Read image as bytes
    with open("photo.jpg", "rb") as f:
        img_bytes = f.read()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Describe what you see in this image",
        image_data=img_bytes,
        image_mime_type="image/jpeg"  # Supports: image/jpeg, image/png, image/webp, image/gif
    )
    
    if success:
        print(result["text"])

asyncio.run(analyze_image_bytes())
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

### Using Claude Models

```python
import asyncio
from llm_wrapper import get_llm_manager

async def use_claude():
    llm = get_llm_manager()
    
    # Use Claude with default model (Sonnet)
    success, result = await llm.call_claude(
        prompt="Write a haiku about programming",
        temperature=0.7
    )
    
    if success:
        print(result["text"])
    
    # Use specific Claude model (Opus)
    success, result = await llm.call_llm(
        family="claude",
        model="claude-opus-4-1-20250805",
        prompt="Explain machine learning",
        max_tokens=200
    )
    
    if success:
        print(result["text"])
    
    # Get JSON response from Claude
    success, result = await llm.call_claude(
        prompt="Generate a JSON object with name and age fields",
        return_json=True
    )
    
    if success:
        print(result)  # Will be a parsed dict

asyncio.run(use_claude())
```

### Working with Images

The library supports two ways to send images to Gemini models:

#### 1. Using File Path
```python
success, result = await llm.call_llm(
    family="gemini",
    model="gemini-2.5-flash",
    prompt="Analyze this image",
    file_path="image.jpg"
)
```

#### 2. Using Image Data (bytes)
```python
# From file
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

# Or from PIL Image
from PIL import Image
from io import BytesIO

img = Image.new('RGB', (100, 100), color='red')
buffer = BytesIO()
img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

# Send to Gemini
success, result = await llm.call_llm(
    family="gemini",
    model="gemini-2.5-flash",
    prompt="What do you see?",
    image_data=image_bytes,
    image_mime_type="image/jpeg"  # or "image/png", "image/webp", "image/gif"
)
```

**Note**: Image support is only available for Gemini models. Claude models will return an error if image_data or file_path is provided.

### Rate Limit Monitoring

```python
from llm_wrapper import get_llm_manager

llm = get_llm_manager()

# Check Gemini limits
status = llm.get_rate_limit_status("gemini", "gemini-2.0-flash")
print(f"Gemini - API calls remaining: {status['calls_remaining']}")
print(f"Gemini - Tokens remaining: {status['tokens_remaining']}")

# Check Claude limits
status = llm.get_rate_limit_status("claude", "claude-sonnet-4-20250514")
print(f"Claude - API calls remaining: {status['calls_remaining']}")
print(f"Claude - Tokens remaining: {status['tokens_remaining']}")
```

## Supported Models

### Gemini Models

- `gemini-2.5-pro`: 50 calls/min, 800K tokens/min
- `gemini-2.5-flash`: 400 calls/min, 500K tokens/min  
- `gemini-2.0-flash`: 800 calls/min, 1M tokens/min
- `imagen-3.0-generate-002`: 5 calls/min (image generation)

### Claude Models

- `claude-sonnet-4-20250514`: 50 calls/min, 300K tokens/min (default)
- `claude-opus-4-1-20250805`: 50 calls/min, 300K tokens/min

## Configuration

### Environment Variables (.env file)

Create a `.env` file in your project root with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The package automatically loads environment variables from the `.env` file using `python-dotenv`.

**Quick setup**: Copy `env.example` to `.env` and fill in your API keys:
```bash
cp env.example .env
# Then edit .env with your actual API keys
```

### API Keys Setup

1. **Gemini**: Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. **Claude**: Get your API key from [Anthropic Console](https://console.anthropic.com/)

### Example .env file

```env
# Google AI Studio API Key
GEMINI_API_KEY=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890

# Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
