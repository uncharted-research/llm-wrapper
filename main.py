#!/usr/bin/env python3
"""
Example usage of the llm-wrapper package.

This script demonstrates basic functionality of the LLM wrapper.
Make sure to set your GEMINI_API_KEY environment variable before running.
"""

import asyncio
import os
from llm_wrapper import get_llm_manager
from dotenv import load_dotenv
load_dotenv()

async def demo_text_generation():
    """Demonstrate basic text generation."""
    print("🤖 Demo: Text Generation")
    print("-" * 40)
    
    llm = get_llm_manager()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Write a haiku about programming",
        temperature=0.7
    )
    
    if success:
        print("Generated text:")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()
    
async def demo_json_generation():
    """Demonstrate JSON generation."""
    print("🤖 Demo: JSON Generation")
    print("-" * 40)
    
    llm = get_llm_manager()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Generate a JSON object with a random number between 1 and 100",
        return_json=True
    )
    
    if success:
        print("Generated JSON:")
        print(result)
    else:
        print(f"❌ Error: {result}")
    
    print()

async def demo_image_data():
    """Demonstrate image_data parameter usage."""
    print("🖼️ Demo: Image Data Parameter")
    print("-" * 40)
    
    llm = get_llm_manager()
    
    # Create a simple test image
    from PIL import Image
    from io import BytesIO
    
    # Create a colorful test image
    img = Image.new('RGB', (100, 100))
    pixels = img.load()
    for i in range(100):
        for j in range(100):
            pixels[i, j] = (i * 2, j * 2, (i + j) // 2)
    
    # Convert to bytes
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    img_bytes = img_bytes_io.getvalue()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Describe the colors and pattern in this image",
        image_data=img_bytes,
        image_mime_type="image/jpeg"
    )
    
    if success:
        print("Generated description:")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()

async def demo_rate_limits(label="Rate Limit Status"):
    """Demonstrate rate limit monitoring."""
    print(f"📊 Demo: {label}")
    print("-" * 40)
    
    llm = get_llm_manager()
    status = llm.get_rate_limit_status("gemini", "gemini-2.5-flash")
    
    print(f"Calls used: {status.get('calls_used', 0)}/{status.get('calls_limit', 'Unknown')}")
    print(f"Calls remaining: {status.get('calls_remaining', 'Unknown')}")
    
    if 'tokens_used' in status:
        print(f"Tokens used: {status['tokens_used']}/{status['tokens_limit']}")
        print(f"Tokens remaining: {status['tokens_remaining']}")
    
    print()


async def main():
    """Run the demo."""
    print("🚀 LLM Wrapper Demo")
    print("=" * 50)
    
    # Check if API key is set (load_dotenv is called automatically by LLMManager)
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set your GEMINI_API_KEY in a .env file")
        print("   Create a .env file with: GEMINI_API_KEY=your_key_here")
        print("   Get your key from: https://aistudio.google.com/app/apikey")
        return
    
    try:
        await demo_rate_limits("Rate Limits BEFORE Any Calls")
        await demo_text_generation()
        await demo_rate_limits("Rate Limits AFTER Text Generation")
        await demo_json_generation()
        await demo_rate_limits("Rate Limits AFTER JSON Generation")
        await demo_image_data()
        await demo_rate_limits("Rate Limits AFTER Image Data")
        
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
