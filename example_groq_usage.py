"""
Simple usage examples for Groq integration.

Make sure to set GROQ_API_KEY in your environment or .env file.
"""

import asyncio
from llm_wrapper.llm import get_llm_manager


async def main():
    manager = get_llm_manager()

    # Example 1: Simple call with default model
    print("Example 1: Simple call")
    success, result = await manager.call_groq(
        prompt="What is the capital of France?"
    )
    if success:
        print(result["text"])
    print()

    # Example 2: Specify a different model
    print("Example 2: Different model")
    success, result = await manager.call_via_groq(
        model="qwen/qwen3-32b",
        prompt="Write a one-line joke about programming."
    )
    if success:
        print(result["text"])
    print()

    # Example 3: Streaming call
    print("Example 3: Streaming call")
    success, result = await manager.call_groq(
        prompt="Count from 1 to 5",
        stream=True
    )
    if success:
        print(result["text"])
    print()

    # Example 4: JSON response
    print("Example 4: JSON response")
    success, result = await manager.call_via_groq(
        model="llama-3.1-8b-instant",
        prompt='Return a JSON object with "name": "Alice" and "age": 30',
        return_json=True
    )
    if success:
        print(f"Parsed JSON: {result}")
    print()

    # Example 5: Check rate limits
    print("Example 5: Rate limit status")
    status = manager.get_rate_limit_status("groq", "llama-3.3-70b-versatile")
    print(f"Rate limits: {status}")


if __name__ == "__main__":
    asyncio.run(main())
