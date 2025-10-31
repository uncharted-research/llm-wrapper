"""
Test script for Groq integration in LLMManager.

This demonstrates:
1. Non-streaming calls
2. Streaming calls
3. JSON parsing
4. Error handling for invalid models
5. Rate limiting
"""

import asyncio
from llm_wrapper.llm import get_llm_manager


async def test_basic_call():
    """Test basic non-streaming Groq call."""
    print("\n=== Test 1: Basic Non-Streaming Call ===")
    manager = get_llm_manager()

    success, result = await manager.call_via_groq(
        model="llama-3.3-70b-versatile",
        prompt="What is the capital of France? Answer in one sentence."
    )

    if success:
        print(f"Success: {result}")
    else:
        print(f"Error: {result}")


async def test_streaming_call():
    """Test streaming Groq call."""
    print("\n=== Test 2: Streaming Call ===")
    manager = get_llm_manager()

    success, result = await manager.call_via_groq_stream(
        model="qwen/qwen3-32b",
        prompt="Write a haiku about programming."
    )

    if success:
        print(f"Success: {result}")
    else:
        print(f"Error: {result}")


async def test_json_parsing():
    """Test JSON parsing from response."""
    print("\n=== Test 3: JSON Parsing ===")
    manager = get_llm_manager()

    success, result = await manager.call_via_groq(
        model="llama-3.1-8b-instant",
        prompt='Return a JSON object with fields "name", "age", and "city" for a fictional person.',
        return_json=True
    )

    if success:
        print(f"Success: {result}")
        print(f"Type: {type(result)}")
    else:
        print(f"Error: {result}")


async def test_convenience_method():
    """Test the convenience method call_groq."""
    print("\n=== Test 4: Convenience Method ===")
    manager = get_llm_manager()

    # Non-streaming
    success, result = await manager.call_groq(
        prompt="What is 2+2? Answer with just the number."
    )

    if success:
        print(f"Non-streaming: {result}")
    else:
        print(f"Error: {result}")

    # Streaming
    success, result = await manager.call_groq(
        prompt="Count from 1 to 5.",
        stream=True
    )

    if success:
        print(f"Streaming: {result}")
    else:
        print(f"Error: {result}")


async def test_invalid_model():
    """Test error handling for invalid model."""
    print("\n=== Test 5: Invalid Model Error Handling ===")
    manager = get_llm_manager()

    success, result = await manager.call_via_groq(
        model="this-model-does-not-exist",
        prompt="Hello"
    )

    if success:
        print(f"Unexpected success: {result}")
    else:
        print(f"Expected error: {result}")


async def test_rate_limit_status():
    """Test rate limit status checking."""
    print("\n=== Test 6: Rate Limit Status ===")
    manager = get_llm_manager()

    # Make a call first
    await manager.call_via_groq(
        model="llama-3.3-70b-versatile",
        prompt="Hi"
    )

    # Check rate limit status
    status = manager.get_rate_limit_status("groq", "llama-3.3-70b-versatile")
    print(f"Rate limit status: {status}")


async def main():
    """Run all tests."""
    print("Testing Groq Integration")
    print("=" * 50)

    await test_basic_call()
    await test_streaming_call()
    await test_json_parsing()
    await test_convenience_method()
    await test_invalid_model()
    await test_rate_limit_status()

    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
