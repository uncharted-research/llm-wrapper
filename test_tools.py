#!/usr/bin/env python3
"""
Test script for Google Search and URL Context tools integration with Gemini.
"""

import asyncio
from llm_wrapper import get_llm_manager


async def test_url_context():
    """Test URL context tool."""
    print("\n" + "="*50)
    print("Testing URL Context Tool")
    print("="*50)

    llm_manager = get_llm_manager()

    success, result = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Analyze this company and tell me what they do: https://www.anthropic.com",
        enable_url_context=True,
        thinking_budget=-1
    )

    if success:
        print("✅ URL Context test successful!")
        print(f"Response: {result.get('text', result)[:500]}...")
    else:
        print(f"❌ URL Context test failed: {result}")

    return success


async def test_google_search():
    """Test Google Search tool."""
    print("\n" + "="*50)
    print("Testing Google Search Tool")
    print("="*50)

    llm_manager = get_llm_manager()

    success, result = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="What are the latest developments in quantum computing in 2024?",
        enable_google_search=True,
        thinking_budget=-1
    )

    if success:
        print("✅ Google Search test successful!")
        print(f"Response: {result.get('text', result)[:500]}...")
    else:
        print(f"❌ Google Search test failed: {result}")

    return success


async def test_both_tools():
    """Test both tools together."""
    print("\n" + "="*50)
    print("Testing Both Tools Together")
    print("="*50)

    llm_manager = get_llm_manager()

    success, result = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Compare Apple (apple.com) and Microsoft (microsoft.com) based on their latest products and announcements",
        enable_google_search=True,
        enable_url_context=True,
        thinking_budget=10000
    )

    if success:
        print("✅ Both tools test successful!")
        print(f"Response: {result.get('text', result)[:500]}...")
    else:
        print(f"❌ Both tools test failed: {result}")

    return success


async def test_backward_compatibility():
    """Test that existing functionality still works without tools."""
    print("\n" + "="*50)
    print("Testing Backward Compatibility (No Tools)")
    print("="*50)

    llm_manager = get_llm_manager()

    success, result = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="What is 2 + 2?",
        # No tools enabled - should work as before
    )

    if success:
        print("✅ Backward compatibility test successful!")
        print(f"Response: {result.get('text', result)}")
    else:
        print(f"❌ Backward compatibility test failed: {result}")

    return success


async def main():
    """Run all tests."""
    print("\n" + "🚀 Starting Gemini Tools Integration Tests")

    # Run tests
    results = []

    # Test backward compatibility first
    results.append(await test_backward_compatibility())

    # Test individual tools
    results.append(await test_url_context())
    results.append(await test_google_search())

    # Test combined tools
    results.append(await test_both_tools())

    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)

    test_names = [
        "Backward Compatibility",
        "URL Context",
        "Google Search",
        "Both Tools"
    ]

    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")

    total_passed = sum(results)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())