#!/usr/bin/env python3
"""
Test script for Claude integration with the llm-wrapper package.

This script tests the Claude family implementation:
- Sonnet model (default)
- Opus model
- Rate limiting
- JSON responses
"""

import asyncio
import os
from llm_wrapper import get_llm_manager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_claude_sonnet():
    """Test Claude Sonnet model."""
    print("🤖 Test: Claude Sonnet (Default)")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    success, result = await llm.call_claude(
        prompt="Write a haiku about artificial intelligence",
        temperature=0.7
    )
    
    if success:
        print("✅ Sonnet response:")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_claude_opus():
    """Test Claude Opus model."""
    print("🤖 Test: Claude Opus")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    success, result = await llm.call_llm(
        family="claude",
        model="claude-opus-4-1-20250805",
        prompt="Explain quantum computing in one sentence",
        max_tokens=100
    )
    
    if success:
        print("✅ Opus response:")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_claude_json():
    """Test Claude JSON generation."""
    print("🤖 Test: Claude JSON Response")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    success, result = await llm.call_claude(
        prompt="Generate a JSON object with fields: name (a random name), age (between 20-50), city (any major city)",
        return_json=True
    )
    
    if success:
        print("✅ JSON response:")
        print(result)
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_rate_limits():
    """Test rate limit tracking for Claude models."""
    print("📊 Test: Claude Rate Limits")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Check Sonnet limits
    print("Sonnet model limits:")
    status = llm.get_rate_limit_status("claude", "claude-sonnet-4-20250514")
    print(f"  Calls: {status.get('calls_used', 0)}/{status.get('calls_limit', 'Unknown')}")
    print(f"  Tokens: {status.get('tokens_used', 0)}/{status.get('tokens_limit', 'Unknown')}")
    
    # Check Opus limits
    print("\nOpus model limits:")
    status = llm.get_rate_limit_status("claude", "claude-opus-4-1-20250805")
    print(f"  Calls: {status.get('calls_used', 0)}/{status.get('calls_limit', 'Unknown')}")
    print(f"  Tokens: {status.get('tokens_used', 0)}/{status.get('tokens_limit', 'Unknown')}")
    
    print()
    return True

async def test_model_comparison():
    """Compare responses from both Claude models."""
    print("🔄 Test: Model Comparison")
    print("-" * 50)
    
    llm = get_llm_manager()
    prompt = "What is 2+2? Answer in exactly one word."
    
    # Test Sonnet
    print("Testing Sonnet...")
    success1, result1 = await llm.call_claude(prompt=prompt)
    if success1:
        print(f"Sonnet: {result1.get('text', result1)}")
    
    # Test Opus
    print("Testing Opus...")
    success2, result2 = await llm.call_llm(
        family="claude",
        model="claude-opus-4-1-20250805",
        prompt=prompt
    )
    if success2:
        print(f"Opus: {result2.get('text', result2)}")
    
    print()
    return success1 and success2

async def main():
    """Run all Claude tests."""
    print("🚀 Claude Integration Test Suite")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set your ANTHROPIC_API_KEY in a .env file")
        print("   Create a .env file with: ANTHROPIC_API_KEY=your_key_here")
        print("   Get your key from: https://console.anthropic.com/")
        return
    
    # Run tests
    tests_passed = []
    
    try:
        # Test rate limits first (before any calls)
        print("\n📊 Initial Rate Limits:")
        tests_passed.append(await test_rate_limits())
        
        # Test Sonnet (default)
        tests_passed.append(await test_claude_sonnet())
        
        # Test Opus
        tests_passed.append(await test_claude_opus())
        
        # Test JSON response
        tests_passed.append(await test_claude_json())
        
        # Test model comparison
        tests_passed.append(await test_model_comparison())
        
        # Check rate limits again
        print("\n📊 Final Rate Limits:")
        await test_rate_limits()
        
        # Summary
        print("\n" + "=" * 50)
        passed = sum(tests_passed)
        total = len(tests_passed)
        
        if all(tests_passed):
            print(f"✅ All tests passed! ({passed}/{total})")
        else:
            print(f"⚠️ Some tests failed: {passed}/{total} passed")
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())