#!/usr/bin/env python3
"""
Test script for image_data parameter integration with Gemini models.

This script tests the new image_data functionality:
- Sending image bytes directly to Gemini
- Different MIME types
- Error handling
- Integration with other parameters
"""

import asyncio
import os
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage
from llm_wrapper import get_llm_manager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_image_data_basic():
    """Test basic image_data functionality with a generated image."""
    print("🖼️ Test: Basic Image Data")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Create a simple test image in memory
    img = PILImage.new('RGB', (100, 100), color='red')
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    img_bytes = img_bytes_io.getvalue()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Describe this image in one sentence",
        image_data=img_bytes,
        image_mime_type="image/jpeg"
    )
    
    if success:
        print("✅ Response:")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_image_data_png():
    """Test image_data with PNG format."""
    print("🖼️ Test: PNG Image Data")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Create a test image with transparency
    img = PILImage.new('RGBA', (100, 100), color=(0, 0, 255, 128))
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='PNG')
    img_bytes = img_bytes_io.getvalue()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="What color is this image?",
        image_data=img_bytes,
        image_mime_type="image/png"
    )
    
    if success:
        print("✅ Response:")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_image_data_with_file():
    """Test that image_data takes precedence over file_path."""
    print("🖼️ Test: Image Data Precedence")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Create image bytes
    img = PILImage.new('RGB', (50, 50), color='green')
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    img_bytes = img_bytes_io.getvalue()
    
    # This should use image_data, not file_path
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="What do you see?",
        image_data=img_bytes,
        file_path="README.md",  # This should be ignored
        image_mime_type="image/jpeg"
    )
    
    if success:
        print("✅ Response (should describe an image, not text):")
        print(result.get("text", result))
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_claude_restriction():
    """Test that Claude properly rejects image_data."""
    print("🚫 Test: Claude Image Restriction")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Create dummy image bytes
    img_bytes = b"dummy_image_data"
    
    success, result = await llm.call_llm(
        family="claude",
        model="claude-sonnet-4-20250514",
        prompt="Describe this image",
        image_data=img_bytes
    )
    
    if not success and "does not support" in str(result):
        print("✅ Claude correctly rejected image_data")
        print(f"Error message: {result}")
        return True
    else:
        print("❌ Claude should have rejected image_data")
        return False
    
    print()

async def test_empty_image_data():
    """Test error handling for empty image data."""
    print("❌ Test: Empty Image Data")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Describe this image",
        image_data=b"",  # Empty bytes
        image_mime_type="image/jpeg"
    )
    
    if not success and "empty" in str(result).lower():
        print("✅ Correctly handled empty image data")
        print(f"Error message: {result}")
        return True
    else:
        print("❌ Should have rejected empty image data")
        return False
    
    print()

async def test_unsupported_mime_type():
    """Test error handling for unsupported MIME type."""
    print("❌ Test: Unsupported MIME Type")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Create dummy image bytes
    img_bytes = b"dummy_image_data"
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Describe this image",
        image_data=img_bytes,
        image_mime_type="image/bmp"  # Unsupported type
    )
    
    if not success and "Unsupported MIME type" in str(result):
        print("✅ Correctly rejected unsupported MIME type")
        print(f"Error message: {result}")
        return True
    else:
        print("❌ Should have rejected unsupported MIME type")
        return False
    
    print()

async def test_with_json_response():
    """Test image_data with JSON response."""
    print("🖼️ Test: Image Data with JSON Response")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Create a simple test image
    img = PILImage.new('RGB', (100, 100), color='yellow')
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    img_bytes = img_bytes_io.getvalue()
    
    success, result = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="Analyze this image and return a JSON with fields: 'main_color' and 'shape'",
        image_data=img_bytes,
        image_mime_type="image/jpeg",
        return_json=True
    )
    
    if success:
        print("✅ JSON Response:")
        print(result)
    else:
        print(f"❌ Error: {result}")
    
    print()
    return success

async def test_rate_limits():
    """Test that rate limiting works with image_data."""
    print("📊 Test: Rate Limits with Image Data")
    print("-" * 50)
    
    llm = get_llm_manager()
    
    # Check initial rate limits
    status_before = llm.get_rate_limit_status("gemini", "gemini-2.5-flash")
    calls_before = status_before.get('calls_used', 0)
    
    # Make a call with image_data
    img = PILImage.new('RGB', (50, 50), color='blue')
    img_bytes_io = BytesIO()
    img.save(img_bytes_io, format='JPEG')
    img_bytes = img_bytes_io.getvalue()
    
    success, _ = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt="What color?",
        image_data=img_bytes,
        image_mime_type="image/jpeg"
    )
    
    # Check rate limits after
    status_after = llm.get_rate_limit_status("gemini", "gemini-2.5-flash")
    calls_after = status_after.get('calls_used', 0)
    
    if calls_after > calls_before:
        print(f"✅ Rate limit updated: {calls_before} -> {calls_after}")
        return True
    else:
        print("❌ Rate limit counter did not update")
        return False
    
    print()

async def main():
    """Run all image_data tests."""
    print("🚀 Image Data Integration Test Suite")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set your GEMINI_API_KEY in a .env file")
        print("   Create a .env file with: GEMINI_API_KEY=your_key_here")
        print("   Get your key from: https://aistudio.google.com/app/apikey")
        return
    
    # Run tests
    tests_passed = []
    
    try:
        # Test basic functionality
        tests_passed.append(await test_image_data_basic())
        
        # Test PNG format
        tests_passed.append(await test_image_data_png())
        
        # Test precedence over file_path
        tests_passed.append(await test_image_data_with_file())
        
        # Test Claude restriction
        if os.getenv("ANTHROPIC_API_KEY"):
            tests_passed.append(await test_claude_restriction())
        else:
            print("⚠️ Skipping Claude test (no API key)")
        
        # Test error cases
        tests_passed.append(await test_empty_image_data())
        tests_passed.append(await test_unsupported_mime_type())
        
        # Test with JSON response
        tests_passed.append(await test_with_json_response())
        
        # Test rate limits
        tests_passed.append(await test_rate_limits())
        
        # Summary
        print("\n" + "=" * 50)
        passed = sum(1 for t in tests_passed if t)
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