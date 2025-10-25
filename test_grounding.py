#!/usr/bin/env python3
"""Test script for Gemini grounding with Google Search."""

import asyncio
import json
from pathlib import Path
from llm_wrapper import LLMManager

async def test_grounding():
    """Test Gemini with Google Search grounding."""

    # Initialize the LLM manager
    llm_manager = LLMManager()

    print("=" * 80)
    print("Testing Gemini Grounding with Google Search")
    print("=" * 80)

    # Test 1: Basic grounding with Google Search
    print("\n1. Testing basic grounding with Google Search:")
    print("-" * 40)

    prompt = "What is the current stock price of Apple (AAPL)? Please provide recent information with sources."

    success, response = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_google_search=True
    )

    if success:
        print(f"Success: {success}")
        print(f"\nResponse type: {type(response)}")

        if isinstance(response, dict):
            print(f"\nKeys in response: {list(response.keys())}")

            # Print the main text
            if "text" in response:
                print(f"\nText response (first 500 chars):")
                print(response["text"][:500])

            # Print sources if available
            if "sources" in response:
                print(f"\nNumber of sources found: {len(response['sources'])}")
                print("\nSources:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"  {i}. {source}")
            else:
                print("\nNo sources found in response")

            # Print search queries if available
            if "search_queries" in response:
                print(f"\nSearch queries used: {response['search_queries']}")
        else:
            print(f"\nResponse: {response}")
    else:
        print(f"Error: {response}")

    # Test 2: URL Context grounding
    print("\n" + "=" * 80)
    print("2. Testing URL Context grounding:")
    print("-" * 40)

    prompt2 = "What are the main features of Python 3.12? Please search for official information."

    success2, response2 = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt2,
        enable_url_context=True
    )

    if success2:
        print(f"Success: {success2}")

        if isinstance(response2, dict):
            print(f"\nKeys in response: {list(response2.keys())}")

            if "text" in response2:
                print(f"\nText response (first 500 chars):")
                print(response2["text"][:500])

            if "sources" in response2:
                print(f"\nNumber of sources found: {len(response2['sources'])}")
                print("\nSources:")
                for i, source in enumerate(response2["sources"], 1):
                    print(f"  {i}. {source}")
        else:
            print(f"\nResponse: {response2}")
    else:
        print(f"Error: {response2}")

    # Test 3: Both tools enabled
    print("\n" + "=" * 80)
    print("3. Testing with both Google Search and URL Context:")
    print("-" * 40)

    prompt3 = "What are the latest features in TypeScript 5.0? Search for official documentation."

    success3, response3 = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt3,
        enable_google_search=True,
        enable_url_context=True
    )

    if success3:
        print(f"Success: {success3}")

        if isinstance(response3, dict):
            print(f"\nKeys in response: {list(response3.keys())}")

            if "sources" in response3:
                print(f"\nNumber of sources found: {len(response3['sources'])}")
                print("\nSources:")
                for i, source in enumerate(response3["sources"], 1):
                    print(f"  {i}. {source}")
    else:
        print(f"Error: {response3}")

    # Test 4: Backward compatibility (no grounding)
    print("\n" + "=" * 80)
    print("4. Testing backward compatibility (no grounding):")
    print("-" * 40)

    prompt4 = "What is 2 + 2?"

    success4, response4 = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt4
    )

    if success4:
        print(f"Success: {success4}")
        print(f"\nResponse type: {type(response4)}")

        if isinstance(response4, dict):
            print(f"Keys in response: {list(response4.keys())}")
            print(f"Response: {response4}")
        else:
            print(f"Response: {response4}")
    else:
        print(f"Error: {response4}")

    # Test 5: JSON return format with grounding
    print("\n" + "=" * 80)
    print("5. Testing JSON return with grounding:")
    print("-" * 40)

    prompt5 = 'Return a JSON object with the current weather in New York. Format: {"temperature": "...", "condition": "..."}'

    success5, response5 = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt5,
        return_json=True,
        enable_google_search=True
    )

    if success5:
        print(f"Success: {success5}")
        print(f"\nResponse type: {type(response5)}")

        if isinstance(response5, dict):
            print(f"Keys in response: {list(response5.keys())}")
            print(f"Response: {json.dumps(response5, indent=2)}")
    else:
        print(f"Error: {response5}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_grounding())