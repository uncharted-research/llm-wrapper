"""
Test the wrapper fixes to ensure everything works correctly.

Tests:
1. Google Search grounding with JSON response
2. URL context grounding
3. Both tools together
4. Verify grounding_enabled metadata
5. Verify markdown stripping works
"""

import asyncio
import json
from llm_wrapper import LLMManager


async def test_google_search_grounding():
    """Test Google Search grounding with JSON parsing."""
    print("=" * 80)
    print("TEST 1: Google Search Grounding with JSON")
    print("=" * 80)

    llm = LLMManager()

    prompt = """Search for information about medical device compliance software market size.
Return 3 data points as a JSON array with this format:
[
    {
        "market_size": [NUMBER, "CURRENCY", "MULTIPLIER"],
        "year": YEAR,
        "market_description": "DESCRIPTION",
        "geography": "GEOGRAPHY"
    }
]

Return ONLY valid JSON, no markdown."""

    success, response = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_google_search=True,
        return_json=True
    )

    print(f"\nSuccess: {success}")
    if success:
        print(f"Response type: {type(response)}")
        print(f"Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")

        # Check for grounding_enabled metadata
        if "grounding_enabled" in response:
            print(f"✓ grounding_enabled present: {response['grounding_enabled']}")
        else:
            print("✗ grounding_enabled missing")

        # Check for sources
        if "sources" in response:
            print(f"✓ sources present: {len(response['sources'])} sources")
        else:
            print("✗ sources missing")

        # Check for search queries
        if "search_queries" in response:
            print(f"✓ search_queries present: {len(response['search_queries'])} queries")
        else:
            print("✗ search_queries missing")

        # Check if text is valid JSON
        if "text" in response:
            try:
                data = json.loads(response["text"]) if isinstance(response["text"], str) else response["text"]
                print(f"✓ JSON parsing successful: {len(data)} items")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")
        elif isinstance(response, list):
            print(f"✓ Response is already parsed JSON: {len(response)} items")

        print("\nFull response structure:")
        print(json.dumps({k: str(v)[:100] + "..." if len(str(v)) > 100 else v
                          for k, v in response.items() if k != "grounding_metadata"},
                         indent=2))

    else:
        print(f"✗ Error: {response}")

    return success


async def test_url_context():
    """Test URL context grounding."""
    print("\n" + "=" * 80)
    print("TEST 2: URL Context Grounding")
    print("=" * 80)

    llm = LLMManager()

    prompt = """Based on https://www.certhub.de/en, summarize what CertHub does in 2-3 sentences."""

    success, response = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_url_context=True
    )

    print(f"\nSuccess: {success}")
    if success:
        print(f"Response type: {type(response)}")

        # Check for grounding_enabled metadata
        if "grounding_enabled" in response:
            print(f"✓ grounding_enabled present: {response['grounding_enabled']}")
            if response['grounding_enabled']['url_context']:
                print("  ✓ url_context is True")
            if not response['grounding_enabled']['google_search']:
                print("  ✓ google_search is False")
        else:
            print("✗ grounding_enabled missing")

        # Check for sources with url_context type
        if "sources" in response:
            print(f"✓ sources present: {len(response['sources'])} sources")
            url_context_sources = [s for s in response['sources'] if s.get('source_type') == 'url_context']
            if url_context_sources:
                print(f"  ✓ {len(url_context_sources)} sources marked as 'url_context'")
                for source in url_context_sources:
                    print(f"    - {source.get('url')}")
                    print(f"      Status: {source.get('retrieval_status')}")
        else:
            print("✗ sources missing")

        # Print response text (truncated)
        if "text" in response:
            print(f"\nResponse text (first 200 chars):")
            print(response["text"][:200])

    else:
        print(f"✗ Error: {response}")

    return success


async def test_both_tools():
    """Test both tools together."""
    print("\n" + "=" * 80)
    print("TEST 3: Both Google Search and URL Context")
    print("=" * 80)

    llm = LLMManager()

    prompt = """Using both https://www.certhub.de/en and web search,
provide a brief market analysis for CertHub (2-3 sentences)."""

    success, response = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_google_search=True,
        enable_url_context=True
    )

    print(f"\nSuccess: {success}")
    if success:
        # Check for grounding_enabled metadata
        if "grounding_enabled" in response:
            print(f"✓ grounding_enabled present: {response['grounding_enabled']}")
            if response['grounding_enabled']['google_search']:
                print("  ✓ google_search is True")
            if response['grounding_enabled']['url_context']:
                print("  ✓ url_context is True")
        else:
            print("✗ grounding_enabled missing")

        # Check for sources
        if "sources" in response:
            print(f"✓ sources present: {len(response['sources'])} sources")

            # Count by type
            url_context_sources = [s for s in response['sources'] if s.get('source_type') == 'url_context']
            web_sources = [s for s in response['sources'] if s.get('source_type') != 'url_context']

            print(f"  - URL context sources: {len(url_context_sources)}")
            print(f"  - Web search sources: {len(web_sources)}")

    else:
        print(f"✗ Error: {response}")

    return success


async def test_markdown_stripping():
    """Test that markdown code blocks are properly stripped."""
    print("\n" + "=" * 80)
    print("TEST 4: Markdown Stripping (JSON Parsing)")
    print("=" * 80)

    llm = LLMManager()

    # This prompt is likely to return markdown-wrapped JSON
    prompt = """Return this exact JSON: [{"test": "value", "number": 123}]"""

    success, response = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        return_json=True
    )

    print(f"\nSuccess: {success}")
    if success:
        print(f"Response type: {type(response)}")

        # Check if JSON was parsed
        if isinstance(response, dict) and "text" in response:
            try:
                data = json.loads(response["text"])
                print(f"✓ JSON parsing successful")
                print(f"  Parsed data: {data}")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")
        elif isinstance(response, list) or isinstance(response, dict):
            print(f"✓ Response is already parsed JSON")
            print(f"  Data: {response}")

    else:
        print(f"✗ Error: {response}")

    return success


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  TESTING WRAPPER FIXES")
    print("=" * 80)

    results = []

    # Run tests
    try:
        result1 = await test_google_search_grounding()
        results.append(("Google Search Grounding", result1))

        result2 = await test_url_context()
        results.append(("URL Context", result2))

        result3 = await test_both_tools()
        results.append(("Both Tools", result3))

        result4 = await test_markdown_stripping()
        results.append(("Markdown Stripping", result4))

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} {status}")
        if not result:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓".center(80))
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗".center(80))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
