"""
URL Context vs Google Search Grounding separation test.

This test will verify that:
1. enable_url_context and enable_google_search are truly separate features
2. They populate different metadata fields
3. They can be used independently or together
4. The current code correctly distinguishes between them
"""

import asyncio
import sys
from pathlib import Path

# Add the llm_wrapper to path
sys.path.insert(0, str(Path(__file__).parent))

import google.genai as genai
from google.genai import types
from llm_wrapper import LLMManager


# Prompts for different scenarios
URL_CONTEXT_PROMPT = """Based on the content from https://www.certhub.de/en, summarize what the company does in 2-3 sentences.

Return only a brief text summary, no JSON needed."""

GOOGLE_SEARCH_PROMPT = """Search for information about CertHub company and their market.

Return a brief summary of what you find (2-3 sentences)."""

BOTH_TOOLS_PROMPT = """Using both the company website https://www.certhub.de/en and web search,
provide a comprehensive summary of CertHub and their market position (3-4 sentences)."""


def print_separator(title):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def analyze_metadata(response, test_name):
    """Analyze what metadata fields are populated."""
    print_separator(f"METADATA ANALYSIS: {test_name}")

    if not response.candidates or len(response.candidates) == 0:
        print("⚠️  No candidates!")
        return {}

    candidate = response.candidates[0]

    metadata = {
        'test_name': test_name,
        'has_grounding_metadata': False,
        'has_url_context_metadata': False,
        'grounding_details': {},
        'url_context_details': {}
    }

    # Check for grounding_metadata
    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
        metadata['has_grounding_metadata'] = True
        gm = candidate.grounding_metadata

        print("✓ GROUNDING METADATA found:")

        if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
            metadata['grounding_details']['web_search_queries'] = list(gm.web_search_queries)
            print(f"  - Web search queries: {len(gm.web_search_queries)}")
            for query in gm.web_search_queries:
                print(f"    • {query}")

        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            metadata['grounding_details']['num_chunks'] = len(gm.grounding_chunks)
            print(f"  - Grounding chunks: {len(gm.grounding_chunks)}")

            # Show first few sources
            for i, chunk in enumerate(gm.grounding_chunks[:3]):
                if hasattr(chunk, 'web') and chunk.web:
                    if hasattr(chunk.web, 'uri'):
                        print(f"    [{i}] {chunk.web.uri}")

        if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
            metadata['grounding_details']['num_supports'] = len(gm.grounding_supports)
            print(f"  - Grounding supports: {len(gm.grounding_supports)}")

        if hasattr(gm, 'search_entry_point') and gm.search_entry_point:
            print(f"  - Search entry point: Present")

        if hasattr(gm, 'retrieval_metadata') and gm.retrieval_metadata:
            print(f"  - Retrieval metadata: Present")
            metadata['grounding_details']['has_retrieval_metadata'] = True
    else:
        print("✗ No grounding_metadata")

    # Check for url_context_metadata
    if hasattr(candidate, 'url_context_metadata') and candidate.url_context_metadata:
        metadata['has_url_context_metadata'] = True
        ucm = candidate.url_context_metadata

        print("\n✓ URL CONTEXT METADATA found:")
        print(f"  - Type: {type(ucm)}")

        # Try to inspect the structure
        if hasattr(ucm, 'references') and ucm.references:
            metadata['url_context_details']['num_references'] = len(ucm.references)
            print(f"  - References: {len(ucm.references)}")

            for i, ref in enumerate(ucm.references[:3]):
                print(f"    [{i}] {ref}")
        else:
            # Just note that it exists
            print(f"  - Metadata present but structure unclear")
            # Try to convert to dict if possible
            try:
                if hasattr(ucm, 'to_json_dict'):
                    ucm_dict = ucm.to_json_dict()
                    metadata['url_context_details']['raw'] = str(ucm_dict)
                    print(f"  - Raw data: {ucm_dict}")
            except:
                pass
    else:
        print("\n✗ No url_context_metadata")

    # Additional checks
    print(f"\nResponse has {len(candidate.content.parts) if hasattr(candidate, 'content') else 0} content parts")

    return metadata


async def test_url_context_only():
    """Test with ONLY url_context enabled."""
    print_separator("TEST 1: URL CONTEXT ONLY")

    llm = LLMManager()

    config_dict = {
        'thinking_config': types.ThinkingConfig(thinking_budget=-1),
        'tools': [types.Tool(url_context=types.UrlContext())]
    }
    config = types.GenerateContentConfig(**config_dict)

    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=URL_CONTEXT_PROMPT)]
    )]

    print("Calling Gemini with ONLY url_context enabled...")
    response = llm.gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config
    )

    return response


async def test_google_search_only():
    """Test with ONLY google_search enabled."""
    print_separator("TEST 2: GOOGLE SEARCH ONLY")

    llm = LLMManager()

    config_dict = {
        'thinking_config': types.ThinkingConfig(thinking_budget=-1),
        'tools': [types.Tool(googleSearch=types.GoogleSearch())]
    }
    config = types.GenerateContentConfig(**config_dict)

    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=GOOGLE_SEARCH_PROMPT)]
    )]

    print("Calling Gemini with ONLY googleSearch enabled...")
    response = llm.gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config
    )

    return response


async def test_both_tools():
    """Test with BOTH url_context and google_search enabled."""
    print_separator("TEST 3: BOTH TOOLS ENABLED")

    llm = LLMManager()

    config_dict = {
        'thinking_config': types.ThinkingConfig(thinking_budget=-1),
        'tools': [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(googleSearch=types.GoogleSearch())
        ]
    }
    config = types.GenerateContentConfig(**config_dict)

    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=BOTH_TOOLS_PROMPT)]
    )]

    print("Calling Gemini with BOTH url_context AND googleSearch enabled...")
    response = llm.gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config
    )

    return response


def compare_metadata(results):
    """Compare metadata across all three tests."""
    print_separator("CROSS-TEST COMPARISON")

    print("Metadata field comparison:")
    print(f"{'Test':<30} {'Grounding Metadata':<25} {'URL Context Metadata':<25}")
    print("-" * 80)

    for result in results:
        test_name = result['test_name']
        has_grounding = "✓ Present" if result['has_grounding_metadata'] else "✗ Absent"
        has_url_context = "✓ Present" if result['has_url_context_metadata'] else "✗ Absent"

        print(f"{test_name:<30} {has_grounding:<25} {has_url_context:<25}")

    # Expected behavior
    print("\n" + "=" * 80)
    print("EXPECTED BEHAVIOR:")
    print("=" * 80)
    print("Test 1 (URL Context only):  grounding_metadata=✗  url_context_metadata=✓")
    print("Test 2 (Google Search only): grounding_metadata=✓  url_context_metadata=✗")
    print("Test 3 (Both tools):         grounding_metadata=✓  url_context_metadata=✓")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION:")
    print("=" * 80)

    url_context_result = results[0]
    google_search_result = results[1]
    both_result = results[2]

    all_correct = True

    # Check Test 1
    if url_context_result['has_url_context_metadata'] and not url_context_result['has_grounding_metadata']:
        print("✓ Test 1 CORRECT: URL context only populated url_context_metadata")
    else:
        print(f"✗ Test 1 INCORRECT:")
        print(f"   - url_context_metadata: {url_context_result['has_url_context_metadata']} (expected: True)")
        print(f"   - grounding_metadata: {url_context_result['has_grounding_metadata']} (expected: False)")
        all_correct = False

    # Check Test 2
    if google_search_result['has_grounding_metadata'] and not google_search_result['has_url_context_metadata']:
        print("✓ Test 2 CORRECT: Google Search only populated grounding_metadata")
    else:
        print(f"✗ Test 2 INCORRECT:")
        print(f"   - grounding_metadata: {google_search_result['has_grounding_metadata']} (expected: True)")
        print(f"   - url_context_metadata: {google_search_result['has_url_context_metadata']} (expected: False)")
        all_correct = False

    # Check Test 3
    if both_result['has_grounding_metadata'] and both_result['has_url_context_metadata']:
        print("✓ Test 3 CORRECT: Both tools populated both metadata fields")
    else:
        print(f"✗ Test 3 INCORRECT:")
        print(f"   - grounding_metadata: {both_result['has_grounding_metadata']} (expected: True)")
        print(f"   - url_context_metadata: {both_result['has_url_context_metadata']} (expected: True)")
        all_correct = False

    print("\n" + "=" * 80)
    if all_correct:
        print("✓✓✓ ALL TESTS PASSED - Tools are properly separated!")
    else:
        print("✗✗✗ SOME TESTS FAILED - Check the validation details above")
    print("=" * 80)


async def main():
    """Main test function."""
    print("=" * 80)
    print("  URL CONTEXT vs GOOGLE SEARCH SEPARATION TEST")
    print("  Verifying that these are truly independent features")
    print("=" * 80)

    try:
        # Run all three tests
        print("\n⏳ Test 1: URL Context only...")
        response1 = await test_url_context_only()
        result1 = analyze_metadata(response1, "URL Context Only")

        print("\n⏳ Test 2: Google Search only...")
        response2 = await test_google_search_only()
        result2 = analyze_metadata(response2, "Google Search Only")

        print("\n⏳ Test 3: Both tools...")
        response3 = await test_both_tools()
        result3 = analyze_metadata(response3, "Both Tools")

        # Compare results
        compare_metadata([result1, result2, result3])

        # Additional analysis of current wrapper code
        print_separator("CURRENT WRAPPER CODE ANALYSIS")

        print("Checking how the wrapper currently handles these tools...")
        print("\nIn llm_wrapper/llm.py:")

        # Check lines around 330, 458, 746 where grounding is checked
        print("\n⚠️  POTENTIAL ISSUE:")
        print("   The wrapper currently uses:")
        print("   `if enable_google_search or enable_url_context:`")
        print("   to determine whether to include sources.")
        print()
        print("   This INCORRECTLY treats both tools the same way!")
        print()
        print("   CORRECT behavior:")
        print("   - enable_google_search → extract from grounding_metadata")
        print("   - enable_url_context → extract from url_context_metadata")
        print("   - They should be handled SEPARATELY")

        print("\n✓ Test complete!")

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
