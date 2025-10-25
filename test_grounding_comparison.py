"""
Comparison test: Gemini responses WITH and WITHOUT Google Search grounding.

This test will:
1. Call Gemini with the same prompt twice:
   - Once WITH enable_google_search=True
   - Once WITHOUT enable_google_search=False
2. Compare the response structures
3. Identify differences in:
   - Number of parts
   - Content structure
   - Metadata fields
   - Text content
"""

import asyncio
import sys
from pathlib import Path

# Add the llm_wrapper to path
sys.path.insert(0, str(Path(__file__).parent))

import google.genai as genai
from google.genai import types
from llm_wrapper import LLMManager


# Simplified prompt for faster testing
SIMPLE_PROMPT = """Search for information about the medical device regulatory compliance software market size.

Return a JSON array with at least 3 market size data points in this format:
[
    {
        "market_size": [50, "USD", "M"],
        "year": 2024,
        "market_description": "Example market",
        "geography": "Global",
        "fit": "BROADER",
        "confidence": "MEDIUM",
        "source": "https://example.com",
        "context": "Quote from source"
    }
]

Return ONLY valid JSON (no markdown)."""


def print_separator(title):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def analyze_response_structure(response, label):
    """Analyze and print response structure."""
    print_separator(f"RESPONSE STRUCTURE: {label}")

    print(f"Response type: {type(response)}")
    print(f"Has candidates: {hasattr(response, 'candidates')}")

    if not response.candidates or len(response.candidates) == 0:
        print("⚠️  No candidates!")
        return {}

    candidate = response.candidates[0]

    structure = {
        'has_response_text': hasattr(response, 'text'),
        'has_candidate_text': hasattr(candidate, 'text'),
        'has_content': hasattr(candidate, 'content'),
        'has_grounding_metadata': hasattr(candidate, 'grounding_metadata'),
        'has_grounding_attributions': hasattr(candidate, 'grounding_attributions'),
        'has_url_context_metadata': hasattr(candidate, 'url_context_metadata'),
        'num_parts': 0,
        'part_lengths': []
    }

    # Check response.text
    if hasattr(response, 'text') and response.text:
        structure['response_text_length'] = len(response.text)
        print(f"response.text: {len(response.text)} chars")
    else:
        print("response.text: Not available or empty")

    # Check candidate attributes
    print(f"\nCandidate attributes:")
    print(f"  - has_text: {structure['has_candidate_text']}")
    print(f"  - has_content: {structure['has_content']}")
    print(f"  - has_grounding_metadata: {structure['has_grounding_metadata']}")
    print(f"  - has_grounding_attributions: {structure['has_grounding_attributions']}")
    print(f"  - has_url_context_metadata: {structure['has_url_context_metadata']}")

    # Analyze parts
    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
        parts = candidate.content.parts
        structure['num_parts'] = len(parts)
        print(f"\nContent parts: {len(parts)}")

        for i, part in enumerate(parts):
            if hasattr(part, 'text') and part.text:
                length = len(part.text)
                structure['part_lengths'].append(length)
                print(f"  Part [{i}]: {length} chars")
                print(f"    Preview (first 100): {part.text[:100]}")
                print(f"    Preview (last 100): ...{part.text[-100:]}")
            else:
                structure['part_lengths'].append(0)
                print(f"  Part [{i}]: No text")

    # Analyze grounding metadata
    if structure['has_grounding_metadata'] and candidate.grounding_metadata:
        gm = candidate.grounding_metadata
        print(f"\nGrounding metadata:")

        if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
            print(f"  - Web search queries: {len(gm.web_search_queries)}")
            for query in gm.web_search_queries:
                print(f"    • {query}")

        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            print(f"  - Grounding chunks: {len(gm.grounding_chunks)}")
            structure['num_grounding_chunks'] = len(gm.grounding_chunks)

        if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
            print(f"  - Grounding supports: {len(gm.grounding_supports)}")
            structure['num_grounding_supports'] = len(gm.grounding_supports)

    return structure


async def test_with_grounding():
    """Call Gemini WITH Google Search grounding."""
    llm = LLMManager()

    print_separator("TEST 1: WITH GOOGLE SEARCH GROUNDING")
    print("Calling Gemini with enable_google_search=True...")

    config_dict = {
        'thinking_config': types.ThinkingConfig(thinking_budget=-1),
        'tools': [types.Tool(googleSearch=types.GoogleSearch())]
    }
    config = types.GenerateContentConfig(**config_dict)

    contents = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=SIMPLE_PROMPT)]
    )]

    response = llm.gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config
    )

    return response


async def test_without_grounding():
    """Call Gemini WITHOUT Google Search grounding."""
    llm = LLMManager()

    print_separator("TEST 2: WITHOUT GOOGLE SEARCH GROUNDING")
    print("Calling Gemini with no grounding tools...")

    config_dict = {
        'thinking_config': types.ThinkingConfig(thinking_budget=-1)
    }
    config = types.GenerateContentConfig(**config_dict)

    # Use simple string format when no tools
    contents = [SIMPLE_PROMPT]

    response = llm.gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config
    )

    return response


def compare_structures(with_grounding, without_grounding):
    """Compare the two response structures."""
    print_separator("COMPARISON SUMMARY")

    print("Attribute comparison:")
    print(f"{'Attribute':<30} {'With Grounding':<20} {'Without Grounding':<20} {'Match':<10}")
    print("-" * 80)

    all_keys = set(with_grounding.keys()) | set(without_grounding.keys())

    for key in sorted(all_keys):
        val_with = with_grounding.get(key, 'N/A')
        val_without = without_grounding.get(key, 'N/A')
        match = "✓" if val_with == val_without else "✗"

        print(f"{key:<30} {str(val_with):<20} {str(val_without):<20} {match:<10}")

    # Specific observations
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)

    if with_grounding.get('num_parts', 0) != without_grounding.get('num_parts', 0):
        print(f"\n⚠️  Number of parts DIFFERS:")
        print(f"   - With grounding: {with_grounding.get('num_parts', 0)} parts")
        print(f"   - Without grounding: {without_grounding.get('num_parts', 0)} parts")
        print(f"   → This could explain duplication if multiple parts are concatenated!")

    if with_grounding.get('has_grounding_metadata', False) and not without_grounding.get('has_grounding_metadata', False):
        print(f"\n✓ Grounding metadata present ONLY with grounding enabled (as expected)")

    if with_grounding.get('response_text_length', 0) > 0 and without_grounding.get('response_text_length', 0) > 0:
        ratio = with_grounding['response_text_length'] / without_grounding['response_text_length']
        print(f"\nText length ratio (with/without grounding): {ratio:.2f}x")
        if ratio > 1.8:
            print(f"   ⚠️  Response with grounding is significantly longer!")
            print(f"   This might indicate duplication or additional content.")


async def main():
    """Main test function."""
    print("=" * 80)
    print("  GROUNDING COMPARISON TEST")
    print("  Comparing Gemini responses with/without Google Search")
    print("=" * 80)

    try:
        # Run both tests
        print("\n⏳ Running test WITH grounding (may take 30-60 seconds)...")
        response_with = await test_with_grounding()
        structure_with = analyze_response_structure(response_with, "WITH GROUNDING")

        print("\n⏳ Running test WITHOUT grounding (should be faster)...")
        response_without = await test_without_grounding()
        structure_without = analyze_response_structure(response_without, "WITHOUT GROUNDING")

        # Compare
        compare_structures(structure_with, structure_without)

        # Save outputs
        print_separator("SAVING RESULTS")

        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)

        # Save text from both responses
        if hasattr(response_with, 'text') and response_with.text:
            (output_dir / "comparison_with_grounding.txt").write_text(response_with.text)
            print("Saved: comparison_with_grounding.txt")

        if hasattr(response_without, 'text') and response_without.text:
            (output_dir / "comparison_without_grounding.txt").write_text(response_without.text)
            print("Saved: comparison_without_grounding.txt")

        print("\n✓ Comparison test complete!")

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
