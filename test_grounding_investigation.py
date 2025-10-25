"""
Diagnostic test to investigate Gemini response structure with Google Search grounding.

This test will:
1. Call Gemini with Google Search enabled using the provided market research prompt
2. Inspect the raw response structure in detail
3. Test different text extraction methods
4. Check for duplication patterns
5. Validate JSON parsing
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the llm_wrapper to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_wrapper import LLMManager

# The exact prompt provided by the user
MARKET_RESEARCH_PROMPT = """Use web search to investigate the market size of the company certhub.
Business description:
The company develops, operates, and distributes software applications and services in the field of information technology, digital quality management, and the creation of technical documentation. It provides a digital platform to support and accelerate certification, especially for medical products. The company's platform offers artificial intelligence-powered tools for quality management systems, technical documentation, and conformity checks. It enables healthcare businesses to reduce manual work, improve accuracy, and achieve certifications. The company's software simplifies medical device regulatory compliance through automation and AI.


This is the company's website:
https://www.certhub.de/en

Search at least 5-7 diverse sources including industry reports, market research firms, analyst reports, and company financial filings. Ensure to investigate various sources. It is likely to find different, or even contradicting information.

Return the information that you find in the following format:

[
    {
        "market_size": [NUMBER, "CURRENCY", "MULTIPLIER"],
        "year": YEAR,
        "market_description": "MARKET_DESCRIPTION",
        "geography": "GEOGRAPHY",
        "fit": "EXACT|BROADER|NARROWER",
        "confidence": "HIGH|MEDIUM|LOW",
        "source": "SOURCE_URL",
        "context": "VERBATIM_QUOTE_FROM_SOURCE"
    },
    {
        "market_size": [NUMBER, "CURRENCY", "MULTIPLIER"],
        "year": YEAR,
        "market_description": "MARKET_DESCRIPTION",
        "geography": "GEOGRAPHY",
        "fit": "EXACT|BROADER|NARROWER",
        "confidence": "HIGH|MEDIUM|LOW",
        "source": "SOURCE_URL",
        "context": "VERBATIM_QUOTE_FROM_SOURCE"
    },
    ...
]

Explanations:
- market_size: the market size you found in a specific source
    - NUMBER: the numeric value only (e.g., 5.2)
    - CURRENCY: the currency code (e.g., "USD", "EUR")
    - MULTIPLIER: "K", "M", "B", "T" for thousands, millions, billions, trillions
- year: The year for which the market size was reported (integer)
- market_description: Description of the market to which the market size refers
- geography: The geography of the market (e.g., "Global", "North America", "USA")
- fit: Either "EXACT", "BROADER", or "NARROWER"
    - EXACT: market matches the target company exactly
    - BROADER: market is broader than what the company targets
    - NARROWER: market is narrower than what the company targets
- confidence: Your confidence in the data quality and source reliability ("HIGH", "MEDIUM", "LOW")
- source: the URL of the source
- context: a verbatim quote from the source that contains the market size

IMPORTANT: Return ONLY valid JSON. Do not wrap the response in markdown code blocks (no ```json or ```).
Return the raw JSON array directly."""


def print_separator(title):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def inspect_response_object(response):
    """Deeply inspect the response object structure."""
    print_separator("RAW RESPONSE OBJECT INSPECTION")

    # Check response-level attributes
    print("Response-level attributes:")
    print(f"  - Has 'text' attribute: {hasattr(response, 'text')}")
    if hasattr(response, 'text'):
        print(f"  - response.text type: {type(response.text)}")
        print(f"  - response.text length: {len(response.text) if response.text else 0}")
        if response.text:
            print(f"  - response.text preview (first 200 chars):")
            print(f"    {response.text[:200]}")

    print(f"\n  - Number of candidates: {len(response.candidates) if response.candidates else 0}")

    if not response.candidates or len(response.candidates) == 0:
        print("  ⚠️  No candidates in response!")
        return

    # Inspect first candidate
    candidate = response.candidates[0]
    print("\nCandidate[0] attributes:")
    print(f"  - Has 'text' attribute: {hasattr(candidate, 'text')}")
    if hasattr(candidate, 'text'):
        try:
            text = candidate.text
            print(f"  - candidate.text type: {type(text)}")
            print(f"  - candidate.text length: {len(text) if text else 0}")
        except Exception as e:
            print(f"  - Error accessing candidate.text: {e}")

    print(f"  - Has 'content' attribute: {hasattr(candidate, 'content')}")
    print(f"  - Has 'grounding_metadata' attribute: {hasattr(candidate, 'grounding_metadata')}")
    print(f"  - Has 'grounding_attributions' attribute: {hasattr(candidate, 'grounding_attributions')}")
    print(f"  - Has 'url_context_metadata' attribute: {hasattr(candidate, 'url_context_metadata')}")

    # Inspect content parts
    if hasattr(candidate, 'content') and candidate.content:
        content = candidate.content
        print(f"\nCandidate.content attributes:")
        print(f"  - Has 'parts' attribute: {hasattr(content, 'parts')}")

        if hasattr(content, 'parts') and content.parts:
            parts = content.parts
            print(f"  - Number of parts: {len(parts)}")

            for i, part in enumerate(parts):
                print(f"\n  Part [{i}]:")
                print(f"    - Has 'text' attribute: {hasattr(part, 'text')}")

                if hasattr(part, 'text') and part.text:
                    print(f"    - Text length: {len(part.text)}")
                    print(f"    - Text preview (first 200 chars):")
                    print(f"      {part.text[:200]}")
                    print(f"    - Text preview (last 200 chars):")
                    print(f"      ...{part.text[-200:]}")
                else:
                    print(f"    - No text in this part")

                # Check for other attributes
                print(f"    - Has 'inline_data': {hasattr(part, 'inline_data')}")
                print(f"    - Has 'function_call': {hasattr(part, 'function_call')}")
                print(f"    - Has 'function_response': {hasattr(part, 'function_response')}")

    # Inspect grounding metadata
    print_separator("GROUNDING METADATA INSPECTION")

    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
        gm = candidate.grounding_metadata
        print("Grounding metadata found!")
        print(f"  - Type: {type(gm)}")
        print(f"  - Has 'grounding_chunks': {hasattr(gm, 'grounding_chunks')}")
        print(f"  - Has 'grounding_supports': {hasattr(gm, 'grounding_supports')}")
        print(f"  - Has 'web_search_queries': {hasattr(gm, 'web_search_queries')}")
        print(f"  - Has 'retrieval_queries': {hasattr(gm, 'retrieval_queries')}")
        print(f"  - Has 'search_entry_point': {hasattr(gm, 'search_entry_point')}")

        if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
            print(f"\n  Web search queries ({len(gm.web_search_queries)}):")
            for i, query in enumerate(gm.web_search_queries):
                print(f"    [{i}] {query}")

        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            print(f"\n  Grounding chunks ({len(gm.grounding_chunks)}):")
            for i, chunk in enumerate(gm.grounding_chunks[:3]):  # Show first 3
                print(f"    [{i}] Has 'web': {hasattr(chunk, 'web')}")
                if hasattr(chunk, 'web') and chunk.web:
                    print(f"        URI: {chunk.web.uri if hasattr(chunk.web, 'uri') else 'N/A'}")
                    print(f"        Title: {chunk.web.title if hasattr(chunk.web, 'title') else 'N/A'}")

        if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
            print(f"\n  Grounding supports: {len(gm.grounding_supports)} items")
    else:
        print("⚠️  No grounding_metadata found!")

    # Check for grounding_attributions (old/wrong field name)
    if hasattr(candidate, 'grounding_attributions') and candidate.grounding_attributions:
        print("\n⚠️  grounding_attributions field exists (this might be old API)")
        print(f"    Length: {len(candidate.grounding_attributions)}")

    # Check for url_context_metadata
    print_separator("URL CONTEXT METADATA INSPECTION")

    if hasattr(candidate, 'url_context_metadata') and candidate.url_context_metadata:
        print("URL context metadata found!")
        print(f"  - Type: {type(candidate.url_context_metadata)}")
    else:
        print("No url_context_metadata found (expected since we didn't enable it)")


def test_extraction_methods(response):
    """Test different methods of extracting text from the response."""
    print_separator("TESTING DIFFERENT EXTRACTION METHODS")

    methods = {}

    # Method 1: Using response.text
    if hasattr(response, 'text') and response.text:
        methods['response.text'] = response.text
        print(f"✓ Method 1: response.text - {len(response.text)} chars")
    else:
        print(f"✗ Method 1: response.text - Not available")

    # Method 2: Using candidate.text
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if hasattr(candidate, 'text'):
            try:
                text = candidate.text
                methods['candidate.text'] = text
                print(f"✓ Method 2: candidate.text - {len(text) if text else 0} chars")
            except Exception as e:
                print(f"✗ Method 2: candidate.text - Error: {e}")
        else:
            print(f"✗ Method 2: candidate.text - Not available")

        # Method 3: Manual concatenation with \n.join (current implementation)
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                concatenated = "\n".join(text_parts)
                methods['manual_newline_join'] = concatenated
                print(f"✓ Method 3: Manual \\n.join - {len(concatenated)} chars ({len(text_parts)} parts)")
            else:
                print(f"✗ Method 3: Manual \\n.join - No text parts")

        # Method 4: Taking only the last part (proposed fix)
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                last_part = text_parts[-1]
                methods['last_part_only'] = last_part
                print(f"✓ Method 4: Last part only - {len(last_part)} chars")
            else:
                print(f"✗ Method 4: Last part only - No text parts")

        # Method 5: Concatenate without separator (for comparison)
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                concatenated = "".join(text_parts)
                methods['manual_direct_join'] = concatenated
                print(f"✓ Method 5: Direct join (no separator) - {len(concatenated)} chars")
            else:
                print(f"✗ Method 5: Direct join - No text parts")

    return methods


def check_for_duplication(methods):
    """Check if extracted text contains duplication."""
    print_separator("DUPLICATION ANALYSIS")

    for method_name, text in methods.items():
        if not text:
            continue

        print(f"\nMethod: {method_name}")
        print(f"  Total length: {len(text)} chars")

        # Check if first half appears again in second half
        midpoint = len(text) // 2
        first_half = text[:midpoint]
        second_half = text[midpoint:]

        # Look for substantial overlap (more than 100 chars)
        overlap_threshold = 100
        if len(first_half) > overlap_threshold:
            # Check if beginning appears later
            beginning = first_half[:overlap_threshold]
            if beginning in text[overlap_threshold:]:
                positions = []
                start = 0
                while True:
                    pos = text.find(beginning, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1

                if len(positions) > 1:
                    print(f"  ⚠️  DUPLICATION DETECTED!")
                    print(f"     First {overlap_threshold} chars appear at positions: {positions}")
                else:
                    print(f"  ✓ No obvious duplication detected")
            else:
                print(f"  ✓ No obvious duplication detected")

        # Check for JSON duplication specifically
        # Count occurrences of opening brackets
        bracket_count = text.count('[{')
        if bracket_count > 1:
            print(f"  ⚠️  Multiple '[{{' patterns found: {bracket_count} times")
            print(f"     This might indicate duplicated JSON")


def validate_json(methods):
    """Try to parse JSON from each extraction method."""
    print_separator("JSON VALIDATION")

    for method_name, text in methods.items():
        if not text:
            continue

        print(f"\nMethod: {method_name}")

        # Try direct parsing
        try:
            data = json.loads(text)
            print(f"  ✓ Valid JSON - parsed successfully")
            print(f"    Type: {type(data)}")
            if isinstance(data, list):
                print(f"    Array length: {len(data)}")
            elif isinstance(data, dict):
                print(f"    Object keys: {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"  ✗ Invalid JSON - Error: {e}")
            print(f"    Error at position: {e.pos}")

            # Show context around error
            if e.pos:
                start = max(0, e.pos - 100)
                end = min(len(text), e.pos + 100)
                print(f"    Context around error:")
                print(f"    ...{text[start:end]}...")

            # Try to find where JSON starts and ends
            first_bracket = text.find('[')
            last_bracket = text.rfind(']')

            if first_bracket != -1 and last_bracket != -1:
                print(f"\n    First '[' at position: {first_bracket}")
                print(f"    Last ']' at position: {last_bracket}")

                # Check if there are multiple JSON arrays
                all_brackets = []
                for i, char in enumerate(text):
                    if char == '[':
                        all_brackets.append(('open', i))
                    elif char == ']':
                        all_brackets.append(('close', i))

                print(f"    Total '[' count: {text.count('[')}")
                print(f"    Total ']' count: {text.count(']')}")

                # Try extracting JSON between first and last bracket
                potential_json = text[first_bracket:last_bracket+1]
                print(f"\n    Trying to parse extracted JSON ({len(potential_json)} chars)...")
                try:
                    data = json.loads(potential_json)
                    print(f"    ✓ Extracted JSON is valid!")
                    if isinstance(data, list):
                        print(f"      Array length: {len(data)}")
                except json.JSONDecodeError as e2:
                    print(f"    ✗ Extracted JSON also invalid: {e2}")


async def main():
    """Main test function."""
    print("=" * 80)
    print("  GEMINI GROUNDING RESPONSE INVESTIGATION")
    print("  Testing with market research prompt for certhub")
    print("=" * 80)

    # Initialize LLM manager
    llm = LLMManager()

    print("\nCalling Gemini with Google Search enabled...")
    print("Model: gemini-2.5-flash")
    print(f"Prompt length: {len(MARKET_RESEARCH_PROMPT)} chars\n")

    # Make the actual API call using the internal method to get raw response
    # We'll need to call the Gemini API directly to inspect the raw response
    print("⚠️  Making API call (this may take 30-60 seconds with grounding)...\n")

    try:
        # Call the internal method but capture the raw response first
        import google.genai as genai
        from google.genai import types

        # Build configuration
        config_dict = {
            'thinking_config': types.ThinkingConfig(thinking_budget=-1),
            'tools': [types.Tool(googleSearch=types.GoogleSearch())]
        }
        config = types.GenerateContentConfig(**config_dict)

        # Format contents
        contents = [types.Content(
            role="user",
            parts=[types.Part.from_text(text=MARKET_RESEARCH_PROMPT)]
        )]

        # Make the call
        response = llm.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )

        print("✓ API call completed successfully!\n")

        # Now do all the inspections
        inspect_response_object(response)

        methods = test_extraction_methods(response)

        check_for_duplication(methods)

        validate_json(methods)

        # Save the raw response for later inspection
        print_separator("SAVING RESULTS")

        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)

        # Save extracted texts
        for method_name, text in methods.items():
            if text:
                filename = output_dir / f"extracted_{method_name}.txt"
                filename.write_text(text)
                print(f"Saved: {filename}")

        print("\n✓ Investigation complete! Check test_results/ directory for outputs.")

    except Exception as e:
        print(f"\n✗ Error during API call: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
