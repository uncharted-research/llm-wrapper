#!/usr/bin/env python3
"""Test script to verify source extraction from Gemini responses."""

import asyncio
from llm_wrapper import LLMManager

async def test_source_extraction():
    """Test that all sources are properly extracted from Gemini responses."""

    # Initialize the LLM manager
    llm_manager = LLMManager()

    print("=" * 80)
    print("Testing Source Extraction from Gemini Responses")
    print("=" * 80)

    # Test with a real query that should return multiple sources
    prompt = """
    What is the Total Addressable Market (TAM) for uncharted group GmbH?
    First, research what uncharted group GmbH does, then find market size data.
    """

    print("\nSending query to Gemini with Google Search enabled...")
    print(f"Prompt: {prompt[:100]}...")

    success, response = await llm_manager.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_google_search=True
    )

    if success:
        print(f"\n✓ Success: {success}")
        print(f"Response type: {type(response)}")

        if isinstance(response, dict):
            print(f"\nKeys in response: {list(response.keys())}")

            # Print text preview
            if "text" in response:
                text = response["text"]
                print(f"\nText response length: {len(text)} characters")

                # Check for inline citations
                import re
                inline_citations = re.findall(r'\[cite:(\d+),\s*\d+~[^\]]+\]', text)
                print(f"Inline citations found: {len(inline_citations)} citations")
                if inline_citations:
                    print(f"Citation numbers: {sorted(set(inline_citations))}")

                # Check for sources section in text
                if "**Sources:**" in text or "Sources:" in text:
                    print("✓ Sources section found in text")

                    # Extract just the sources section
                    sources_section = text.split("Sources:")[-1] if "Sources:" in text else text.split("**Sources:**")[-1]
                    source_lines = [line.strip() for line in sources_section.strip().split('\n') if line.strip() and line.strip()[0].isdigit()]
                    print(f"Number of source lines in text: {len(source_lines)}")

            # Print extracted sources
            if "sources" in response:
                sources = response["sources"]
                print(f"\n✓ Extracted sources: {len(sources)} sources")

                print("\nExtracted Sources:")
                for i, source in enumerate(sources, 1):
                    if "number" in source:
                        print(f"  {source.get('number', i)}. {source.get('title', 'No title')} - {source.get('url', 'No URL')[:80]}...")
                    else:
                        print(f"  {i}. {source.get('title', 'No title')} - {source.get('url', 'No URL')[:80]}...")

                # Verify all sources have URLs
                sources_with_urls = [s for s in sources if 'url' in s and s['url']]
                print(f"\nSources with valid URLs: {len(sources_with_urls)}/{len(sources)}")

                # Check for different types of URLs
                vertexai_urls = [s for s in sources if 'url' in s and 'vertexaisearch.cloud.google.com' in s['url']]
                direct_urls = [s for s in sources if 'url' in s and 'vertexaisearch.cloud.google.com' not in s['url']]

                print(f"Vertex AI redirect URLs: {len(vertexai_urls)}")
                print(f"Direct URLs: {len(direct_urls)}")
            else:
                print("\n✗ No sources found in response")

            # Print search queries if available
            if "search_queries" in response:
                queries = response["search_queries"]
                print(f"\nSearch queries used: {len(queries)} queries")
                for q in queries:
                    print(f"  - {q}")

            # Check if grounding metadata exists
            if "grounding_metadata" in response and response["grounding_metadata"]:
                print("\n✓ Grounding metadata present")
            else:
                print("\n✗ No grounding metadata")

        else:
            print(f"\nUnexpected response type: {response}")
    else:
        print(f"\n✗ Error: {response}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_source_extraction())