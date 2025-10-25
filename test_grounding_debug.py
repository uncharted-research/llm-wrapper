#!/usr/bin/env python3
"""Debug script to inspect Gemini grounding response structure."""

import asyncio
import pprint
from llm_wrapper import LLMManager
from google import genai
import json

async def debug_grounding():
    """Debug Gemini grounding response structure."""

    # Initialize the LLM manager
    llm_manager = LLMManager()

    print("=" * 80)
    print("Debugging Gemini Grounding Response Structure")
    print("=" * 80)

    # Test with Google Search
    prompt = "What is the current weather in San Francisco? Please search for current information."

    # Call Gemini directly to inspect raw response
    config_dict = {
        'tools': [genai.types.Tool(googleSearch=genai.types.GoogleSearch())]
    }
    config = genai.types.GenerateContentConfig(**config_dict)

    contents = [genai.types.Content(
        role="user",
        parts=[genai.types.Part.from_text(text=prompt)]
    )]

    try:
        response = llm_manager.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )

        print("\n1. Response Object Type:")
        print(f"   {type(response)}")

        print("\n2. Response Attributes:")
        attrs = [attr for attr in dir(response) if not attr.startswith('_')]
        for attr in attrs:
            print(f"   - {attr}")

        print("\n3. Candidates Structure:")
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            print(f"   Number of candidates: {len(response.candidates)}")
            print(f"   Candidate type: {type(candidate)}")

            print("\n4. Candidate Attributes:")
            candidate_attrs = [attr for attr in dir(candidate) if not attr.startswith('_')]
            for attr in candidate_attrs:
                print(f"   - {attr}")

            print("\n5. Content Structure:")
            if hasattr(candidate, 'content'):
                print(f"   Content type: {type(candidate.content)}")
                if hasattr(candidate.content, 'parts'):
                    print(f"   Number of parts: {len(candidate.content.parts)}")
                    for i, part in enumerate(candidate.content.parts):
                        print(f"   Part {i} type: {type(part)}")
                        if hasattr(part, 'text'):
                            print(f"   Part {i} has text: {part.text[:100] if part.text else None}...")

            print("\n6. Grounding Attributions:")
            if hasattr(candidate, 'grounding_attributions'):
                print(f"   Has grounding_attributions: {hasattr(candidate, 'grounding_attributions')}")
                print(f"   grounding_attributions value: {candidate.grounding_attributions}")
                if candidate.grounding_attributions:
                    print(f"   Type: {type(candidate.grounding_attributions)}")
                    print(f"   Length: {len(candidate.grounding_attributions)}")
                    for i, attr in enumerate(candidate.grounding_attributions[:2]):
                        print(f"   Attribution {i}: {attr}")

            print("\n7. Grounding Metadata:")
            if hasattr(candidate, 'grounding_metadata'):
                print(f"   Has grounding_metadata: {hasattr(candidate, 'grounding_metadata')}")
                print(f"   grounding_metadata value: {candidate.grounding_metadata}")
                if candidate.grounding_metadata:
                    print(f"   Type: {type(candidate.grounding_metadata)}")
                    metadata_attrs = [attr for attr in dir(candidate.grounding_metadata) if not attr.startswith('_')]
                    for attr in metadata_attrs:
                        print(f"   - {attr}: {getattr(candidate.grounding_metadata, attr, None)}")

            print("\n8. Citation Metadata:")
            if hasattr(candidate, 'citation_metadata'):
                print(f"   Has citation_metadata: {hasattr(candidate, 'citation_metadata')}")
                print(f"   citation_metadata value: {candidate.citation_metadata}")
                if candidate.citation_metadata:
                    print(f"   Type: {type(candidate.citation_metadata)}")
                    if hasattr(candidate.citation_metadata, 'citations'):
                        print(f"   Number of citations: {len(candidate.citation_metadata.citations)}")
                        for i, citation in enumerate(candidate.citation_metadata.citations[:3]):
                            print(f"   Citation {i}:")
                            citation_attrs = [attr for attr in dir(citation) if not attr.startswith('_')]
                            for attr in citation_attrs:
                                val = getattr(citation, attr, None)
                                if not callable(val):
                                    print(f"      - {attr}: {val}")

            print("\n9. Full Response Object (pretty printed):")
            # Try to convert to dict if possible
            try:
                if hasattr(response, 'to_dict'):
                    response_dict = response.to_dict()
                    print(json.dumps(response_dict, indent=2, default=str))
                elif hasattr(response, '__dict__'):
                    print(json.dumps(response.__dict__, indent=2, default=str))
                else:
                    print(f"Cannot convert to dict. Response: {response}")
            except Exception as e:
                print(f"Error converting to dict: {e}")
                print(f"Response string: {str(response)[:500]}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(debug_grounding())