"""
Simple working example of using the LLM wrapper with Google Search grounding.

This shows the correct way to:
1. Make API calls with grounding
2. Parse JSON from responses
3. Access sources
"""

import asyncio
import json
from llm_wrapper import LLMManager


async def simple_example():
    """Simple example: Get market data with sources."""

    print("=" * 80)
    print("SIMPLE EXAMPLE: Market Research with Google Search Grounding")
    print("=" * 80)

    llm = LLMManager()

    prompt = """Search for information about medical device compliance software market size.

Return 3-5 data points as a JSON array:
[
    {
        "market_size": [1.5, "USD", "B"],
        "year": 2024,
        "market_description": "Medical Device QMS Software Market",
        "geography": "Global"
    },
    ...
]

Return ONLY valid JSON, no markdown code blocks."""

    print("\n1. Making API call...")
    success, response = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_google_search=True,
        return_json=True,
        max_tokens=8192  # Prevent truncation
    )

    if not success:
        print(f"❌ Error: {response}")
        return

    print("✅ API call successful\n")

    # 2. Parse JSON from response
    print("2. Parsing JSON...")
    try:
        # Check if JSON is already wrapped (JSON array case)
        if 'data' in response:
            market_data = response['data']
            print(f"✅ Got {len(market_data)} market data points (pre-parsed)\n")
        # Otherwise parse from 'text' field
        elif 'text' in response:
            market_data = json.loads(response['text'])
            print(f"✅ Parsed {len(market_data)} market data points\n")
        else:
            # Response is the data itself
            market_data = response if isinstance(response, list) else [response]
            print(f"✅ Got {len(market_data)} market data points\n")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Parsing failed: {e}")
        print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        return

    # 3. Access metadata
    print("3. Accessing metadata...")
    sources = response.get('sources', [])
    search_queries = response.get('search_queries', [])
    grounding_enabled = response.get('grounding_enabled', {})

    print(f"   - Sources: {len(sources)}")
    print(f"   - Search queries: {len(search_queries)}")
    print(f"   - Google Search: {grounding_enabled.get('google_search')}")
    print(f"   - URL Context: {grounding_enabled.get('url_context')}\n")

    # 4. Display results
    print("4. Market Data:")
    print("-" * 80)
    for item in market_data:
        size = item.get('market_size', [0, 'USD', 'B'])
        year = item.get('year', 'N/A')
        desc = item.get('market_description', 'Unknown')

        print(f"\n📊 {year}: {size[0]} {size[2]} {size[1]}")
        print(f"   {desc}")

    print("\n" + "-" * 80)
    print("\n5. Sources:")
    print("-" * 80)
    for i, source in enumerate(sources[:5], 1):  # Show first 5
        url = source.get('url', 'N/A')
        title = source.get('title', 'Unknown')
        source_type = source.get('source_type', 'unknown')

        print(f"\n[{i}] {title}")
        print(f"    Type: {source_type}")
        print(f"    URL: {url[:80]}...")

    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)


async def your_use_case():
    """Your specific use case: CertHub market research."""

    print("\n\n")
    print("=" * 80)
    print("YOUR USE CASE: CertHub Market Research")
    print("=" * 80)

    llm = LLMManager()

    prompt = """Use web search to investigate the market size of the company certhub.
Business description:
The company develops, operates, and distributes software applications and services
in the field of information technology, digital quality management, and the creation
of technical documentation. It provides a digital platform to support and accelerate
certification, especially for medical products.

This is the company's website:
https://www.certhub.de/en

Search at least 5-7 diverse sources. Return data as JSON array:

[
    {
        "market_size": [NUMBER, "CURRENCY", "MULTIPLIER"],
        "year": YEAR,
        "market_description": "DESCRIPTION",
        "geography": "GEOGRAPHY",
        "fit": "EXACT|BROADER|NARROWER",
        "confidence": "HIGH|MEDIUM|LOW",
        "source": "SOURCE_URL",
        "context": "VERBATIM_QUOTE"
    },
    ...
]

IMPORTANT: Return ONLY valid JSON. No markdown code blocks. Maximum 10 entries."""

    print("\nMaking API call (this may take 30-60 seconds)...\n")

    success, response = await llm.call_llm(
        family="gemini",
        model="gemini-2.5-flash",
        prompt=prompt,
        enable_google_search=True,
        return_json=True,
        max_tokens=8192
    )

    if not success:
        print(f"❌ Error: {response}")
        return

    # Parse JSON
    try:
        # Check if JSON is already wrapped (JSON array case)
        if 'data' in response:
            market_data = response['data']
        # Otherwise parse from 'text' field
        elif 'text' in response:
            market_data = json.loads(response['text'])
        else:
            market_data = response if isinstance(response, list) else [response]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ JSON parsing failed: {e}")
        if 'text' in response:
            print(f"\nRaw response (first 500 chars):")
            print(response['text'][:500])
        return

    # Get metadata
    sources = response.get('sources', [])
    search_queries = response.get('search_queries', [])

    # Display results
    print(f"✅ Retrieved {len(market_data)} market data points")
    print(f"✅ From {len(sources)} sources")
    print(f"✅ Used {len(search_queries)} search queries")

    print("\n" + "-" * 80)
    print("Market Size Data:")
    print("-" * 80)

    for item in market_data:
        size = item['market_size']
        year = item['year']
        desc = item['market_description']
        fit = item['fit']
        confidence = item.get('confidence', 'N/A')

        print(f"\n📊 {year}: {size[0]} {size[2]} {size[1]} [{fit} | {confidence}]")
        print(f"   {desc}")
        print(f"   Source: {item.get('source', 'N/A')[:60]}...")

    print("\n" + "-" * 80)
    print(f"Wrapper-extracted sources ({len(sources)}):")
    print("-" * 80)
    for i, source in enumerate(sources[:10], 1):
        title = source.get('title', 'Unknown')
        url = source.get('url', 'N/A')
        print(f"  [{i}] {title}")
        print(f"      {url[:70]}...")

    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)


async def main():
    """Run both examples."""

    # Run simple example
    await simple_example()

    # Ask if user wants to run the full example
    print("\n\nWould you like to run your full CertHub use case? (takes ~60 seconds)")
    print("This will make a real API call with grounding.")

    # For automated testing, just run it
    # In interactive mode, you'd ask for confirmation
    # input("Press Enter to continue or Ctrl+C to exit...")

    # await your_use_case()


if __name__ == "__main__":
    asyncio.run(main())
