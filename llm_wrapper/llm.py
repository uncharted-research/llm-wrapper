import asyncio
import json
import os
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from threading import Lock
from collections import defaultdict, deque
from pathlib import Path
from io import BytesIO

import google.genai as genai
from google.genai.types import Part
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import anthropic
from groq import AsyncGroq

class LLMManager:
    """Singleton class for managing LLM calls with rate limiting."""
    
    _instance = None
    _lock = Lock()
    
    # Rate limits: (calls_per_minute, tokens_per_minute)
    dict_gemini_limits = {
        "gemini-2.5-pro": (100, 2_000_000),
        "gemini-2.5-flash": (400, 500_000),
        "gemini-2.0-flash": (800, 1_000_000),
        "gemini-2.5-flash-lite": (3000, 2_500_000),
        "gemini-3-pro-preview": (100, 2_000_000),  # 45 calls/min, 1M tokens/min
        "imagen-3.0-generate-002": (5,)  # Only calls per minute limit
    }
    
    dict_claude_limits = {
        "claude-sonnet-4-20250514": (40, 20_000),  # Conservative limits
        "claude-opus-4-1-20250805": (40, 20_000),  # Conservative limits
        "claude-opus-4-6": (40, 20_000),  # Conservative limits
    }

    # Gemini embedding rate limits: (calls_per_minute, tokens_per_minute)
    dict_gemini_embedding_limits = {
        "gemini-embedding-001": (3000, 3_000_000),  # Conservative: 3000 calls/min, 3M tokens/min
    }

    # Valid embedding task types
    EMBEDDING_TASK_TYPES = {
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "CODE_RETRIEVAL_QUERY",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
    }

    # Valid embedding output dimensions
    EMBEDDING_DIMENSIONS = {768, 1536, 3072}

    # Default models
    DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_OPUS_MODEL = "claude-opus-4-6"
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
    DEFAULT_GEMINI3_MODEL = "gemini-3-pro-preview"
    DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
    DEFAULT_EMBEDDING_DIMENSIONS = 3072
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return

        # Load environment variables from .env file
        load_dotenv()

        # Initialize clients
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        # Load Groq rate limits from JSON file
        groq_limits_path = Path(__file__).parent / "groq_limits.json"
        with open(groq_limits_path, 'r') as f:
            self.dict_groq_limits = json.load(f)

        # Rate limiting counters: family -> model -> {'calls': deque, 'tokens': deque}
        self.counters = defaultdict(lambda: defaultdict(lambda: {
            'calls': deque(),
            'tokens': deque()
        }))

        self._initialized = True
    
    def _clean_old_entries(self, counter_deque: deque, window_seconds: int = 60, is_tokens: bool = False):
        """Remove entries older than the specified window."""
        current_time = time.time()
        if is_tokens:
            # For tokens: entries are (timestamp, token_count) tuples
            while counter_deque and counter_deque[0][0] <= current_time - window_seconds:
                counter_deque.popleft()
        else:
            # For calls: entries are just timestamps
            while counter_deque and counter_deque[0] <= current_time - window_seconds:
                counter_deque.popleft()
    
    def _check_rate_limits(self, family: str, model: str, estimated_tokens: int) -> bool:
        """Check if the request would exceed rate limits."""
        family_lower = family.lower()

        if family_lower == "gemini":
            if model not in self.dict_gemini_limits:
                raise ValueError(f"Unknown Gemini model: {model}")
            limits = self.dict_gemini_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
        elif family_lower == "claude":
            if model not in self.dict_claude_limits:
                raise ValueError(f"Unknown Claude model: {model}")
            limits = self.dict_claude_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
        elif family_lower == "groq":
            if model not in self.dict_groq_limits:
                # For Groq, we don't raise an error for unknown models
                # We'll let the API handle it, but we can't rate limit
                return True
            limits = self.dict_groq_limits[model]
            calls_limit = limits["rpm"]
            tokens_limit = limits["tpm"]
        elif family_lower == "gemini_embedding":
            if model not in self.dict_gemini_embedding_limits:
                raise ValueError(f"Unknown Gemini embedding model: {model}")
            limits = self.dict_gemini_embedding_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
        else:
            raise ValueError(f"Family {family} not supported")
        
        counters = self.counters[family_lower][model]
        
        # Clean old entries
        self._clean_old_entries(counters['calls'])
        if tokens_limit:
            self._clean_old_entries(counters['tokens'], is_tokens=True)
        
        # Check calls limit
        if len(counters['calls']) >= calls_limit:
            return False
        
        # Check tokens limit (if applicable)
        if tokens_limit:
            current_tokens = sum(token_count for timestamp, token_count in counters['tokens'])
            if current_tokens + estimated_tokens > tokens_limit:
                return False
        
        return True
    
    def _update_counters(self, family: str, model: str, tokens_used: int):
        """Update rate limiting counters."""
        current_time = time.time()
        counters = self.counters[family][model]
        
        counters['calls'].append(current_time)
        if tokens_used > 0:
            counters['tokens'].append((current_time, tokens_used))
    
    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate tokens from prompt text."""
        return len(prompt) // 3 if prompt else 0

    def _strip_markdown_json(self, text: str) -> str:
        """
        Strip markdown code block markers from JSON text.

        Handles variations like:
        - ```json\n{...}\n```
        - ```JSON\n{...}\n```
        - ``` json\n{...}\n```
        - ```{...}```

        Args:
            text: Text that may contain markdown code blocks

        Returns:
            Text with markdown markers removed
        """
        if not text:
            return text

        # Remove opening markdown (handles variations like ```json, ```JSON, ``` json, etc.)
        text = re.sub(r'^```\s*json\s*\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove closing markdown
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)

        # Clean up whitespace
        return text.strip()

    async def call_llm(
        self,
        family: str,
        model: str,
        prompt: str,
        file_path: Optional[Union[str, Path]] = None,
        image_data: Optional[bytes] = None,
        image_mime_type: str = "image/jpeg",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        enable_google_search: bool = False,
        enable_url_context: bool = False,
        thinking_budget: int = -1
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Make an LLM call with rate limiting.

        Args:
            family: LLM family ('Gemini' or 'Claude')
            model: Model name
            prompt: Text prompt
            file_path: Optional file to include (for Gemini only)
            image_data: Optional image bytes to include (for Gemini only)
            image_mime_type: MIME type of the image data (default: "image/jpeg")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
            enable_google_search: Enable Google Search tool (Gemini only, default: False)
            enable_url_context: Enable URL context tool (Gemini only, default: False)
            thinking_budget: Thinking budget for Gemini (-1 for unlimited, default: -1)
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        family = family.lower()
        
        if family not in ["gemini", "claude"]:
            return False, f"Unsupported family: {family}"
        
        estimated_tokens = self._estimate_tokens(prompt)
        
        # Check rate limits
        if not self._check_rate_limits(family, model, estimated_tokens):
            return False, "Rate limit exceeded"
        
        try:
            if family == "claude":
                if file_path or image_data:
                    return False, "Claude does not support file or image attachments in this implementation yet"
                result = await self._call_claude_with_prompt(
                    model, prompt, max_tokens, temperature, return_json
                )
            else:  # gemini
                if image_data is not None:
                    # Image data takes precedence over file_path
                    result = await self._call_gemini_with_image_data(
                        model, prompt, image_data, image_mime_type, max_tokens, temperature, return_json,
                        enable_google_search, enable_url_context, thinking_budget
                    )
                elif file_path:
                    result = await self._call_gemini_with_file(
                        model, prompt, file_path, max_tokens, temperature, return_json,
                        enable_google_search, enable_url_context, thinking_budget
                    )
                else:
                    result = await self._call_gemini_with_prompt(
                        model, prompt, max_tokens, temperature, return_json,
                        enable_google_search, enable_url_context, thinking_budget
                    )
            
            # Update counters on successful call
            self._update_counters(family, model, estimated_tokens)
            
            return result
            
        except Exception as e:
            return False, str(e)
    
    async def generate_image(
        self,
        model: str,
        prompt: str,
        input_image_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Union[Image.Image, str]]:
        """
        Generate an image using Gemini imagen models.
        
        Args:
            model: Model name (should be an imagen model)
            prompt: Text prompt for image generation
            input_image_path: Optional input image for editing
            
        Returns:
            Tuple of (success: bool, image: PIL.Image or error_message: str)
        """
        if model not in self.dict_gemini_limits:
            return False, f"Unknown model: {model}"
        
        if not model.startswith("imagen"):
            return False, f"Model {model} is not an image generation model"
        
        # Check rate limits (no token limit for imagen)
        if not self._check_rate_limits("gemini", model, 0):
            return False, "Rate limit exceeded"
        
        try:
            result = await self._generate_image_gemini(model, prompt, input_image_path)
            
            # Update counters (only calls, no tokens for imagen)
            self._update_counters("gemini", model, 0)
            
            return result
            
        except Exception as e:
            return False, str(e)
    
    async def _call_gemini_with_file(
        self,
        model: str,
        prompt: str,
        file_path: Union[str, Path],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        enable_google_search: bool = False,
        enable_url_context: bool = False,
        thinking_budget: int = -1
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Gemini with a file attachment."""
        def sync_call():
            try:
                file_path_obj = Path(file_path)
                
                # Determine MIME type based on file extension
                mime_type = "application/pdf"
                if file_path_obj.suffix.lower() in ['.jpg', '.jpeg']:
                    mime_type = "image/jpeg"
                elif file_path_obj.suffix.lower() in ['.png']:
                    mime_type = "image/png"
                elif file_path_obj.suffix.lower() in ['.txt']:
                    mime_type = "text/plain"

                # Build configuration
                config_dict = {}
                if max_tokens:
                    config_dict['max_output_tokens'] = max_tokens
                if temperature is not None:
                    config_dict['temperature'] = temperature

                # Always add thinking config
                config_dict['thinking_config'] = types.ThinkingConfig(thinking_budget=thinking_budget)

                # Add tools if any are enabled
                tools = []
                if enable_url_context:
                    tools.append(types.Tool(url_context=types.UrlContext()))
                if enable_google_search:
                    tools.append(types.Tool(googleSearch=types.GoogleSearch()))
                if tools:
                    config_dict['tools'] = tools

                config = types.GenerateContentConfig(**config_dict) if config_dict else None

                # Format contents based on whether tools are enabled
                if tools:
                    contents = [types.Content(
                        role="user",
                        parts=[
                            Part.from_bytes(
                                data=file_path_obj.read_bytes(),
                                mime_type=mime_type,
                            ),
                            types.Part.from_text(text=prompt)
                        ]
                    )]
                else:
                    contents = [
                        Part.from_bytes(
                            data=file_path_obj.read_bytes(),
                            mime_type=mime_type,
                        ),
                        prompt,
                    ]

                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                # Use the new helper to extract grounding info
                extracted_info = self._extract_grounding_info(response)

                if not extracted_info["text"]:
                    return False, "No response from Gemini"

                text_response = extracted_info["text"]

                # Prepare the result based on whether grounding is enabled
                if enable_google_search or enable_url_context:
                    # Include sources if grounding tools were enabled
                    result = {
                        "text": text_response,
                        "sources": extracted_info["sources"],
                        "grounding_enabled": {
                            "google_search": enable_google_search,
                            "url_context": enable_url_context
                        }
                    }

                    # Add search queries if available
                    if "search_queries" in extracted_info:
                        result["search_queries"] = extracted_info["search_queries"]

                    # Add grounding metadata if requested (could be made optional)
                    if extracted_info["grounding_metadata"]:
                        result["grounding_metadata"] = extracted_info["grounding_metadata"]
                else:
                    # No grounding tools enabled, return text only
                    result = {"text": text_response}

                # Handle JSON parsing if requested
                if return_json:
                    try:
                        # Strip markdown code blocks if present
                        text_response = self._strip_markdown_json(text_response)
                        parsed_data = json.loads(text_response)

                        # If grounding is enabled, merge the parsed JSON with sources
                        if enable_google_search or enable_url_context:
                            # If parsed data is a list, wrap it in a dict with sources
                            if isinstance(parsed_data, list):
                                wrapped_response = {
                                    "data": parsed_data,
                                    "sources": extracted_info["sources"],
                                    "search_queries": extracted_info.get("search_queries", []),
                                    "grounding_enabled": {
                                        "google_search": enable_google_search,
                                        "url_context": enable_url_context
                                    }
                                }
                                # Include grounding_metadata if available
                                if extracted_info.get("grounding_metadata"):
                                    wrapped_response["grounding_metadata"] = extracted_info["grounding_metadata"]
                                return True, wrapped_response
                            # If it's a dict, add sources directly
                            elif isinstance(parsed_data, dict):
                                parsed_data["sources"] = extracted_info["sources"]
                                if "search_queries" in extracted_info:
                                    parsed_data["search_queries"] = extracted_info["search_queries"]
                                parsed_data["grounding_enabled"] = {
                                    "google_search": enable_google_search,
                                    "url_context": enable_url_context
                                }

                        return True, parsed_data
                    except json.JSONDecodeError:
                        # Return the result as-is if JSON parsing fails
                        return True, result
                else:
                    return True, result

            except Exception as e:
                return False, str(e)

            # This should never be reached, but adding for safety
            return False, "Unexpected error: no return path taken"
        
        return await asyncio.to_thread(sync_call)
    
    async def _call_gemini_with_image_data(
        self,
        model: str,
        prompt: str,
        image_data: bytes,
        mime_type: str = "image/jpeg",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        enable_google_search: bool = False,
        enable_url_context: bool = False,
        thinking_budget: int = -1
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Gemini with image data bytes."""
        def sync_call():
            try:
                # Validate image_data
                if not image_data or len(image_data) == 0:
                    return False, "Image data is empty"
                
                # Validate mime_type
                supported_mime_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
                if mime_type not in supported_mime_types:
                    return False, f"Unsupported MIME type: {mime_type}. Supported types: {', '.join(supported_mime_types)}"

                # Build configuration
                config_dict = {}
                if max_tokens:
                    config_dict['max_output_tokens'] = max_tokens
                if temperature is not None:
                    config_dict['temperature'] = temperature

                # Always add thinking config
                config_dict['thinking_config'] = types.ThinkingConfig(thinking_budget=thinking_budget)

                # Add tools if any are enabled
                tools = []
                if enable_url_context:
                    tools.append(types.Tool(url_context=types.UrlContext()))
                if enable_google_search:
                    tools.append(types.Tool(googleSearch=types.GoogleSearch()))
                if tools:
                    config_dict['tools'] = tools

                config = types.GenerateContentConfig(**config_dict) if config_dict else None

                # Format contents based on whether tools are enabled
                if tools:
                    contents = [types.Content(
                        role="user",
                        parts=[
                            Part.from_bytes(
                                data=image_data,
                                mime_type=mime_type,
                            ),
                            types.Part.from_text(text=prompt)
                        ]
                    )]
                else:
                    contents = [
                        Part.from_bytes(
                            data=image_data,
                            mime_type=mime_type,
                        ),
                        prompt,
                    ]

                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                # Use the new helper to extract grounding info
                extracted_info = self._extract_grounding_info(response)

                if not extracted_info["text"]:
                    return False, "No response from Gemini"

                text_response = extracted_info["text"]

                # Prepare the result based on whether grounding is enabled
                if enable_google_search or enable_url_context:
                    # Include sources if grounding tools were enabled
                    result = {
                        "text": text_response,
                        "sources": extracted_info["sources"],
                        "grounding_enabled": {
                            "google_search": enable_google_search,
                            "url_context": enable_url_context
                        }
                    }

                    # Add search queries if available
                    if "search_queries" in extracted_info:
                        result["search_queries"] = extracted_info["search_queries"]

                    # Add grounding metadata if requested (could be made optional)
                    if extracted_info["grounding_metadata"]:
                        result["grounding_metadata"] = extracted_info["grounding_metadata"]
                else:
                    # No grounding tools enabled, return text only
                    result = {"text": text_response}

                # Handle JSON parsing if requested
                if return_json:
                    try:
                        # Strip markdown code blocks if present
                        text_response = self._strip_markdown_json(text_response)
                        parsed_data = json.loads(text_response)

                        # If grounding is enabled, merge the parsed JSON with sources
                        if enable_google_search or enable_url_context:
                            # If parsed data is a list, wrap it in a dict with sources
                            if isinstance(parsed_data, list):
                                wrapped_response = {
                                    "data": parsed_data,
                                    "sources": extracted_info["sources"],
                                    "search_queries": extracted_info.get("search_queries", []),
                                    "grounding_enabled": {
                                        "google_search": enable_google_search,
                                        "url_context": enable_url_context
                                    }
                                }
                                # Include grounding_metadata if available
                                if extracted_info.get("grounding_metadata"):
                                    wrapped_response["grounding_metadata"] = extracted_info["grounding_metadata"]
                                return True, wrapped_response
                            # If it's a dict, add sources directly
                            elif isinstance(parsed_data, dict):
                                parsed_data["sources"] = extracted_info["sources"]
                                if "search_queries" in extracted_info:
                                    parsed_data["search_queries"] = extracted_info["search_queries"]
                                parsed_data["grounding_enabled"] = {
                                    "google_search": enable_google_search,
                                    "url_context": enable_url_context
                                }

                        return True, parsed_data
                    except json.JSONDecodeError:
                        # Return the result as-is if JSON parsing fails
                        return True, result
                else:
                    return True, result

            except Exception as e:
                return False, str(e)
        
        return await asyncio.to_thread(sync_call)

    def _extract_grounding_info(self, response):
        """Extract grounding information and sources from Gemini response.

        Args:
            response: The Gemini API response object

        Returns:
            A dictionary containing:
                - text: The main response text
                - sources: List of source dictionaries with url, title, and snippet
                - grounding_metadata: Raw grounding metadata if available
        """
        result = {
            "text": "",
            "sources": [],
            "grounding_metadata": None
        }

        # Extract main text - use response.text if available (simpler and more reliable)
        if hasattr(response, 'text') and response.text:
            result["text"] = response.text.strip()

        # Also need candidate reference for metadata extraction below
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]

            # If we didn't get text from response.text, extract from parts manually
            if not result["text"]:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)

                    # Combine all text parts
                    result["text"] = "\n".join(text_parts).strip() if text_parts else ""

            # Check for grounding support in the content itself
            # Sometimes Gemini includes sources directly in the response
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                result["grounding_metadata"] = candidate.grounding_metadata

                # Extract web search queries
                if hasattr(candidate.grounding_metadata, 'web_search_queries') and candidate.grounding_metadata.web_search_queries:
                    for query in candidate.grounding_metadata.web_search_queries:
                        if query not in result.get("search_queries", []):
                            result.setdefault("search_queries", []).append(query)

                # Extract grounding chunks (sources)
                if hasattr(candidate.grounding_metadata, 'grounding_chunks') and candidate.grounding_metadata.grounding_chunks:
                    for chunk in candidate.grounding_metadata.grounding_chunks:
                        source_info = {}

                        # Check for web reference
                        if hasattr(chunk, 'web') and chunk.web:
                            web_ref = chunk.web
                            if hasattr(web_ref, 'uri'):
                                source_info['url'] = web_ref.uri
                            if hasattr(web_ref, 'title'):
                                source_info['title'] = web_ref.title

                        # Check for retrieval reference (for document retrieval)
                        if hasattr(chunk, 'retrieval') and chunk.retrieval:
                            retrieval_ref = chunk.retrieval
                            if hasattr(retrieval_ref, 'uri'):
                                source_info['url'] = retrieval_ref.uri
                            if hasattr(retrieval_ref, 'title'):
                                source_info['title'] = retrieval_ref.title

                        if source_info and source_info not in result["sources"]:
                            result["sources"].append(source_info)

                # Extract grounding supports (text segments linked to sources)
                if hasattr(candidate.grounding_metadata, 'grounding_supports') and candidate.grounding_metadata.grounding_supports:
                    # This links text segments to specific grounding chunks
                    # We could use this to provide more detailed citation information
                    pass

                # Extract URLs from search_entry_point if grounding_chunks is empty
                # Note: grounding_chunks is often None even when grounding works
                # The actual source URLs are in the HTML of search_entry_point
                if hasattr(candidate.grounding_metadata, 'search_entry_point') and candidate.grounding_metadata.search_entry_point:
                    sep = candidate.grounding_metadata.search_entry_point

                    if hasattr(sep, 'rendered_content') and sep.rendered_content:
                        # Extract URLs from the HTML using regex
                        # Look for href="https://..." patterns
                        url_pattern = r'href="(https://[^"]+)"'
                        found_urls = re.findall(url_pattern, sep.rendered_content)

                        # Filter out SVG namespace URLs and add real sources
                        for url in found_urls:
                            if not url.startswith('http://www.w3.org/'):
                                # Check if not already in sources
                                existing_urls = [s.get('url') for s in result["sources"]]
                                if url not in existing_urls:
                                    result["sources"].append({
                                        'url': url,
                                        'source_type': 'google_search'
                                    })

            # Extract sources from url_context_metadata if available
            # This is separate from grounding_metadata and provides URL retrieval status
            if hasattr(candidate, 'url_context_metadata') and candidate.url_context_metadata:
                ucm = candidate.url_context_metadata

                # Extract URL metadata
                if hasattr(ucm, 'url_metadata') and ucm.url_metadata:
                    for url_meta in ucm.url_metadata:
                        source_info = {}

                        if hasattr(url_meta, 'retrieved_url'):
                            source_info['url'] = url_meta.retrieved_url
                            source_info['source_type'] = 'url_context'

                        if hasattr(url_meta, 'url_retrieval_status'):
                            source_info['retrieval_status'] = str(url_meta.url_retrieval_status)

                        # Only add if URL was successfully retrieved
                        if source_info.get('retrieval_status') == 'URL_RETRIEVAL_STATUS_SUCCESS':
                            # Check if not already in sources (avoid duplicates)
                            existing_urls = [s.get('url') for s in result["sources"]]
                            if source_info.get('url') not in existing_urls:
                                result["sources"].append(source_info)

        # Check if sources are embedded in the response text itself
        # Gemini sometimes includes citations and a source list at the end

        # First, check for inline citations like [cite:1, 1~description]
        inline_citation_pattern = r'\[cite:(\d+),\s*\d+~[^\]]+\]'
        inline_citations = re.findall(inline_citation_pattern, result["text"])

        # Look for a sources section at the end of the text
        # Match patterns like "Sources:", "**Sources:**", etc.
        sources_section_pattern = r'(?:\*{0,2}Sources?\*{0,2}:?\s*\n)((?:\d+\..*\n?)+)'
        sources_match = re.search(sources_section_pattern, result["text"], re.DOTALL | re.IGNORECASE)

        if not sources_match:
            # Try alternative pattern without markdown
            sources_section_pattern = r'(?:Sources?:?\s*\n)((?:\d+\..*\n?)+)$'
            sources_match = re.search(sources_section_pattern, result["text"], re.DOTALL | re.IGNORECASE)

        if sources_match:
            sources_text = sources_match.group(1)
            # Parse individual source lines
            source_lines = sources_text.strip().split('\n')

            for line in source_lines:
                line = line.strip()
                if not line:
                    continue

                # Match numbered sources like "1. https://url" or "1. title - url"
                # Handle both direct URLs and URLs with titles
                numbered_pattern = r'^(\d+)\.\s+(.+)$'
                numbered_match = re.match(numbered_pattern, line)

                if numbered_match:
                    source_num = numbered_match.group(1)
                    source_content = numbered_match.group(2).strip()

                    # Check if it's just a URL
                    if source_content.startswith('http'):
                        # Extract domain name as title from URL
                        domain_match = re.search(r'https?://([^/]+)', source_content)
                        title = domain_match.group(1) if domain_match else f"Source {source_num}"
                        result["sources"].append({
                            "number": int(source_num),
                            "title": title,
                            "url": source_content
                        })
                    else:
                        # Try to extract title and URL
                        # Pattern for "title - url" or "title: url"
                        title_url_pattern = r'^(.+?)\s*[-–:]\s*(https?://\S+)'
                        title_match = re.match(title_url_pattern, source_content)

                        if title_match:
                            result["sources"].append({
                                "number": int(source_num),
                                "title": title_match.group(1).strip(),
                                "url": title_match.group(2).strip()
                            })
                        elif source_content.startswith('http'):
                            # Just a URL without title
                            domain_match = re.search(r'https?://([^/]+)', source_content)
                            title = domain_match.group(1) if domain_match else f"Source {source_num}"
                            result["sources"].append({
                                "number": int(source_num),
                                "title": title,
                                "url": source_content
                            })

        # Remove duplicate sources
        seen_urls = set()
        unique_sources = []
        for source in result["sources"]:
            if 'url' in source and source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)
        result["sources"] = unique_sources

        return result

    async def _call_gemini_with_prompt(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        enable_google_search: bool = False,
        enable_url_context: bool = False,
        thinking_budget: int = -1
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Gemini with text prompt only."""
        def sync_call():
            try:
                # Build configuration
                config_dict = {}
                if max_tokens:
                    config_dict['max_output_tokens'] = max_tokens
                if temperature is not None:
                    config_dict['temperature'] = temperature

                # Always add thinking config
                config_dict['thinking_config'] = types.ThinkingConfig(thinking_budget=thinking_budget)

                # Add tools if any are enabled
                tools = []
                if enable_url_context:
                    tools.append(types.Tool(url_context=types.UrlContext()))
                if enable_google_search:
                    tools.append(types.Tool(googleSearch=types.GoogleSearch()))
                if tools:
                    config_dict['tools'] = tools

                config = types.GenerateContentConfig(**config_dict) if config_dict else None

                # Format contents based on whether tools are enabled
                if tools:
                    contents = [types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )]
                else:
                    contents = [prompt]

                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                # Use the new helper to extract grounding info
                extracted_info = self._extract_grounding_info(response)

                if not extracted_info["text"]:
                    return False, "No response from Gemini"

                text_response = extracted_info["text"]

                # Prepare the result based on whether grounding is enabled
                if enable_google_search or enable_url_context:
                    # Include sources if grounding tools were enabled
                    result = {
                        "text": text_response,
                        "sources": extracted_info["sources"],
                        "grounding_enabled": {
                            "google_search": enable_google_search,
                            "url_context": enable_url_context
                        }
                    }

                    # Add search queries if available
                    if "search_queries" in extracted_info:
                        result["search_queries"] = extracted_info["search_queries"]

                    # Add grounding metadata if requested (could be made optional)
                    if extracted_info["grounding_metadata"]:
                        result["grounding_metadata"] = extracted_info["grounding_metadata"]
                else:
                    # No grounding tools enabled, return text only
                    result = {"text": text_response}

                # Handle JSON parsing if requested
                if return_json:
                    try:
                        # Strip markdown code blocks if present
                        text_response = self._strip_markdown_json(text_response)
                        parsed_data = json.loads(text_response)

                        # If grounding is enabled, merge the parsed JSON with sources
                        if enable_google_search or enable_url_context:
                            # If parsed data is a list, wrap it in a dict with sources
                            if isinstance(parsed_data, list):
                                wrapped_response = {
                                    "data": parsed_data,
                                    "sources": extracted_info["sources"],
                                    "search_queries": extracted_info.get("search_queries", []),
                                    "grounding_enabled": {
                                        "google_search": enable_google_search,
                                        "url_context": enable_url_context
                                    }
                                }
                                # Include grounding_metadata if available
                                if extracted_info.get("grounding_metadata"):
                                    wrapped_response["grounding_metadata"] = extracted_info["grounding_metadata"]
                                return True, wrapped_response
                            # If it's a dict, add sources directly
                            elif isinstance(parsed_data, dict):
                                parsed_data["sources"] = extracted_info["sources"]
                                if "search_queries" in extracted_info:
                                    parsed_data["search_queries"] = extracted_info["search_queries"]
                                parsed_data["grounding_enabled"] = {
                                    "google_search": enable_google_search,
                                    "url_context": enable_url_context
                                }

                        return True, parsed_data
                    except json.JSONDecodeError:
                        # Return the result as-is if JSON parsing fails
                        return True, result
                else:
                    return True, result

            except Exception as e:
                return False, str(e)
            
            # This should never be reached, but adding for safety
            return False, "Unexpected error: no return path taken"
        
        return await asyncio.to_thread(sync_call)

    async def _call_gemini3_with_prompt(
        self,
        model: str,
        prompt: str,
        file_path: Optional[Union[str, Path]] = None,
        image_data: Optional[bytes] = None,
        image_mime_type: str = "image/jpeg",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        enable_google_search: bool = False,
        enable_url_context: bool = False,
        thinking_level: str = "HIGH",
        media_resolution: str = "HIGH"
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Gemini 3 with extended thinking support and optional file/image input."""
        def sync_call():
            try:
                # Validate thinking_level
                valid_thinking_levels = ["HIGH", "LOW"]
                if thinking_level not in valid_thinking_levels:
                    return False, f"Invalid thinking_level: {thinking_level}. Valid values: {', '.join(valid_thinking_levels)}"

                # Validate media_resolution
                valid_resolutions = ["HIGH", "MEDIUM", "LOW"]
                if media_resolution not in valid_resolutions:
                    return False, f"Invalid media_resolution: {media_resolution}. Valid values: {', '.join(valid_resolutions)}"

                # Map media_resolution to API format
                resolution_map = {
                    "HIGH": "MEDIA_RESOLUTION_HIGH",
                    "MEDIUM": "MEDIA_RESOLUTION_MEDIUM",
                    "LOW": "MEDIA_RESOLUTION_LOW"
                }

                # Build configuration
                config_dict = {}
                if max_tokens:
                    config_dict['max_output_tokens'] = max_tokens
                if temperature is not None:
                    config_dict['temperature'] = temperature

                # Gemini 3 thinking config (different from Gemini 2.x)
                config_dict['thinking_config'] = {"thinking_level": thinking_level}

                # Media resolution (new for Gemini 3)
                config_dict['media_resolution'] = resolution_map[media_resolution]

                # Add tools if any are enabled
                tools = []
                if enable_url_context:
                    tools.append(types.Tool(url_context=types.UrlContext()))
                if enable_google_search:
                    tools.append(types.Tool(googleSearch=types.GoogleSearch()))
                if tools:
                    config_dict['tools'] = tools

                config = types.GenerateContentConfig(**config_dict)

                # Build contents based on input type
                parts = []

                # Handle file input
                if file_path:
                    file_path_obj = Path(file_path)

                    # Determine MIME type based on file extension
                    mime_type = "application/pdf"
                    if file_path_obj.suffix.lower() in ['.jpg', '.jpeg']:
                        mime_type = "image/jpeg"
                    elif file_path_obj.suffix.lower() in ['.png']:
                        mime_type = "image/png"
                    elif file_path_obj.suffix.lower() in ['.webp']:
                        mime_type = "image/webp"
                    elif file_path_obj.suffix.lower() in ['.gif']:
                        mime_type = "image/gif"
                    elif file_path_obj.suffix.lower() in ['.txt']:
                        mime_type = "text/plain"

                    parts.append(Part.from_bytes(
                        data=file_path_obj.read_bytes(),
                        mime_type=mime_type,
                    ))

                # Handle image data input
                elif image_data:
                    # Validate image_data
                    if len(image_data) == 0:
                        return False, "Image data is empty"

                    # Validate mime_type
                    supported_mime_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
                    if image_mime_type not in supported_mime_types:
                        return False, f"Unsupported MIME type: {image_mime_type}. Supported types: {', '.join(supported_mime_types)}"

                    parts.append(Part.from_bytes(
                        data=image_data,
                        mime_type=image_mime_type,
                    ))

                # Add prompt as text part
                parts.append(types.Part.from_text(text=prompt))

                # Format contents based on whether tools are enabled
                if tools or file_path or image_data:
                    contents = [types.Content(role="user", parts=parts)]
                else:
                    contents = [prompt]

                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                # Use the helper to extract grounding info
                extracted_info = self._extract_grounding_info(response)

                if not extracted_info["text"]:
                    return False, "No response from Gemini 3"

                text_response = extracted_info["text"]

                # Prepare the result based on whether grounding is enabled
                if enable_google_search or enable_url_context:
                    result = {
                        "text": text_response,
                        "sources": extracted_info["sources"],
                        "grounding_enabled": {
                            "google_search": enable_google_search,
                            "url_context": enable_url_context
                        }
                    }

                    if "search_queries" in extracted_info:
                        result["search_queries"] = extracted_info["search_queries"]

                    if extracted_info["grounding_metadata"]:
                        result["grounding_metadata"] = extracted_info["grounding_metadata"]
                else:
                    result = {"text": text_response}

                # Handle JSON parsing if requested
                if return_json:
                    try:
                        text_response = self._strip_markdown_json(text_response)
                        parsed_data = json.loads(text_response)

                        if enable_google_search or enable_url_context:
                            if isinstance(parsed_data, list):
                                wrapped_response = {
                                    "data": parsed_data,
                                    "sources": extracted_info["sources"],
                                    "search_queries": extracted_info.get("search_queries", []),
                                    "grounding_enabled": {
                                        "google_search": enable_google_search,
                                        "url_context": enable_url_context
                                    }
                                }
                                if extracted_info.get("grounding_metadata"):
                                    wrapped_response["grounding_metadata"] = extracted_info["grounding_metadata"]
                                return True, wrapped_response
                            elif isinstance(parsed_data, dict):
                                parsed_data["sources"] = extracted_info["sources"]
                                if "search_queries" in extracted_info:
                                    parsed_data["search_queries"] = extracted_info["search_queries"]
                                parsed_data["grounding_enabled"] = {
                                    "google_search": enable_google_search,
                                    "url_context": enable_url_context
                                }

                        return True, parsed_data
                    except json.JSONDecodeError:
                        return True, result
                else:
                    return True, result

            except Exception as e:
                return False, str(e)

            return False, "Unexpected error: no return path taken"

        return await asyncio.to_thread(sync_call)

    async def _call_claude_with_prompt(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Claude with text prompt only."""
        def sync_call():
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens or 1024
                }
                
                if temperature is not None:
                    kwargs["temperature"] = temperature
                
                message = self.claude_client.messages.create(**kwargs)
                
                if not message.content or not message.content[0].text:
                    return False, "No response from Claude"
                
                text_response = message.content[0].text.strip()
                
                # Try to parse as JSON if requested
                if return_json:
                    try:
                        # Strip markdown code blocks if present
                        text_response = self._strip_markdown_json(text_response)
                        data_dict = json.loads(text_response)
                        return True, data_dict
                    except json.JSONDecodeError:
                        return True, {"text": text_response}
                else:
                    return True, {"text": text_response}
                
            except Exception as e:
                return False, str(e)
            
            # This should never be reached, but adding for safety
            return False, "Unexpected error: no return path taken"
        
        return await asyncio.to_thread(sync_call)

    async def _call_opus_with_prompt(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128000,
        temperature: float = 1,
        budget_tokens: int = 128000,
        effort: str = "max",
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Claude Opus with extended thinking enabled."""
        def sync_call():
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": budget_tokens
                    },
                    "output_config": {"effort": effort}
                }

                message = self.claude_client.messages.create(**kwargs)

                # Extract text from response content blocks
                # With thinking enabled, response contains both thinking and text blocks
                text_parts = []
                thinking_text = None
                for block in message.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                    elif block.type == "text":
                        text_parts.append(block.text)

                if not text_parts:
                    return False, "No response from Claude Opus"

                text_response = "\n".join(text_parts).strip()

                if return_json:
                    try:
                        text_response = self._strip_markdown_json(text_response)
                        data_dict = json.loads(text_response)
                        return True, data_dict
                    except json.JSONDecodeError:
                        return True, {"text": text_response}
                else:
                    result = {"text": text_response}
                    if thinking_text:
                        result["thinking"] = thinking_text
                    return True, result

            except Exception as e:
                return False, str(e)

            return False, "Unexpected error: no return path taken"

        return await asyncio.to_thread(sync_call)

    async def _call_groq_with_prompt(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Groq with text prompt only (non-streaming)."""
        try:
            # Build request parameters
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            # Make API call
            chat_completion = await self.groq_client.chat.completions.create(**kwargs)

            if not chat_completion.choices or not chat_completion.choices[0].message.content:
                return False, "No response from Groq"

            text_response = chat_completion.choices[0].message.content.strip()

            # Try to parse as JSON if requested
            if return_json:
                try:
                    # Strip markdown code blocks if present
                    text_response = self._strip_markdown_json(text_response)
                    data_dict = json.loads(text_response)
                    return True, data_dict
                except json.JSONDecodeError:
                    return True, {"text": text_response}
            else:
                return True, {"text": text_response}

        except Exception as e:
            # Handle Groq API errors (including invalid model names)
            error_msg = str(e)
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                return False, f"Unknown Groq model: {model}. Error: {error_msg}"
            return False, f"Groq API error: {error_msg}"

    async def _call_groq_with_prompt_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Groq with text prompt only (streaming, but returns full accumulated text)."""
        try:
            # Build request parameters
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            # Make streaming API call
            stream = await self.groq_client.chat.completions.create(**kwargs)

            # Accumulate chunks
            accumulated_text = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    accumulated_text += chunk.choices[0].delta.content

            if not accumulated_text:
                return False, "No response from Groq"

            text_response = accumulated_text.strip()

            # Try to parse as JSON if requested
            if return_json:
                try:
                    # Strip markdown code blocks if present
                    text_response = self._strip_markdown_json(text_response)
                    data_dict = json.loads(text_response)
                    return True, data_dict
                except json.JSONDecodeError:
                    return True, {"text": text_response}
            else:
                return True, {"text": text_response}

        except Exception as e:
            # Handle Groq API errors (including invalid model names)
            error_msg = str(e)
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                return False, f"Unknown Groq model: {model}. Error: {error_msg}"
            return False, f"Groq API error: {error_msg}"

    async def _generate_image_gemini(
        self,
        model: str,
        prompt: str,
        input_image_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, Union[Image.Image, str]]:
        """Generate image using Gemini imagen models."""
        def sync_call():
            try:
                contents = [prompt]
                
                if input_image_path:
                    input_image = Image.open(input_image_path)
                    contents.append(input_image)
                
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE']
                    )
                )
                
                # Extract image from response
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        return True, image
                
                return False, "No image generated"
                
            except Exception as e:
                return False, str(e)
        
        return await asyncio.to_thread(sync_call)

    async def _embed_gemini(
        self,
        model: str,
        contents: Union[str, List[str]],
        task_type: Optional[str] = None,
        output_dimensionality: Optional[int] = None
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Embed text(s) using Gemini embedding models.

        Args:
            model: Embedding model name
            contents: Single text or list of texts to embed
            task_type: Optional embedding task type
            output_dimensionality: Optional output vector size

        Returns:
            Tuple of (success, result dict with embeddings or error message)
        """
        def sync_call():
            try:
                # Build config
                config_dict = {}
                if task_type:
                    config_dict['task_type'] = task_type
                if output_dimensionality:
                    config_dict['output_dimensionality'] = output_dimensionality

                config = types.EmbedContentConfig(**config_dict) if config_dict else None

                response = self.gemini_client.models.embed_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                # Extract embeddings from response
                if hasattr(response, 'embeddings') and response.embeddings:
                    embeddings = []
                    for embedding in response.embeddings:
                        if hasattr(embedding, 'values'):
                            embeddings.append(list(embedding.values))
                        else:
                            embeddings.append(list(embedding))

                    return True, {
                        "embeddings": embeddings,
                        "model": model,
                        "dimensions": len(embeddings[0]) if embeddings else 0,
                        "task_type": task_type
                    }
                else:
                    return False, "No embeddings in response"

            except Exception as e:
                return False, str(e)

        return await asyncio.to_thread(sync_call)

    async def embed_content(
        self,
        contents: Union[str, List[str]],
        model: Optional[str] = None,
        task_type: str = "SEMANTIC_SIMILARITY",
        output_dimensionality: int = 3072
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Generate embeddings for text content using Gemini.

        Args:
            contents: Single text or list of texts to embed
            model: Embedding model (default: gemini-embedding-001)
            task_type: One of SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING,
                       RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, CODE_RETRIEVAL_QUERY,
                       QUESTION_ANSWERING, FACT_VERIFICATION
            output_dimensionality: Size of output vector (768, 1536, or 3072; default: 3072)

        Returns:
            Tuple of (success, result dict with "embeddings" key or error message)
            Result dict format: {"embeddings": [[float, ...], ...], "model": str, "dimensions": int, "task_type": str}
        """
        model = model or self.DEFAULT_EMBEDDING_MODEL

        # Validate task_type
        if task_type not in self.EMBEDDING_TASK_TYPES:
            return False, f"Invalid task_type: {task_type}. Valid types: {', '.join(sorted(self.EMBEDDING_TASK_TYPES))}"

        # Validate output_dimensionality
        if output_dimensionality not in self.EMBEDDING_DIMENSIONS:
            return False, f"Invalid output_dimensionality: {output_dimensionality}. Valid values: {sorted(self.EMBEDDING_DIMENSIONS)}"

        # Estimate tokens for rate limiting
        if isinstance(contents, str):
            estimated_tokens = self._estimate_tokens(contents)
        else:
            estimated_tokens = sum(self._estimate_tokens(text) for text in contents)

        # Check rate limits
        if not self._check_rate_limits("gemini_embedding", model, estimated_tokens):
            return False, "Rate limit exceeded"

        try:
            result = await self._embed_gemini(
                model=model,
                contents=contents,
                task_type=task_type,
                output_dimensionality=output_dimensionality
            )

            # Update counters on successful call
            if result[0]:  # If success
                self._update_counters("gemini_embedding", model, estimated_tokens)

            return result

        except Exception as e:
            return False, str(e)

    def get_rate_limit_status(self, family: str, model: str) -> Dict[str, Any]:
        """Get current rate limit status for a model."""
        family = family.lower()

        if family == "gemini" and model in self.dict_gemini_limits:
            limits = self.dict_gemini_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
        elif family == "claude" and model in self.dict_claude_limits:
            limits = self.dict_claude_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
        elif family == "groq" and model in self.dict_groq_limits:
            limits = self.dict_groq_limits[model]
            calls_limit = limits["rpm"]
            tokens_limit = limits["tpm"]
        elif family == "gemini_embedding" and model in self.dict_gemini_embedding_limits:
            limits = self.dict_gemini_embedding_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
        else:
            return {"error": f"Unknown model {model} for family {family}"}

        counters = self.counters[family][model]

        # Clean old entries
        self._clean_old_entries(counters['calls'])
        if tokens_limit:
            self._clean_old_entries(counters['tokens'], is_tokens=True)

        status = {
            "calls_used": len(counters['calls']),
            "calls_limit": calls_limit,
            "calls_remaining": calls_limit - len(counters['calls'])
        }

        if tokens_limit:  # Has token limit
            tokens_used = sum(token_count for timestamp, token_count in counters['tokens'])
            status.update({
                "tokens_used": tokens_used,
                "tokens_limit": tokens_limit,
                "tokens_remaining": tokens_limit - tokens_used
            })

        return status
    
    async def call_claude(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Convenience method to call Claude with Sonnet as default model.

        Args:
            prompt: Text prompt
            model: Model name (defaults to Sonnet)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        return await self.call_llm(
            family="claude",
            model=model or self.DEFAULT_CLAUDE_MODEL,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            return_json=return_json
        )

    async def call_opus(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 128000,
        temperature: float = 1,
        budget_tokens: int = 128000,
        effort: str = "max",
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Call Claude Opus 4.6 with extended thinking enabled.

        Args:
            prompt: Text prompt
            model: Model name (defaults to claude-opus-4-6)
            max_tokens: Maximum tokens to generate (default: 128000)
            temperature: Sampling temperature (default: 1, required for extended thinking)
            budget_tokens: Thinking budget tokens (default: 128000)
            effort: Output effort level (default: "max")
            return_json: Whether to return the response as a JSON object
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
            Result dict contains "text" key, and optionally "thinking" key with the model's reasoning.
        """
        model = model or self.DEFAULT_OPUS_MODEL
        estimated_tokens = self._estimate_tokens(prompt)

        # Check rate limits
        if not self._check_rate_limits("claude", model, estimated_tokens):
            return False, "Rate limit exceeded"

        try:
            result = await self._call_opus_with_prompt(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                budget_tokens=budget_tokens,
                effort=effort,
                return_json=return_json
            )

            # Update counters on successful call
            if result[0]:
                self._update_counters("claude", model, estimated_tokens)

            return result

        except Exception as e:
            return False, str(e)

    async def call_gemini3(
        self,
        prompt: str,
        model: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        image_data: Optional[bytes] = None,
        image_mime_type: str = "image/jpeg",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        enable_google_search: bool = False,
        enable_url_context: bool = False,
        thinking_level: str = "HIGH",
        media_resolution: str = "HIGH"
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Convenience method to call Gemini 3 with extended thinking support.

        Args:
            prompt: Text prompt
            model: Model name (defaults to gemini-3-pro-preview)
            file_path: Optional path to a file (PDF, image, text)
            image_data: Optional image bytes to send directly
            image_mime_type: MIME type for image_data (default: image/jpeg)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
            enable_google_search: Enable Google Search grounding
            enable_url_context: Enable URL context analysis
            thinking_level: Thinking depth ("HIGH" or "LOW", default "HIGH")
            media_resolution: Media resolution ("HIGH", "MEDIUM", or "LOW", default "HIGH")
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        model = model or self.DEFAULT_GEMINI3_MODEL
        estimated_tokens = self._estimate_tokens(prompt)

        # Check rate limits
        if not self._check_rate_limits("gemini", model, estimated_tokens):
            return False, "Rate limit exceeded"

        try:
            result = await self._call_gemini3_with_prompt(
                model=model,
                prompt=prompt,
                file_path=file_path,
                image_data=image_data,
                image_mime_type=image_mime_type,
                max_tokens=max_tokens,
                temperature=temperature,
                return_json=return_json,
                enable_google_search=enable_google_search,
                enable_url_context=enable_url_context,
                thinking_level=thinking_level,
                media_resolution=media_resolution
            )

            # Update counters on successful call
            if result[0]:  # If success
                self._update_counters("gemini", model, estimated_tokens)

            return result

        except Exception as e:
            return False, str(e)

    async def call_via_groq(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Call Groq API with the specified model (non-streaming).

        Args:
            model: Groq model name (e.g., "llama-3.3-70b-versatile", "qwen/qwen3-32b")
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        estimated_tokens = self._estimate_tokens(prompt)

        # Check rate limits
        if not self._check_rate_limits("groq", model, estimated_tokens):
            return False, "Rate limit exceeded"

        try:
            result = await self._call_groq_with_prompt(
                model, prompt, max_tokens, temperature, return_json
            )

            # Update counters on successful call
            if result[0]:  # If success
                self._update_counters("groq", model, estimated_tokens)

            return result

        except Exception as e:
            return False, str(e)

    async def call_via_groq_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Call Groq API with the specified model (streaming, returns full accumulated text).

        Args:
            model: Groq model name (e.g., "llama-3.3-70b-versatile", "qwen/qwen3-32b")
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        estimated_tokens = self._estimate_tokens(prompt)

        # Check rate limits
        if not self._check_rate_limits("groq", model, estimated_tokens):
            return False, "Rate limit exceeded"

        try:
            result = await self._call_groq_with_prompt_stream(
                model, prompt, max_tokens, temperature, return_json
            )

            # Update counters on successful call
            if result[0]:  # If success
                self._update_counters("groq", model, estimated_tokens)

            return result

        except Exception as e:
            return False, str(e)

    async def call_groq(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False,
        stream: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Convenience method to call Groq with llama-3.3-70b-versatile as default model.

        Args:
            prompt: Text prompt
            model: Model name (defaults to llama-3.3-70b-versatile)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
            stream: Whether to use streaming (default: False)
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        if stream:
            return await self.call_via_groq_stream(
                model=model or self.DEFAULT_GROQ_MODEL,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                return_json=return_json
            )
        else:
            return await self.call_via_groq(
                model=model or self.DEFAULT_GROQ_MODEL,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                return_json=return_json
            )


# Convenience function to get the singleton instance
def get_llm_manager() -> LLMManager:
    """Get the singleton LLM manager instance."""
    return LLMManager()
