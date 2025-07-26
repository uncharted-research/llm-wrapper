import asyncio
import json
import os
import time
import warnings
from typing import Dict, Tuple, Optional, Union, Any
from threading import Lock
from collections import defaultdict, deque
from pathlib import Path
from io import BytesIO

import google.genai as genai
from google.genai.types import Part
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

class LLMManager:
    """Singleton class for managing LLM calls with rate limiting."""
    
    _instance = None
    _lock = Lock()
    
    # Rate limits: (calls_per_minute, tokens_per_minute)
    dict_gemini_limits = {
        "gemini-2.5-pro": (50, 800_000),
        "gemini-2.5-flash": (400, 500_000),
        "gemini-2.0-flash": (800, 1_000_000),
        "imagen-3.0-generate-002": (5,)  # Only calls per minute limit
    }
    
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
        if family.lower() == "gemini":
            if model not in self.dict_gemini_limits:
                raise ValueError(f"Unknown Gemini model: {model}")
            
            limits = self.dict_gemini_limits[model]
            calls_limit = limits[0]
            tokens_limit = limits[1] if len(limits) > 1 else None
            
            counters = self.counters[family][model]
            
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
        else:
            raise ValueError(f"Family {family} not supported")
    
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
    
    async def call_llm(
        self,
        family: str,
        model: str,
        prompt: str,
        file_path: Optional[Union[str, Path]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """
        Make an LLM call with rate limiting.
        
        Args:
            family: LLM family ('Gemini' or 'Claude')
            model: Model name
            prompt: Text prompt
            file_path: Optional file to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_json: Whether to return the response as a JSON object
        Returns:
            Tuple of (success: bool, result: dict or error_message: str)
        """
        family = family.lower()
        
        if family == "claude":
            warnings.warn("Claude family has not been implemented yet", UserWarning)
            return False, "Claude family not implemented"
        
        if family != "gemini":
            return False, f"Unsupported family: {family}"
        
        estimated_tokens = self._estimate_tokens(prompt)
        
        # Check rate limits
        if not self._check_rate_limits(family, model, estimated_tokens):
            return False, "Rate limit exceeded"
        
        try:
            if file_path:
                result = await self._call_gemini_with_file(
                    model, prompt, file_path, max_tokens, temperature, return_json
                )
            else:
                result = await self._call_gemini_with_prompt(
                    model, prompt, max_tokens, temperature, return_json
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
        return_json: bool = False
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
                
                config = types.GenerateContentConfig()
                if max_tokens:
                    config.max_output_tokens = max_tokens
                if temperature is not None:
                    config.temperature = temperature
                
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=[
                        Part.from_bytes(
                            data=file_path_obj.read_bytes(),
                            mime_type=mime_type,
                        ),
                        prompt,
                    ],
                    config=config if max_tokens or temperature is not None else None
                )
                
                if not response.candidates[0].content.parts[0].text:
                    return False, "No response from Gemini"
                
                text_response = response.candidates[0].content.parts[0].text.strip()
                
                # Try to parse as JSON, fall back to plain text
                if return_json:
                    try:
                        if text_response.startswith("```json"):
                            text_response = text_response.removeprefix("```json").removesuffix("```").strip()
                        data_dict = json.loads(text_response)
                        return True, data_dict
                    except json.JSONDecodeError:
                        return True, {"text": text_response}
                else:
                    return True, {"text": text_response}
                
            except Exception as e:
                return False, str(e)
        
        return await asyncio.to_thread(sync_call)
    
    async def _call_gemini_with_prompt(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_json: bool = False
    ) -> Tuple[bool, Union[Dict[str, Any], str]]:
        """Call Gemini with text prompt only."""
        def sync_call():
            try:
                config = types.GenerateContentConfig()
                if max_tokens:
                    config.max_output_tokens = max_tokens
                if temperature is not None:
                    config.temperature = temperature
                
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=[prompt],
                    config=config if max_tokens or temperature is not None else None
                )
                
                if not response.candidates[0].content.parts[0].text:
                    return False, "No response from Gemini"
                
                text_response = response.candidates[0].content.parts[0].text.strip()
                
                # Try to parse as JSON, fall back to plain text
                if return_json:
                    try:
                        if text_response.startswith("```json"):
                            text_response = text_response.removeprefix("```json").removesuffix("```").strip()
                        data_dict = json.loads(text_response)
                        return True, data_dict
                    except json.JSONDecodeError:
                        return True, {"text": text_response}
                else:
                    return True, {"text": text_response}
                
            except Exception as e:
                return False, str(e)
        
        return await asyncio.to_thread(sync_call)
    
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
    
    def get_rate_limit_status(self, family: str, model: str) -> Dict[str, Any]:
        """Get current rate limit status for a model."""
        family = family.lower()
        if family == "gemini" and model in self.dict_gemini_limits:
            limits = self.dict_gemini_limits[model]
            counters = self.counters[family][model]
            
            # Clean old entries
            self._clean_old_entries(counters['calls'])
            if len(limits) > 1:
                self._clean_old_entries(counters['tokens'], is_tokens=True)
            
            status = {
                "calls_used": len(counters['calls']),
                "calls_limit": limits[0],
                "calls_remaining": limits[0] - len(counters['calls'])
            }
            
            if len(limits) > 1:  # Has token limit
                tokens_used = sum(token_count for timestamp, token_count in counters['tokens'])
                status.update({
                    "tokens_used": tokens_used,
                    "tokens_limit": limits[1],
                    "tokens_remaining": limits[1] - tokens_used
                })
            
            return status
        
        return {"error": f"Unknown model {model} for family {family}"}


# Convenience function to get the singleton instance
def get_llm_manager() -> LLMManager:
    """Get the singleton LLM manager instance."""
    return LLMManager()
