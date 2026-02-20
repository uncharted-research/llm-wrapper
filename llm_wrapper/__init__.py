"""
LLM Wrapper - A unified interface for LLM APIs with rate limiting.

This package provides a simple, unified interface for calling various LLM APIs
(currently supporting Gemini, with Claude support planned) while handling
rate limiting automatically.
"""

from .llm import LLMManager, get_llm_manager

__version__ = "0.1.1"
__all__ = ["LLMManager", "get_llm_manager"]
