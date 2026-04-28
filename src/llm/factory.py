"""
Factory function to create the right LLMClient from env / settings.

Priority order (first non-empty key wins):
  1. GROQ_API_KEY        → GroqLLMClient        (FREE ✓)
  2. HF_API_TOKEN        → HuggingFaceLLMClient  (FREE ✓)
  3. ANTHROPIC_API_KEY   → AnthropicLLMClient    (paid)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMClient
from .providers import GroqLLMClient, HuggingFaceLLMClient, AnthropicLLMClient

logger = logging.getLogger(__name__)

_PROVIDER_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "huggingface": "HuggingFaceH4/zephyr-7b-beta",
    "anthropic": "claude-sonnet-4-6",
}


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """
    Return an LLMClient for the given provider.

    If *provider* is None, auto-detect from environment variables.
    """
    # Auto-detect provider
    if provider is None:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("HF_API_TOKEN"):
            provider = "huggingface"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise ValueError(
                "No LLM API key found. Set GROQ_API_KEY (free at console.groq.com), "
                "HF_API_TOKEN (free at huggingface.co), or ANTHROPIC_API_KEY."
            )

    provider = provider.lower()
    effective_model = model or _PROVIDER_MODELS.get(provider)

    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        logger.info(f"Using Groq LLM: {effective_model}")
        return GroqLLMClient(api_key=key, model=effective_model)

    elif provider == "huggingface":
        key = api_key or os.getenv("HF_API_TOKEN", "")
        if not key:
            raise ValueError("HF_API_TOKEN not set. Get a free token at https://huggingface.co/settings/tokens")
        logger.info(f"Using HuggingFace Inference API: {effective_model}")
        return HuggingFaceLLMClient(api_token=key, model=effective_model)

    elif provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        logger.info(f"Using Anthropic: {effective_model}")
        return AnthropicLLMClient(api_key=key, model=effective_model)

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Choose: groq, huggingface, anthropic")
