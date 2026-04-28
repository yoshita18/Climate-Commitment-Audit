"""
Concrete LLM provider implementations.

All providers implement LLMClient.complete() so the rest of the pipeline
is fully provider-agnostic.

Free providers
--------------
  GroqLLMClient       — Groq API (free tier, Llama 3.x / Mixtral); sign up at
                        https://console.groq.com  — no credit card required.
  HuggingFaceLLMClient — HuggingFace Inference API (free tier); sign up at
                        https://huggingface.co/settings/tokens

Paid (optional)
---------------
  AnthropicLLMClient  — Anthropic Claude (requires paid API key).
"""
from __future__ import annotations

import logging
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMClient

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────── Groq (FREE) ──────────────────

class GroqLLMClient(LLMClient):
    """
    Groq Cloud API — completely free tier.
    Recommended model: llama-3.1-8b-instant (fast) or llama-3.3-70b-versatile (quality).

    Get your free API key at: https://console.groq.com
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def get_provider_name(self) -> str:
        return "groq"

    def get_model_name(self) -> str:
        return self.model


# ────────────────────────────── HuggingFace Inference API (FREE) ──────────

class HuggingFaceLLMClient(LLMClient):
    """
    HuggingFace Inference API — free tier.

    Uses the chat_completion endpoint (OpenAI-compatible Messages API), which
    works reliably with all modern instruction-tuned models and avoids the
    ValueError that huggingface_hub>=0.20 raises when text_generation is
    called on a chat-format model.

    Recommended free models:
      - HuggingFaceH4/zephyr-7b-beta        (default — reliably free)
      - mistralai/Mistral-7B-Instruct-v0.3  (may require HF Pro)
      - microsoft/Phi-3-mini-4k-instruct

    Get your free token at: https://huggingface.co/settings/tokens
    """

    DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"

    def __init__(self, api_token: str, model: str = DEFAULT_MODEL):
        self.model = model
        self._api_token = api_token
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self._api_token)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        reraise=True,          # surface the real exception, not RetryError
    )
    def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Preferred: Messages API — works with all chat-format models
            response = client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.3,    # some models reject values < 0.05
            )
            return response.choices[0].message.content.strip()

        except Exception as chat_err:
            logger.warning(
                f"chat_completion failed ({chat_err}); "
                "falling back to text_generation"
            )
            # Fallback: plain text_generation (older models / text-only endpoints)
            full_prompt = (
                f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
                if system
                else prompt
            )
            result = client.text_generation(
                full_prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                return_full_text=False,   # only return generated tokens, not the prompt
            )
            # text_generation returns str by default (details=False)
            return (result if isinstance(result, str) else result.generated_text).strip()

    def get_provider_name(self) -> str:
        return "huggingface"

    def get_model_name(self) -> str:
        return self.model


# ─────────────────────────────── Anthropic (optional, paid) ───────────────

class AnthropicLLMClient(LLMClient):
    """
    Anthropic Claude — paid API (optional; kept for backward compatibility).
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_model_name(self) -> str:
        return self.model
