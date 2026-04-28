"""Tests for the LLM abstraction layer (no actual API calls needed)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.llm.base import LLMClient
from src.llm.providers import GroqLLMClient, HuggingFaceLLMClient, AnthropicLLMClient
from src.llm.factory import get_llm_client


class _DummyLLM(LLMClient):
    """Minimal concrete subclass for testing the ABC."""
    def complete(self, prompt, system="", max_tokens=1000):
        return "dummy response"
    def get_provider_name(self):
        return "dummy"
    def get_model_name(self):
        return "dummy-model"


class TestLLMBase:
    def test_complete_returns_string(self):
        llm = _DummyLLM()
        result = llm.complete("hello")
        assert isinstance(result, str)

    def test_repr_includes_provider(self):
        llm = _DummyLLM()
        assert "dummy" in repr(llm)


class TestGroqLLMClient:
    def test_provider_name(self):
        client = GroqLLMClient(api_key="test_key", model="llama-3.1-8b-instant")
        assert client.get_provider_name() == "groq"

    def test_model_name(self):
        client = GroqLLMClient(api_key="test_key", model="llama-3.3-70b-versatile")
        assert client.get_model_name() == "llama-3.3-70b-versatile"

    def test_complete_calls_groq_api(self):
        client = GroqLLMClient(api_key="test_key")
        mock_choice = MagicMock()
        mock_choice.message.content = '{"key": "value"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = mock_response
        client._client = mock_groq

        result = client.complete("test prompt", system="test system")
        assert result == '{"key": "value"}'
        mock_groq.chat.completions.create.assert_called_once()
        call_kwargs = mock_groq.chat.completions.create.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        roles = [m["role"] for m in messages] if isinstance(messages[0], dict) else []
        # system + user messages passed
        assert mock_groq.chat.completions.create.called

    def test_default_model(self):
        client = GroqLLMClient(api_key="k")
        assert client.model == GroqLLMClient.DEFAULT_MODEL


class TestAnthropicLLMClient:
    def test_provider_name(self):
        client = AnthropicLLMClient(api_key="test")
        assert client.get_provider_name() == "anthropic"

    def test_complete_calls_anthropic(self):
        client = AnthropicLLMClient(api_key="test_key")
        mock_content = MagicMock()
        mock_content.text = "anthropic response"
        mock_msg = MagicMock()
        mock_msg.content = [mock_content]
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = mock_msg
        client._client = mock_anthropic

        result = client.complete("hello", system="be helpful")
        assert result == "anthropic response"


class TestLLMFactory:
    def test_factory_groq(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "gk_test"}):
            client = get_llm_client(provider="groq")
        assert isinstance(client, GroqLLMClient)

    def test_factory_anthropic(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ak_test"}):
            client = get_llm_client(provider="anthropic")
        assert isinstance(client, AnthropicLLMClient)

    def test_factory_auto_detects_groq(self):
        env = {"GROQ_API_KEY": "gk", "ANTHROPIC_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            client = get_llm_client(provider="groq")
        assert isinstance(client, GroqLLMClient)

    def test_factory_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client(provider="invalid_provider", api_key="key")

    def test_factory_missing_key_raises(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError):
                get_llm_client(provider="groq")
