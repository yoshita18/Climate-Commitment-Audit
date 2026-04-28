"""
Abstract LLM client interface.

Supports any provider (Groq, HuggingFace, Anthropic) through a common
complete() interface so audit/eval classes are provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMMessage:
    role: str        # "system" | "user" | "assistant"
    content: str


class LLMClient(ABC):
    """Minimal interface for LLM text completion used throughout the audit pipeline."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate a completion for *prompt* with an optional *system* message.

        Returns the raw text response (stripped).
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.get_provider_name()}, model={self.get_model_name()})"
