"""Provider-agnostic LLM adapter.

RAGService depends only on the LLMClient interface (duck-typed: any object
with .generate(prompt) -> str works). This is the ONLY module allowed to
import a specific provider SDK (anthropic), so swapping providers later
(e.g. a free-tier Groq/Gemini client) means adding a class here, not
touching rag_service.py.
"""

import logging

import anthropic
import groq

logger = logging.getLogger(__name__)


class LLMClient:
    """Base interface every LLM client must implement."""

    def generate(self, prompt: str) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: Full prompt text to send to the model.

        Returns:
            The model's text response.
        """
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """LLMClient implementation backed by the Anthropic API (Claude)."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key (load from .env, never hardcode).
            model: Anthropic model id to use for generation.
        """
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Send prompt to Claude and return the text of the response.

        Args:
            prompt: Full prompt text to send to the model.

        Returns:
            The text content of the model's reply.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GroqClient(LLMClient):
    """LLMClient implementation backed by the Groq API (free tier, no card)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        """Initialize the Groq client.

        Args:
            api_key: Groq API key (load from .env, never hardcode).
            model: Groq model id to use for generation.
        """
        self._client = groq.Groq(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Send prompt to Groq and return the text of the response.

        Args:
            prompt: Full prompt text to send to the model.

        Returns:
            The text content of the model's reply.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
