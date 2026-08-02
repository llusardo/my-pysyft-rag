"""Tests for rag.llm_client: AnthropicClient and GroqClient.

Both providers' SDK clients are mocked -- zero real API calls, zero cost,
fully deterministic.
"""

from unittest.mock import MagicMock

import rag.llm_client as llm_client
from rag.llm_client import AnthropicClient, GroqClient


def test_anthropic_client_generate_builds_request_and_returns_text(monkeypatch):
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="4")]
    mock_instance = MagicMock()
    mock_instance.messages.create.return_value = fake_response
    mock_anthropic_class = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(llm_client.anthropic, "Anthropic", mock_anthropic_class)

    client = AnthropicClient(api_key="test-key")
    result = client.generate("What is 2+2?")

    assert result == "4"
    mock_anthropic_class.assert_called_once_with(api_key="test-key")
    mock_instance.messages.create.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )


def test_anthropic_client_uses_custom_model(monkeypatch):
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="ok")]
    mock_instance = MagicMock()
    mock_instance.messages.create.return_value = fake_response
    monkeypatch.setattr(llm_client.anthropic, "Anthropic", MagicMock(return_value=mock_instance))

    client = AnthropicClient(api_key="test-key", model="claude-opus-5")
    client.generate("hi")

    assert mock_instance.messages.create.call_args.kwargs["model"] == "claude-opus-5"


def test_anthropic_client_passes_empty_prompt_through_unchanged(monkeypatch):
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="")]
    mock_instance = MagicMock()
    mock_instance.messages.create.return_value = fake_response
    monkeypatch.setattr(llm_client.anthropic, "Anthropic", MagicMock(return_value=mock_instance))

    client = AnthropicClient(api_key="test-key")
    client.generate("")

    assert mock_instance.messages.create.call_args.kwargs["messages"] == [{"role": "user", "content": ""}]


def test_groq_client_generate_builds_request_and_returns_text(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="4"))]
    mock_instance = MagicMock()
    mock_instance.chat.completions.create.return_value = fake_response
    mock_groq_class = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(llm_client.groq, "Groq", mock_groq_class)

    client = GroqClient(api_key="gsk_test")
    result = client.generate("What is 2+2?")

    assert result == "4"
    mock_groq_class.assert_called_once_with(api_key="gsk_test")
    mock_instance.chat.completions.create.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )


def test_groq_client_uses_custom_model(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    mock_instance = MagicMock()
    mock_instance.chat.completions.create.return_value = fake_response
    monkeypatch.setattr(llm_client.groq, "Groq", MagicMock(return_value=mock_instance))

    client = GroqClient(api_key="gsk_test", model="llama-3.1-8b-instant")
    client.generate("hi")

    assert mock_instance.chat.completions.create.call_args.kwargs["model"] == "llama-3.1-8b-instant"


def test_groq_client_passes_empty_prompt_through_unchanged(monkeypatch):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=""))]
    mock_instance = MagicMock()
    mock_instance.chat.completions.create.return_value = fake_response
    monkeypatch.setattr(llm_client.groq, "Groq", MagicMock(return_value=mock_instance))

    client = GroqClient(api_key="gsk_test")
    client.generate("")

    assert mock_instance.chat.completions.create.call_args.kwargs["messages"] == [{"role": "user", "content": ""}]
