## Feature: GroqClient (free-tier LLM provider for production)

Description: Add a second LLMClient implementation, GroqClient, using
Groq's API (free tier, no credit card). This is the production LLM
client — it validates that the provider-agnostic architecture works: no
changes to RAGService are needed, only a different client passed at
construction time.

Requirements:

- New class GroqClient(LLMClient) in rag/llm_client.py, next to the
  existing AnthropicClient, implementing the same interface:
  - __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile")
    — stores a groq.Groq(api_key=api_key) client instance
  - generate(self, prompt: str) -> str — calls
    self._client.chat.completions.create(model=self.model,
    messages=[{"role": "user", "content": prompt}]) and returns
    response.choices[0].message.content
- Add "groq" to the project's dependency list (wherever "anthropic" is
  currently declared)
- No changes to RAGService, rag_service.py, or the LLMClient base class
  — GroqClient must be a drop-in replacement

Input/Output Example:

```python
client = GroqClient(api_key="gsk_...")
answer = client.generate("What is 2+2?")
# answer: "4" (a plain string, same return type as AnthropicClient.generate)
```

Edge Cases:

- Groq API returns an error (rate limit, invalid key, network failure) —
  let the exception propagate naturally (same behavior as
  AnthropicClient today — no special handling required, this matches
  existing project convention of not over-engineering error handling
  for provider clients)
- Empty prompt string — no special handling needed; pass it through as-is
  (mirrors AnthropicClient, which also has no guard for this)

Constraints:

- Groq free tier: 30 requests/minute, 6,000 tokens/minute, 14,400
  requests/day (org-level, not per-key) — fine for a low-traffic public
  demo, no special rate-limit handling needed in code for this pass
- Follow the exact same class shape as AnthropicClient (constructor
  signature, generate() signature) so it's a true drop-in replacement
- Tests must NOT call the real Groq API — mock/patch the groq.Groq
  client so tests are free and deterministic, same standard as every
  other test in this project

Generate:

1. rag/llm_client.py — add GroqClient(LLMClient), following
   AnthropicClient's existing pattern exactly
2. tests/test_llm_client.py — create if it doesn't exist (check first;
   AnthropicClient may not have dedicated tests yet, in which case add
   tests for both). Test GroqClient: mock the groq.Groq client, verify
   generate() builds the request with the right model/messages and
   returns the response text correctly.

Run: pytest tests/test_llm_client.py -v (all tests pass, zero real API calls)