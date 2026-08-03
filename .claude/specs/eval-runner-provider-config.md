## Feature: Configurable LLM provider for eval runner entry point

Description: Let the eval runner's manual entry point (__main__ block)
select which provider generates answers (Anthropic or Groq) via an
environment variable, instead of hardcoding one client. The judge stays
fixed on Anthropic/Haiku regardless of provider selected, so eval
numbers stay comparable across runs.

Requirements:

- New env var EVAL_PROVIDER, accepted values: "anthropic" or "groq".
  Defaults to "anthropic" if unset (keeps current behavior as default).
- Extract a small helper function, e.g.
  _build_generation_client(provider: str) -> LLMClient, that:
  - returns AnthropicClient(...) for "anthropic"
  - returns GroqClient(...) for "groq"
  - raises a clear ValueError for any other value, listing the valid
    options
  - raises a clear error if the relevant API key
    (ANTHROPIC_API_KEY / GROQ_API_KEY) is missing from the environment
    (same pattern as the existing missing-key handling already in
    __main__)
- The judge client is NOT configurable — always AnthropicClient with
  model="claude-haiku-4-5-20251001", regardless of EVAL_PROVIDER. This
  is intentional: keeps the evaluator constant so scores are comparable
  across different generation providers.
- results_path becomes provider-specific: evals/results_{provider}.json
  (e.g. evals/results_groq.json, evals/results_anthropic.json) so
  runs for different providers don't overwrite each other.
- Print which provider is being used at the start of the run (simple
  print/log statement), for clarity when watching the terminal.

Edge Cases:

- EVAL_PROVIDER set to an unsupported value (e.g. "gemini", typo) —
  raise ValueError immediately, before making any API calls, with a
  message listing valid options ("anthropic", "groq")
- EVAL_PROVIDER unset — default to "anthropic", no error
- Selected provider's API key missing from environment — clear error
  naming which env var is missing, raised before any API calls

Constraints:

- run_evals() itself is unchanged — it still accepts injected
  rag_service and judge_client, no new parameters needed there. All
  provider-selection logic lives in __main__ and the new helper function.
- _build_generation_client() must be a plain function (not buried
  inline in __main__) so it's unit-testable without real API calls.

Generate:

1. evals/eval_runner.py — add _build_generation_client(provider: str)
   -> LLMClient, and update __main__ to read EVAL_PROVIDER, call the
   helper, build the provider-specific results_path, and print which
   provider is running.
2. tests/test_eval_runner.py — add tests for
   _build_generation_client(): returns AnthropicClient for "anthropic",
   GroqClient for "groq", raises ValueError for an invalid value.
   (Mock/patch the SDK constructors the same way tests/test_llm_client.py
   already does — no real API calls.)

Run: pytest tests/test_eval_runner.py -v (all tests pass, zero real API calls)