from __future__ import annotations

from dataclasses import dataclass, field

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None


@dataclass
class LLMClient:
    """Light wrapper around the OpenAI client with deterministic settings."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if OpenAI is None:  # pragma: no cover - optional dependency
            raise ImportError("openai package is required for LLMClient")
        self._client = OpenAI()

    def complete(self, prompt: str) -> str:
        """Return completion text for the given prompt."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message["content"].strip()
