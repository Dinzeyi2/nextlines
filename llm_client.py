from __future__ import annotations

from dataclasses import dataclass, field
import os

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None


def _default_model() -> str:
    """Default model name configurable via ``LLM_MODEL`` env var."""
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def _default_temperature() -> float:
    """Default temperature configurable via ``LLM_TEMPERATURE`` env var."""
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.0"))
    except ValueError:  # pragma: no cover - invalid env value
        return 0.0


@dataclass
class LLMClient:
    """Light wrapper around the OpenAI client with configurable settings."""

    model: str = field(default_factory=_default_model)
    temperature: float = field(default_factory=_default_temperature)
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
