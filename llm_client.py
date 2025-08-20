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

    def complete(
        self,
        prompt: str,
        *,
        history: list[dict[str, str]] | None = None,
        session: dict[str, str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return completion text for the given prompt.

        Parameters
        ----------
        prompt:
            The latest user prompt.
        history:
            Optional list of prior messages represented as ``{"role", "content"}``
            dicts which will be sent before the current prompt.
        session:
            Optional dictionary of session variables to prepend as a system
            message so the model can reference state across requests.
        model:
            Override the model name for this request.
        temperature:
            Override the sampling temperature for this request.
        """
        messages: list[dict[str, str]] = []
        if session:
            session_text = "\n".join(f"{k}: {v}" for k, v in session.items())
            messages.append({"role": "system", "content": f"Session variables:\n{session_text}"})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return response.choices[0].message["content"].strip()
