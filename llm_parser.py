from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClientProtocol(Protocol):
    """Protocol describing the minimal LLM client interface used."""

    def complete(self, prompt: str) -> str:
        ...


FOUR_STEP_PROMPT = (
    "Use a four-step process: 1) analyze the task, 2) plan the solution, "
    "3) draft Python code, 4) return only the Python code."
)


@dataclass
class LLMParser:
    """Parser that delegates code generation to an LLM client."""

    client: LLMClientProtocol

    def parse(self, command: str) -> str:
        """Generate Python code from a natural language command."""
        step1 = self.client.complete(f"Step 1 - Analyze the task:\n{command}")
        step2 = self.client.complete(
            f"Step 2 - Plan the solution.\nTask: {command}\nAnalysis: {step1}"
        )
        step3 = self.client.complete(
            f"Step 3 - Draft Python code.\nPlan: {step2}"
        )
        final = self.client.complete(
            "Step 4 - Return ONLY the final Python code."\
            f"\n{step3}"
        )
        return self._extract_code(final)

    def _extract_code(self, text: str) -> str:
        """Extract Python code from an LLM response."""
        if "```" not in text:
            return text.strip()
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if stripped.startswith("python"):
                return stripped[len("python"):].strip()
            return stripped
        return text.strip()
