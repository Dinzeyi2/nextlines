from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple


class LLMClientProtocol(Protocol):
    """Protocol describing the minimal LLM client interface used."""

    def complete(
        self,
        prompt: str,
        *,
        history: List[Dict[str, str]] | None = None,
        session: Dict[str, str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        ...


FOUR_STEP_PROMPT = (
    "Use a four-step process: 1) analyze the task, 2) plan the solution, "
    "3) draft Python code, 4) return only the Python code."
)


@dataclass
class LLMParser:
    """Parser that delegates code generation to an LLM client."""

    client: LLMClientProtocol

    def parse(
        self,
        command: str,
        *,
        conversation_history: List[Tuple[str, str]] | None = None,
        session_variables: Dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate Python code from a natural language command."""

        history: List[Dict[str, str]] = []
        if session_variables:
            session_text = "\n".join(
                f"{k} = {repr(v)}" for k, v in session_variables.items()
            )
            history.append({"role": "system", "content": f"Session variables:\n{session_text}"})
        if conversation_history:
            for role, content in conversation_history:
                history.append({"role": role, "content": content})

        def _ask(prompt: str) -> str:
            response = self.client.complete(
                prompt,
                history=history,
                model=model,
                temperature=temperature,
            )
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})
            return response

        step1 = _ask(f"Step 1 - Analyze the task:\n{command}")
        step2 = _ask(
            f"Step 2 - Plan the solution.\nTask: {command}\nAnalysis: {step1}"
        )
        step3 = _ask(f"Step 3 - Draft Python code.\nPlan: {step2}")
        final = _ask(
            "Step 4 - Return ONLY the final Python code.\n" + step3
        )

        code = self._extract_code(final)

        try:
            ast.parse(code)
            return code
        except SyntaxError as exc:
            retry = _ask(
                "The previous code had a syntax error: "
                f"{exc}. Please fix it and return only valid Python code.\n{code}"
            )
            code = self._extract_code(retry)
            try:
                ast.parse(code)
            except SyntaxError as exc2:
                raise ValueError(
                    f"Generated code could not be parsed: {exc2.msg}"
                ) from exc2
            return code

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
