import re
from dataclasses import dataclass
from typing import Callable, Dict, Pattern, Any


class ExecutionError(Exception):
    """Raised when a command cannot be executed."""


@dataclass
class ExecutionTemplate:
    pattern: Pattern[str]
    handler: str


class NaturalLanguageExecutor:
    """Very small natural language to Python executor.

    This is intentionally minimal: it supports setting variables and
    printing values or variables. The goal is to provide a concrete,
    well-tested example of how a natural language interpreter might be
    structured while keeping the code easy to understand.
    """

    def __init__(self) -> None:
        self.templates = [
            ExecutionTemplate(
                re.compile(r"set (?P<var>\w+) to (?P<value>.+)", re.IGNORECASE),
                "set_var",
            ),
            ExecutionTemplate(
                re.compile(r"print variable (?P<var>\w+)", re.IGNORECASE),
                "print_var",
            ),
            ExecutionTemplate(
                re.compile(r"print (?P<value>.+)", re.IGNORECASE),
                "print_value",
            ),
        ]
        self.variables: Dict[str, Any] = {}

    def execute(self, command: str) -> Any:
        """Execute a natural language command."""
        command = command.strip()
        for template in self.templates:
            match = template.pattern.fullmatch(command)
            if match:
                handler: Callable[..., Any] = getattr(self, template.handler)
                return handler(**match.groupdict())
        raise ExecutionError(f"Unknown command: {command}")

    # Handlers -------------------------------------------------------------
    def set_var(self, var: str, value: str) -> Any:
        """Set a variable to the evaluated value."""
        try:
            self.variables[var] = eval(value, {}, self.variables)
        except Exception as exc:
            raise ExecutionError(f"Invalid value {value}") from exc
        return self.variables[var]

    def print_var(self, var: str) -> Any:
        """Print the value of a variable."""
        if var not in self.variables:
            raise ExecutionError(f"Variable {var} is not defined")
        value = self.variables[var]
        print(value)
        return value

    def print_value(self, value: str) -> Any:
        """Evaluate a Python expression and print the result."""
        try:
            result = eval(value, {}, self.variables)
        except Exception as exc:
            raise ExecutionError(f"Invalid value {value}") from exc
        print(result)
        return result
