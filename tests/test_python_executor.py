import pytest

from execution import LLMExecutor, PythonExecutor


class DummyParser:
    def __init__(self, code: str):
        self.code = code

    def parse(self, command: str) -> str:  # pragma: no cover - trivial
        return self.code


def _run(code: str):
    parser = DummyParser(code)
    executor = LLMExecutor(parser)
    return executor.execute("cmd")


def test_success_execution():
    result = _run("print('hi')")
    assert result["success"] is True
    assert result["output"] == "hi"


def test_syntax_error():
    result = _run("def bad:")
    assert result["success"] is False
    assert "expected" in result["error"].lower()


def test_runtime_error():
    result = _run("1/0")
    assert result["success"] is False
    assert "division" in result["error"].lower()

