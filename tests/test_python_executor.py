import pytest

from codefull import PythonExecutor


def test_benign_code():
    executor = PythonExecutor()
    result = executor.execute_code("a=1\nb=2\nprint(a+b)")
    assert result["success"] is True
    assert result["output"] == "3"
    assert result["locals"]["a"] == 1
    assert result["locals"]["b"] == 2


def test_disallowed_import():
    executor = PythonExecutor()
    result = executor.execute_code("import os")
    assert result["success"] is False
    assert "Disallowed" in result["error"]


def test_disallowed_eval():
    executor = PythonExecutor()
    result = executor.execute_code("eval('2+2')")
    assert result["success"] is False
    assert "Disallowed" in result["error"]


def test_timeout_loop():
    executor = PythonExecutor()
    result = executor.execute_code("while True:\n    pass")
    assert result["success"] is False


def test_memory_exhaustion():
    executor = PythonExecutor()
    result = executor.execute_code("a = [0]*10**8")
    assert result["success"] is False
