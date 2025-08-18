import importlib.machinery
import pathlib
import types
import pytest

path = pathlib.Path(__file__).resolve().parents[1] / "codefull"
loader = importlib.machinery.SourceFileLoader("codefull_module", str(path))
codefull = types.ModuleType("codefull_module")
loader.exec_module(codefull)
PythonExecutor = codefull.PythonExecutor


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
