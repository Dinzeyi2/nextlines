import pytest

from execution import PythonExecutor


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


def test_subprocess_failure(monkeypatch):
    executor = PythonExecutor()

    import multiprocessing

    class FailingProcess:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("subprocess failed")

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(multiprocessing, "Process", lambda *a, **k: FailingProcess())

    result = executor.execute_code("x = 1")
    assert result["success"] is False
    assert "subprocess failed" in result["error"]
    assert "x" not in result["locals"]
