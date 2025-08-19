import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from codefull import NaturalLanguageExecutor, ExecutionError


def test_set_and_print_variable(capsys):
    exe = NaturalLanguageExecutor()
    assert exe.execute("set x to 5") == 5
    exe.execute("print variable x")
    captured = capsys.readouterr()
    assert captured.out.strip() == "5"


def test_print_value(capsys):
    exe = NaturalLanguageExecutor()
    exe.execute("print 2 + 3")
    captured = capsys.readouterr()
    assert captured.out.strip() == "5"


def test_unknown_command():
    exe = NaturalLanguageExecutor()
    with pytest.raises(ExecutionError):
        exe.execute("do something unknown")
