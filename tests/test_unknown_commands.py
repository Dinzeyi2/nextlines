import pytest

from parsing import ExecutionTemplate
from execution import NaturalLanguageExecutor
from errors import UnknownVerbError


def test_unknown_command_message():
    executor = NaturalLanguageExecutor()
    with pytest.raises(UnknownVerbError) as exc:
        executor.execute("gibberish command")
    assert "Sorry, I don't understand" in str(exc.value)


def test_suggestion_for_typo():
    executor = NaturalLanguageExecutor()
    with pytest.raises(UnknownVerbError) as exc:
        executor.execute("pritn 5")
    assert "Did you mean" in str(exc.value)



def test_map_to_code_unknown_template():
    executor = NaturalLanguageExecutor()
    template = ExecutionTemplate("foo", "nonexistent")
    result = executor.map_to_code(template, {})
    assert isinstance(result, dict)
    assert result["error"] == "UNSUPPORTED_TEMPLATE"
