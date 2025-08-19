import pytest

from execution import NaturalLanguageExecutor


def test_convert_condition_nested():
    executor = NaturalLanguageExecutor()
    expr = executor._convert_condition_to_python(
        "age is greater than 21 and (city equals 'NY' or city equals 'LA')"
    )
    assert expr == "age > 21 and (city == 'NY' or city == 'LA')"


def test_convert_condition_invalid():
    executor = NaturalLanguageExecutor()
    with pytest.raises(ValueError):
        executor._convert_condition_to_python(
            "age is greater than and city equals 'NY'"
        )
