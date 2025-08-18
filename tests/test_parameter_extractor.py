import importlib.machinery
import pathlib
import types
import pytest

# Load codefull module
path = pathlib.Path(__file__).resolve().parents[1] / "codefull"
loader = importlib.machinery.SourceFileLoader("codefull_module", str(path))
codefull = types.ModuleType("codefull_module")
loader.exec_module(codefull)

ParameterExtractor = codefull.ParameterExtractor
ExecutionContext = codefull.ExecutionContext
ParameterExtractionError = codefull.ParameterExtractionError


def make_extractor():
    ctx = ExecutionContext()
    ctx.add_variable("id", 42)
    return ParameterExtractor(ctx)


def test_number_word():
    extractor = make_extractor()
    assert extractor.extract_value("two") == 2


def test_ordinal_word():
    extractor = make_extractor()
    assert extractor.extract_value("second") == 2


def test_identifier_plural_synonym():
    extractor = make_extractor()
    assert extractor.extract_identifier("IDs") == "id"


def test_boolean_synonym():
    extractor = make_extractor()
    assert extractor.extract_value("yes") is True


def test_quoted_string():
    extractor = make_extractor()
    assert extractor.extract_value('"hello world"') == "hello world"


def test_unknown_identifier_error():
    extractor = make_extractor()
    with pytest.raises(ParameterExtractionError):
        extractor.extract_identifier("")
