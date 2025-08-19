import importlib.machinery
import pathlib
import types

path = pathlib.Path(__file__).resolve().parents[1] / "codefull"
loader = importlib.machinery.SourceFileLoader("codefull_module", str(path))
codefull = types.ModuleType("codefull_module")
loader.exec_module(codefull)
NaturalLanguageExecutor = codefull.NaturalLanguageExecutor
ExecutionTemplate = codefull.ExecutionTemplate


def test_unknown_command_message():
    executor = NaturalLanguageExecutor()
    result = executor.execute("gibberish command")
    assert "Sorry, I don't understand" in result


def test_map_to_code_unknown_template():
    executor = NaturalLanguageExecutor()
    template = ExecutionTemplate("foo", "nonexistent")
    result = executor.map_to_code(template, {})
    assert isinstance(result, dict)
    assert result["error"] == "UNSUPPORTED_TEMPLATE"
