import importlib.machinery
import pathlib
import types

# Load codefull module
path = pathlib.Path(__file__).resolve().parents[1] / "codefull"
loader = importlib.machinery.SourceFileLoader("codefull_module", str(path))
codefull = types.ModuleType("codefull_module")
loader.exec_module(codefull)

ExecutionTemplate = codefull.ExecutionTemplate
ParameterType = codefull.ParameterType
NaturalLanguageExecutor = codefull.NaturalLanguageExecutor


def build_executor():
    nl = NaturalLanguageExecutor()
    nl.templates = [
        ExecutionTemplate(
            "set {var} to {value}",
            "execute_assignment",
            parameters={"var": ParameterType.IDENTIFIER, "value": ParameterType.VALUE},
        ),
        ExecutionTemplate(
            "add {value} to {var}",
            "execute_add_to_var",
            parameters={"var": ParameterType.IDENTIFIER, "value": ParameterType.VALUE},
        ),
        ExecutionTemplate(
            "print {value}",
            "execute_print",
            parameters={"value": ParameterType.IDENTIFIER},
        ),
    ]
    return nl


def test_natural_language_roundtrip():
    nl = build_executor()
    assert "Variables" in nl.execute("set total to 1")
    assert "total = 3" in nl.execute("add 2 to total")
    output = nl.execute("print total")
    assert "Output: 3" in output


def test_direct_executor_path():
    nl = build_executor()
    codefull._global_executor = nl
    result = codefull.nl("set value to 7")
    assert "value = 7" in result
    codefull.reset_context()
