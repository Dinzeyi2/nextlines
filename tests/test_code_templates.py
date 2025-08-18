import importlib.machinery
import pathlib
import types
import string

# Load codefull module
path = pathlib.Path(__file__).resolve().parents[1] / "codefull"
loader = importlib.machinery.SourceFileLoader("codefull_module", str(path))
codefull = types.ModuleType("codefull_module")
loader.exec_module(codefull)

PythonCodeGenerator = codefull.PythonCodeGenerator


def test_all_templates_generate_code():
    generator = PythonCodeGenerator()
    formatter = string.Formatter()

    for key, template in generator.code_templates.items():
        fields = [name for _, name, _, _ in formatter.parse(template) if name]
        params = {field: "0" for field in fields}
        code = generator.generate_code(key, **params)
        assert isinstance(code, str)
