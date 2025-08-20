import ast
from execution import PythonCodeGenerator

def assert_valid(code: str):
    """Helper to ensure generated code is syntactically valid."""
    ast.parse(code)

def test_generate_for_loop_executes():
    gen = PythonCodeGenerator()
    code = gen.generate_code("for_loop", var="i", collection="range(3)", action="result.append(i)")
    assert_valid(code)
    ns = {"result": []}
    exec(code, ns)
    assert ns["result"] == [0, 1, 2]

def test_generate_if_statement():
    gen = PythonCodeGenerator()
    code = gen.generate_code("if_statement", condition="x > 0", action="y = 1")
    assert_valid(code)
    ns = {"x": 5}
    exec(code, ns)
    assert ns["y"] == 1

def test_generate_class_definition():
    gen = PythonCodeGenerator()
    body = "def __init__(self):\n        self.x = 1"
    code = gen.generate_code("class_definition", name="Foo", body=body)
    assert_valid(code)
    ns = {}
    exec(code, ns)
    obj = ns["Foo"]()
    assert obj.x == 1

def test_generate_import():
    gen = PythonCodeGenerator()
    code = gen.generate_code("import", module="math")
    assert_valid(code)
    ns = {}
    exec(code, ns)
    assert "math" in ns
