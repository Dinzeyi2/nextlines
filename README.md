# Natural Language Interpreter

A minimal example of interpreting simple natural-language commands into Python code.

## Usage

```python
from codefull import NaturalLanguageExecutor

exe = NaturalLanguageExecutor()
exe.execute("set x to 10")
exe.execute("print variable x")  # prints 10
exe.execute("print 2 + 3")        # prints 5
```

## Testing

Run the test suite with:

```bash
pytest
```
