from codefull import ExecutionTemplate, NaturalLanguageExecutor


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
