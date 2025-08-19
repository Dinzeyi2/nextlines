from llm_parser import LLMParser


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.responses.pop(0)


def test_llm_parser_returns_final_code_from_backticks():
    client = DummyClient([
        "analysis",
        "plan",
        "draft",
        "```python\nprint('hi')\n```",
    ])
    parser = LLMParser(client)
    result = parser.parse("print hello")
    assert result == "print('hi')"
    assert len(client.calls) == 4


def test_llm_parser_returns_plain_code():
    client = DummyClient([
        "analysis",
        "plan",
        "draft",
        "print('hi')",
    ])
    parser = LLMParser(client)
    result = parser.parse("print hello")
    assert result == "print('hi')"
    assert len(client.calls) == 4
