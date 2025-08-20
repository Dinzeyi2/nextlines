from llm_parser import LLMParser


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt: str, **_: object) -> str:
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


class HistoryClient:
    def __init__(self):
        self.calls = []
        self.responses = [
            "analysis",
            "plan",
            "```python\nprint('hi'\n```",
            "```python\nprint('hi'\n```",
            "```python\nprint('hi')\n```",
        ]

    def complete(self, prompt, *, history=None, session=None, model=None, temperature=None):
        idx = len(self.calls)
        self.calls.append({"prompt": prompt, "history": history})
        return self.responses[idx]


def test_llm_parser_history_and_verification():
    client = HistoryClient()
    parser = LLMParser(client)
    history = [("user", "prev cmd"), ("assistant", "prev resp")]
    session = {"x": 1}
    code = parser.parse(
        "say hi",
        conversation_history=history,
        session_variables=session,
    )
    assert code == "print('hi')"
    first_history = client.calls[0]["history"]
    assert first_history[0] == {"role": "system", "content": "Session variables:\nx = 1"}
    assert first_history[1] == {"role": "user", "content": "prev cmd"}
    assert len(client.calls) == 5


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
