from llm_parser import LLMParser
from execution import NaturalLanguageExecutor


class DummyClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, prompt: str) -> str:
        return self.responses.pop(0)


def test_executor_uses_llm_fallback():
    executor = NaturalLanguageExecutor()
    executor.llm_parser = LLMParser(DummyClient([
        "analysis",
        "plan",
        "draft",
        "print('hi')",
    ]))
    # Avoid spawning subprocesses during tests
    def fake_execute(code: str) -> str:
        return code

    executor._execute_with_real_python = fake_execute  # type: ignore[attr-defined]
    result = executor.execute("tell me a joke")
    assert result == "print('hi')"
