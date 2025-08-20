import llm_client
from llm_client import LLMClient


class DummyCompletions:
    def __init__(self):
        self.last_args = None

    def create(self, model, messages, temperature):
        self.last_args = (model, messages, temperature)
        class Resp:
            choices = [type('obj', (), {'message': {'content': 'ok'}})]
        return Resp()


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyOpenAI:
    def __init__(self):
        self.chat = DummyChat()


def test_llm_client_configuration(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", DummyOpenAI)
    client = LLMClient(model="test-model", temperature=0.5)
    text = client.complete("hi")
    assert text == "ok"
    assert client._client.chat.completions.last_args[0] == "test-model"
    assert client._client.chat.completions.last_args[2] == 0.5
