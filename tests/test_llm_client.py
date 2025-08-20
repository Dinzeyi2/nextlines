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
    model, messages, temp = client._client.chat.completions.last_args
    assert model == "test-model"
    assert temp == 0.5
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_llm_client_overrides(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", DummyOpenAI)
    client = LLMClient()
    hist = [{"role": "assistant", "content": "prev"}]
    sess = {"x": "1"}
    text = client.complete(
        "hi",
        history=hist,
        session=sess,
        model="big-model",
        temperature=0.9,
    )
    assert text == "ok"
    model, messages, temp = client._client.chat.completions.last_args
    assert model == "big-model"
    assert temp == 0.9
    assert messages[0]["role"] == "system"
    assert "x: 1" in messages[0]["content"]
    assert messages[1] == hist[0]
    assert messages[-1] == {"role": "user", "content": "hi"}
