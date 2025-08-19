import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api import app
from execution import NaturalLanguageExecutor


@pytest.fixture
def client():
    return TestClient(app)


def test_arbitrary_instruction_reaches_executor(client, monkeypatch):
    captured = {}

    def fake_execute(self, command: str):
        captured["command"] = command
        return "executor response"

    monkeypatch.setattr(NaturalLanguageExecutor, "execute", fake_execute)

    cmd = "sing a song about data"
    resp = client.post("/execute", json={"command": cmd})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"]
    assert data["output"] == "executor response"
    assert captured["command"] == cmd
