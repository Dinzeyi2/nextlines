import pathlib
import pytest

fastapi = pytest.importorskip("fastapi")
pandas = pytest.importorskip("pandas")
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_execute_and_reset(client, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    resp = client.post("/execute", json={"command": f"load csv file {csv} into df"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"]
    sid = data["session_id"]

    resp = client.post(
        "/execute", json={"command": "reset session", "session_id": sid}
    )
    assert resp.status_code == 200
    assert resp.json()["output"] == "Session reset."
