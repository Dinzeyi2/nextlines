import pytest


def test_apply_resampling_requires_imblearn(monkeypatch):
    dm_mod = pytest.importorskip("dataset_management")
    monkeypatch.setattr(dm_mod, "SMOTE", None)
    schema = dm_mod.Schema(numeric=["a"], target="b")
    dm = dm_mod.DatasetManager(schema)
    X = [[1], [2]]
    y = [0, 1]
    with pytest.raises(ValueError, match="Install imblearn to use SMOTE/ADASYN"):
        dm.apply_resampling(X, y, method="smote")


def test_api_returns_error_on_value_error(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import api

    client = TestClient(api.app)

    def fake_execute(self, cmd):
        raise ValueError("Install imblearn to use SMOTE/ADASYN")

    monkeypatch.setattr(api.NaturalLanguageExecutor, "execute", fake_execute)
    resp = client.post("/execute", json={"command": "load data"})
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert data["error"] == "Install imblearn to use SMOTE/ADASYN"
