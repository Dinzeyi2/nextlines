import pytest

fastapi = pytest.importorskip("fastapi")
pandas = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")

from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def csv_file(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "age,income,city,target\n"
        "25,50000,London,0\n"
        "30,,Paris,1\n"
        "35,70000,New York,0\n"
        "40,80000,Berlin,1\n"
    )
    return csv


def _post(client, command, session_id=None):
    payload = {"command": command}
    if session_id:
        payload["session_id"] = session_id
    return client.post("/execute", json=payload)


def test_full_workflow_success(client, csv_file):
    resp = _post(client, f"load csv file {csv_file} into df")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"]
    sid = data["session_id"]

    resp = _post(client, "clean remove rows with missing values from df", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "encode one hot encode column city in df", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "scale standardise column age of df", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "split df into train and test sets", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "build pipeline with standard scaler and logistic regression", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "fit pipeline on train", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "evaluate pipeline on test", sid)
    assert resp.status_code == 200
    result = resp.json()
    assert result["success"]
    assert result["output"]


def test_clean_without_df_fails(client):
    resp = _post(client, "clean remove rows with missing values from df")
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "df" in data["error"].lower()


def test_encode_unknown_column_fails(client, csv_file):
    sid = _post(client, f"load csv file {csv_file} into df").json()["session_id"]
    resp = _post(client, "encode one hot encode column missing in df", sid)
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "not found" in data["error"].lower()


def test_scale_unknown_column_fails(client, csv_file):
    sid = _post(client, f"load csv file {csv_file} into df").json()["session_id"]
    resp = _post(client, "scale standardise column missing of df", sid)
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "not found" in data["error"].lower()


def test_split_without_df_fails(client):
    resp = _post(client, "split df into train and test sets")
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "df" in data["error"].lower()


def test_fit_without_pipeline_fails(client, csv_file):
    sid = _post(client, f"load csv file {csv_file} into df").json()["session_id"]
    _post(client, "split df into train and test sets", sid)
    resp = _post(client, "fit pipeline on train", sid)
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "pipeline" in data["error"].lower()


def test_evaluate_without_pipeline_fails(client, csv_file):
    sid = _post(client, f"load csv file {csv_file} into df").json()["session_id"]
    _post(client, "split df into train and test sets", sid)
    resp = _post(client, "evaluate pipeline on test", sid)
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "pipeline" in data["error"].lower()


def test_dataset_manager_workflow(client, csv_file):
    sid = _post(client, f"load csv file {csv_file} into df").json()["session_id"]
    resp = _post(client, "dataset_load df target target", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "dataset_validate df", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]

    resp = _post(client, "dataset_split df", sid)
    assert resp.status_code == 200
    assert resp.json()["success"]


def test_dataset_validate_without_load_fails(client, csv_file):
    sid = _post(client, f"load csv file {csv_file} into df").json()["session_id"]
    resp = _post(client, "dataset_validate df", sid)
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
    assert "dataset manager" in data["error"].lower()
