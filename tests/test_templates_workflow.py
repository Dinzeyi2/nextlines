import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from templates import TASK_INVENTORY  # noqa: E402
from templates import pandas as pandas_templates  # noqa: E402
from templates import sklearn as sklearn_templates  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (pandas_templates.HAS_PANDAS and sklearn_templates.HAS_SKLEARN),
    reason="pandas and scikit-learn are required",
)


def test_inventory_contains_core_tasks():
    assert "read_csv" in TASK_INVENTORY["data_loading"]
    assert "standard_scaler" in TASK_INVENTORY["preprocessing"]
    assert "logistic_regression" in TASK_INVENTORY["model_training"]


def test_end_to_end_workflow(tmp_path):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    csv = tmp_path / "data.csv"
    csv.write_text("a,b,target\n1,2,0\n3,4,1\n")

    read_code = pandas_templates.TEMPLATES["read_csv"].generate(path=str(csv), sep=",")
    df = eval(read_code, {"pd": pd})
    assert list(df.columns) == ["a", "b", "target"]
    scale_code = sklearn_templates.TEMPLATES["standard_scaler"].generate(columns=["a", "b"])
    ns = {"df": df, "StandardScaler": StandardScaler}
    exec(scale_code, ns)
    scaled = ns["scaled"]
    assert scaled.shape == (2, 2)

    train_code = sklearn_templates.TEMPLATES["logistic_regression"].generate()
    ns = {"X": scaled, "y": df["target"], "LogisticRegression": LogisticRegression}
    exec(train_code, ns)
    model = ns["model"]
    assert hasattr(model, "predict")
