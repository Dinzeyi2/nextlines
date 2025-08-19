import pathlib, sys
import pytest
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from templates import TASK_INVENTORY
from templates import pandas as pandas_templates
from templates import sklearn as sklearn_templates

pytestmark = pytest.mark.skipif(
    not (pandas_templates.HAS_PANDAS and sklearn_templates.HAS_SKLEARN),
    reason="pandas and scikit-learn are required",
)

def test_inventory_contains_core_tasks():
    assert "read_csv" in TASK_INVENTORY["data_loading"]
    assert "dropna" in TASK_INVENTORY["dataframe_operations"]
    assert "merge" in TASK_INVENTORY["dataframe_operations"]
    assert "groupby" in TASK_INVENTORY["dataframe_operations"]
    assert "standard_scaler" in TASK_INVENTORY["preprocessing"]
    assert "train_test_split" in TASK_INVENTORY["preprocessing"]
    assert "pca" in TASK_INVENTORY["preprocessing"]
    assert "logistic_regression" in TASK_INVENTORY["model_training"]
    assert "random_forest" in TASK_INVENTORY["model_training"]


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


def test_pandas_templates():
    import pandas as pd

    df = pd.DataFrame({"a": [1, None, 3]})
    drop_code = pandas_templates.TEMPLATES["dropna"].generate()
    cleaned = eval(drop_code, {"df": df})
    assert cleaned.isna().sum().sum() == 0

    df1 = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 3], "val2": [7, 8]})
    merge_code = pandas_templates.TEMPLATES["merge"].generate(on="id")
    merged = eval(merge_code, {"df1": df1, "df2": df2})
    assert merged.shape == (1, 3)

    df_group = pd.DataFrame({"key": ["x", "x", "y"], "val": [1, 2, 3]})
    group_code = pandas_templates.TEMPLATES["groupby"].generate(column="key", agg="sum")
    grouped = eval(group_code, {"df": df_group})
    assert grouped.loc["x", "val"] == 3


def test_sklearn_templates():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    split_code = sklearn_templates.TEMPLATES["train_test_split"].generate(test_size=0.5, random_state=0)
    ns = {"X": X, "y": y, "train_test_split": train_test_split}
    exec(split_code, ns)
    assert ns["X_train"].shape[0] == 2

    pca_code = sklearn_templates.TEMPLATES["pca"].generate(n_components=1)
    ns_pca = {"X": ns["X_train"], "PCA": PCA}
    exec(pca_code, ns_pca)
    transformed = ns_pca["transformed"]
    assert transformed.shape == (2, 1)

    rf_code = sklearn_templates.TEMPLATES["random_forest"].generate()
    ns_rf = {"X": ns["X_train"], "y": ns["y_train"], "RandomForestClassifier": RandomForestClassifier}
    exec(rf_code, ns_rf)
    model = ns_rf["model"]
    assert hasattr(model, "predict")
