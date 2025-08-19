import pytest

pandas = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")

from dataset_management import DatasetManager, Schema, PipelineConfig


def _make_df():
    import pandas as pd
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60] * 20,
        "income": [10, 20, 30, 40, 50] * 20,
        "city": ["A", "B", "A", "B", "C"] * 20,
        "target": [0, 1, 0, 1, 0] * 20,
    })
    return df

CONFIGS = [
    {"use_robust_scaler": False, "use_quantile_transform": False},
    {"use_robust_scaler": True, "use_quantile_transform": False},
    {"use_robust_scaler": False, "use_quantile_transform": True},
    {"use_robust_scaler": True, "use_quantile_transform": True},
    {"power_transform": "yeo-johnson"},
    {"select_k_best": 1},
    {"variance_threshold": 0.0},
    {"use_robust_scaler": True, "select_k_best": 1},
    {"use_quantile_transform": True, "power_transform": "yeo-johnson"},
    {"use_robust_scaler": True, "variance_threshold": 0.0},
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_end_to_end(cfg, tmp_path):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    df = _make_df()
    schema = Schema(
        numeric=["age", "income"],
        categorical=["city"],
        target="target",
        dtypes={"age": "int64", "income": "int64", "city": "object", "target": "int64"},
        categories={"city": ["A", "B", "C"]},
    )
    dm = DatasetManager(schema, pipeline_cfg=PipelineConfig(**cfg))
    df_clean, report = dm.validate_and_clean(df, coerce_dtypes=True)
    assert not report["missing_columns"]
    train, val, test, *_ = dm.train_val_test_split(df_clean)
    dm.fit(train)
    Xt = dm.transform(val)
    model = LogisticRegression()
    model.fit(Xt, val["target"])
    preds = model.predict(dm.transform(test))
    assert preds.shape[0] == test.shape[0]
    path = tmp_path / "pipe.joblib"
    dm.save_pipeline(path)
    dm.load_pipeline(path)
    Xt2 = dm.transform(test)
    assert Xt2.shape[0] == test.shape[0]
