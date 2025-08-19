import pytest

pandas = pytest.importorskip("pandas")

from dataset_management import DatasetManager, Schema


def test_time_series_splits_infers_length_from_df():
    df = pandas.DataFrame({"a": range(10)})
    dm = DatasetManager(Schema())
    splits = list(dm.time_series_splits(n_splits=3, df_or_length=df))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(test_idx) == 2
        assert max(test_idx) < len(df)


def test_label_time_series_folds_uses_inference():
    df = pandas.DataFrame({"a": range(10)})
    dm = DatasetManager(Schema())
    labelled = dm.label_time_series_folds(df, n_splits=3)
    assert "fold" in labelled.columns
    assert set(labelled["fold"].unique()) == {-1, 0, 1, 2}
