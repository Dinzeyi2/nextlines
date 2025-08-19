"""Template inventory for common ML tasks."""
from __future__ import annotations

from . import pandas as pandas
from . import sklearn as sklearn

# Inventory mapping high level tasks to idiomatic templates
TASK_INVENTORY = {
    "data_loading": {
        "read_csv": pandas.TEMPLATES.get("read_csv"),
    },
    "dataframe_operations": {
        "dropna": pandas.TEMPLATES.get("dropna"),
        "merge": pandas.TEMPLATES.get("merge"),
        "groupby": pandas.TEMPLATES.get("groupby"),
    },
    "preprocessing": {
        "standard_scaler": sklearn.TEMPLATES.get("standard_scaler"),
        "train_test_split": sklearn.TEMPLATES.get("train_test_split"),
        "pca": sklearn.TEMPLATES.get("pca"),
    },
    "model_training": {
        "logistic_regression": sklearn.TEMPLATES.get("logistic_regression"),
        "random_forest": sklearn.TEMPLATES.get("random_forest"),
    },
    "feature_engineering": {
        "one_hot_encode": pandas.TEMPLATES.get("one_hot_encode"),
        "polynomial_features": sklearn.TEMPLATES.get("polynomial_features"),
    },
    "model_evaluation": {
        "accuracy_score": sklearn.TEMPLATES.get("accuracy_score"),
    },
}

__all__ = ["TASK_INVENTORY", "pandas", "sklearn"]
