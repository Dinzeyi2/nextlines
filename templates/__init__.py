"""Template inventory for common ML tasks."""
from __future__ import annotations

from . import pandas as pandas
from . import sklearn as sklearn

# Inventory mapping high level tasks to idiomatic templates
TASK_INVENTORY = {
    "data_loading": {
        "read_csv": pandas.TEMPLATES.get("read_csv"),
    },
    "preprocessing": {
        "standard_scaler": sklearn.TEMPLATES.get("standard_scaler"),
    },
    "model_training": {
        "logistic_regression": sklearn.TEMPLATES.get("logistic_regression"),
    },
}

__all__ = ["TASK_INVENTORY", "pandas", "sklearn"]
