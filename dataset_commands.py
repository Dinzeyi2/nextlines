from __future__ import annotations

from typing import Optional


class DatasetCommandMixin:
    """Mixin providing dataset manager commands."""

    dataset_manager = None

    def _dataset_load(self, df_name: str, target: str) -> str:
        from dataset_management import DatasetManager, Schema  # type: ignore
        import pandas as pd  # noqa: F401

        df = self.context.get_variable(df_name)
        if df is None:
            return "\u2717 dataframe not found"
        numeric = [c for c in df.select_dtypes(include="number").columns if c != target]
        categorical = [
            c for c in df.select_dtypes(exclude="number").columns if c != target
        ]
        dtypes = {c: str(df[c].dtype) for c in df.columns}
        schema = Schema(numeric=numeric, categorical=categorical, target=target, dtypes=dtypes)
        self.dataset_manager = DatasetManager(schema)
        return "dataset manager loaded"

    def _dataset_validate(self, df_name: str) -> str:
        if self.dataset_manager is None:
            return "\u2717 dataset manager not loaded"
        df = self.context.get_variable(df_name)
        if df is None:
            return "\u2717 dataframe not found"
        report = self.dataset_manager.validate(df)
        return str(report)

    def _dataset_split(self, df_name: str) -> str:
        if self.dataset_manager is None:
            return "\u2717 dataset manager not loaded"
        df = self.context.get_variable(df_name)
        if df is None:
            return "\u2717 dataframe not found"
        train, val, test, *_ = self.dataset_manager.train_val_test_split(df)
        self.context.add_variable("train_df", train)
        self.context.add_variable("val_df", val)
        self.context.add_variable("test_df", test)
        return "dataset split into train_df, val_df, test_df"

    def handle_dataset_command(self, user_input: str) -> Optional[str]:
        lower_input = user_input.lower()
        if lower_input.startswith("dataset_load "):
            parts = user_input.split()
            if len(parts) >= 4:
                return self._dataset_load(parts[1], parts[3])
            return "\u2717 invalid dataset_load command"
        if lower_input.startswith("dataset_validate "):
            parts = user_input.split()
            if len(parts) >= 2:
                return self._dataset_validate(parts[1])
            return "\u2717 invalid dataset_validate command"
        if lower_input.startswith("dataset_split "):
            parts = user_input.split()
            if len(parts) >= 2:
                return self._dataset_split(parts[1])
            return "\u2717 invalid dataset_split command"
        return None
