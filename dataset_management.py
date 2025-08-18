"""Dataset management utilities for preprocessing, validation, and reproducible splits.

This module exposes a :class:`DatasetManager` that wraps a scikit-learn
``Pipeline`` and ``ColumnTransformer`` to apply leakage-safe transforms. The
manager fits the pipeline only on the training data and applies the resulting
transform to training, validation and test sets consistently. The fitted
pipeline can be serialised to disk for reuse.

Key features implemented:

* Multi-column transformers & scalers
* Robust & power transforms
* Imputation objects
* Rare-category handling & high-cardinality tricks
* Class imbalance utilities
* Feature selection
* Sampling/weights propagation
* Time-series CV extras
* Data validation & schema checks
* Reproducibility hooks
* I/O & dataset registry
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import ADASYN, SMOTE
except Exception:  # pragma: no cover
    ADASYN = SMOTE = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _hash_dataframe(df: pd.DataFrame) -> str:
    data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Schema:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    target: Optional[str] = None
    dtypes: Dict[str, str] = field(default_factory=dict)
    categories: Dict[str, List[Any]] = field(default_factory=dict)
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ImputationConfig:
    numeric_strategy: str = "mean"
    categorical_strategy: str = "most_frequent"
    knn: bool = False


@dataclass
class PipelineConfig:
    use_robust_scaler: bool = False
    use_quantile_transform: bool = False
    power_transform: Optional[str] = None
    select_k_best: Optional[int] = None
    variance_threshold: Optional[float] = None
    rfe_estimator: Optional[Any] = None
    categorical_encoder: str = "onehot"



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoder usable within scikit-learn pipelines."""

    def __init__(self) -> None:
        self.maps: Dict[str, Dict[Any, float]] = {}
        self.columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> 'FrequencyEncoder':
        df = pd.DataFrame(X)
        self.columns = df.columns.tolist()
        for col in self.columns:
            self.maps[col] = df[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            df[col] = df[col].map(self.maps[col]).fillna(0.0)
        return df.values


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder usable within scikit-learn pipelines."""

    def __init__(self) -> None:
        self.maps: Dict[str, Dict[Any, float]] = {}
        self.columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any) -> 'TargetEncoder':
        df = pd.DataFrame(X)
        self.columns = df.columns.tolist()
        y_series = pd.Series(y, index=df.index)
        for col in self.columns:
            self.maps[col] = y_series.groupby(df[col]).mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            df[col] = df[col].map(self.maps[col]).fillna(0.0)
        return df.values

# ---------------------------------------------------------------------------
# Main DatasetManager
# ---------------------------------------------------------------------------


class DatasetManager:
    """High level dataset manager implementing common ML data tasks."""

    def __init__(
        self,
        schema: Schema,
        imputation: ImputationConfig | None = None,
        pipeline_cfg: PipelineConfig | None = None,
        random_state: int | None = 42,
        registry_dir: str | Path = "data_registry",
    ) -> None:
        self.schema = schema
        self.imputation = imputation or ImputationConfig()
        self.pipeline_cfg = pipeline_cfg or PipelineConfig()
        self.random_state = random_state
        self.registry_dir = Path(registry_dir)
        self.pipeline: Pipeline | None = None
        self.freq_encoder: FrequencyEncoder | None = None
        self.target_encoder: TargetEncoder | None = None

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "missing_columns": [],
            "unexpected_columns": [],
            "dtype_mismatches": {},
            "invalid_categories": {},
            "out_of_range": {},
        }
        expected = set(self.schema.numeric + self.schema.categorical)
        if self.schema.target:
            expected.add(self.schema.target)
        report["missing_columns"] = sorted(expected - set(df.columns))
        report["unexpected_columns"] = sorted(set(df.columns) - expected)
        for col, dtype in self.schema.dtypes.items():
            if col in df and str(df[col].dtype) != dtype:
                report["dtype_mismatches"][col] = str(df[col].dtype)
        for col, cats in self.schema.categories.items():
            if col in df:
                invalid = sorted(set(df[col].dropna().unique()) - set(cats))
                if invalid:
                    report["invalid_categories"][col] = invalid
        for col, (low, high) in self.schema.ranges.items():
            if col in df:
                series = df[col].dropna()
                out = series[(series < low) | (series > high)]
                if not out.empty:
                    report["out_of_range"][col] = {"min": float(series.min()), "max": float(series.max())}
        return report

    def fit_frequency_encoder(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        cols = columns or self.schema.categorical
        self.freq_encoder = FrequencyEncoder().fit(df[cols])

    def apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.freq_encoder:
            raise RuntimeError("Frequency encoder not fit")
        df = df.copy()
        encoded = self.freq_encoder.transform(df[self.freq_encoder.columns])
        df[self.freq_encoder.columns] = pd.DataFrame(
            encoded, index=df.index, columns=self.freq_encoder.columns
        )
        return df

    def fit_target_encoder(
        self, df: pd.DataFrame, target: str | pd.Series, columns: Optional[List[str]] = None
    ) -> None:
        cols = columns or self.schema.categorical
        y = df[target] if isinstance(target, str) else target
        self.target_encoder = TargetEncoder().fit(df[cols], y)

    def apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.target_encoder:
            raise RuntimeError("Target encoder not fit")
        df = df.copy()
        encoded = self.target_encoder.transform(df[self.target_encoder.columns])
        df[self.target_encoder.columns] = pd.DataFrame(
            encoded, index=df.index, columns=self.target_encoder.columns
        )
        return df

    # ------------------------------------------------------------------
    # Pipeline creation
    # ------------------------------------------------------------------
    def _build_preprocess_pipeline(self) -> ColumnTransformer:
        transformers: List[Tuple[str, Any, List[str]]] = []

        num_steps: List[Tuple[str, Any]] = []
        if self.imputation.knn:
            num_steps.append(("imputer", KNNImputer()))
        else:
            num_steps.append(("imputer", SimpleImputer(strategy=self.imputation.numeric_strategy)))

        scaler: Any = RobustScaler() if self.pipeline_cfg.use_robust_scaler else StandardScaler()
        num_steps.append(("scaler", scaler))

        if self.pipeline_cfg.use_quantile_transform:
            num_steps.append(("quantile", QuantileTransformer(random_state=self.random_state)))
        if self.pipeline_cfg.power_transform:
            num_steps.append(("power", PowerTransformer(method=self.pipeline_cfg.power_transform)))

        num_pipe = Pipeline(num_steps)
        transformers.append(("num", num_pipe, self.schema.numeric))

        cat_steps: List[Tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=self.imputation.categorical_strategy, fill_value="missing"
                ),
            )
        ]
        enc = self.pipeline_cfg.categorical_encoder
        if enc == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)))
        elif enc == "frequency":
            cat_steps.append(("encoder", FrequencyEncoder()))
        elif enc == "target":
            cat_steps.append(("encoder", TargetEncoder()))
        else:  # pragma: no cover - defensive
            raise ValueError("Unknown categorical_encoder")
        cat_pipe = Pipeline(cat_steps)
        transformers.append(("cat", cat_pipe, self.schema.categorical))

        return ColumnTransformer(transformers)

    # ------------------------------------------------------------------
    def build_pipeline(self) -> Pipeline:
        preprocess = self._build_preprocess_pipeline()
        steps: List[Tuple[str, Any]] = [("preprocess", preprocess)]
        if self.pipeline_cfg.variance_threshold is not None:
            steps.append(("var", VarianceThreshold(self.pipeline_cfg.variance_threshold)))
        if self.pipeline_cfg.select_k_best is not None:
            steps.append(("skb", SelectKBest(k=self.pipeline_cfg.select_k_best)))
        if self.pipeline_cfg.rfe_estimator is not None:
            steps.append(("rfe", RFE(self.pipeline_cfg.rfe_estimator)))
        self.pipeline = Pipeline(steps)
        return self.pipeline

    # ------------------------------------------------------------------
    # Rare category handling
    # ------------------------------------------------------------------
    def handle_rare_categories(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        df = df.copy()
        for col in self.schema.categorical:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < threshold].index
            if len(rare) > 0:
                df[col] = df[col].replace(rare, "Other")
        return df

    # ------------------------------------------------------------------
    # Splitting utilities
    # ------------------------------------------------------------------
    def train_val_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: Optional[pd.Series] = None,
        weights: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        train_df, test_df, w_train, w_test = train_test_split(
            df,
            weights,
            test_size=test_size,
            stratify=stratify,
            random_state=self.random_state,
        )
        rel_val_size = val_size / (1 - test_size)
        strat_train = stratify.loc[train_df.index] if stratify is not None else None
        train_df, val_df, w_train, w_val = train_test_split(
            train_df,
            w_train,
            test_size=rel_val_size,
            stratify=strat_train,
            random_state=self.random_state,
        )
        return train_df, val_df, test_df, w_train, w_val, w_test

    # ------------------------------------------------------------------
    # Fitting and transforming
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> None:
        if not self.pipeline:
            self.build_pipeline()
        X = train_df[self.schema.numeric + self.schema.categorical]
        y = train_df[self.schema.target] if self.schema.target else None
        self.pipeline.fit(X, y)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been fit yet")
        X = df[self.schema.numeric + self.schema.categorical]
        return self.pipeline.transform(X)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save_pipeline(self, path: str | Path) -> None:
        import joblib
        if not self.pipeline:
            raise RuntimeError("Pipeline has not been fit yet")
        joblib.dump(self.pipeline, path)

    def load_pipeline(self, path: str | Path) -> None:
        import joblib
        self.pipeline = joblib.load(path)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def register_dataset(self, name: str, df: pd.DataFrame) -> str:
        hashval = _hash_dataframe(df)
        _save_dataframe(df, Path(self.registry_dir) / f"{name}-{hashval}.parquet")
        return hashval

    def load_registered(self, name: str, hashval: str) -> pd.DataFrame:
        return _load_dataframe(Path(self.registry_dir) / f"{name}-{hashval}.parquet")

    # ------------------------------------------------------------------
    # Class imbalance utilities
    # ------------------------------------------------------------------

    def apply_resampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "smote",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method == "smote" and SMOTE:
            sampler = SMOTE(random_state=self.random_state)
        elif method == "adasyn" and ADASYN:
            sampler = ADASYN(random_state=self.random_state)
        else:
            raise ValueError("Requested resampling method not available")
        return sampler.fit_resample(X, y)

    # ------------------------------------------------------------------
    # Time-series CV extras
    # ------------------------------------------------------------------
    def time_series_splits(self, n_splits: int, gap: int = 0, embargo: int = 0, expanding: bool = True, n_samples: Optional[int] = None):
        """Yield train/test indices for time-series validation."""
        n_samples = n_samples or 0
        if not n_samples:
            raise ValueError("n_samples must be provided")
        indices = np.arange(n_samples)
        fold_size = n_samples // (n_splits + 1)
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            if expanding:
                train_idx = indices[: max(0, train_end - gap)]
            else:
                start = fold_size * i
                train_idx = indices[start: max(start, train_end - gap)]
            test_start = train_end + embargo
            test_end = test_start + fold_size
            test_idx = indices[test_start:test_end]
            yield train_idx, test_idx



# ------------------------------------------------------------------
# Natural language templates
# ------------------------------------------------------------------

def run_template(dm: "DatasetManager", command: str, **kwargs):
    """Execute high level operations based on simple NL commands."""
    if command == "build preprocessing pipeline with robust scaler and power transform yeo-johnson":
        dm.pipeline_cfg = PipelineConfig(use_robust_scaler=True, power_transform="yeo-johnson")
        return dm.build_pipeline()
    if command == "fit pipeline on train_df":
        return dm.fit(kwargs["train_df"])
    if command == "transform val_df as X_val":
        return dm.transform(kwargs["val_df"])
    if command == "stratified split df by y with test size 0.2 and val size 0.1":
        return dm.train_val_test_split(
            kwargs["df"], stratify=kwargs["y"], test_size=0.2, val_size=0.1
        )
    if command == "handle rare categories under 1% in df":
        return dm.handle_rare_categories(kwargs["df"], threshold=0.01)
    if command == "apply smote to X and y":
        return dm.apply_resampling(kwargs["X"], kwargs["y"], method="smote")
    if command == "save pipeline to prep.pkl":
        return dm.save_pipeline("prep.pkl")
    if command == "load pipeline from prep.pkl":
        return dm.load_pipeline("prep.pkl")
    raise ValueError(f"Unknown command: {command}")


__all__ = ["DatasetManager", "Schema", "ImputationConfig", "PipelineConfig", "run_template"]

