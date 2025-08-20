"""scikit-learn templates for preprocessing and modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from errors import MissingParameterError, UnsupportedLibraryError


def _parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split("."):
        if p.isdigit():
            parts.append(int(p))
        else:
            break
    return tuple(parts)

try:  # pragma: no cover - optional dependency
    import sklearn  # type: ignore
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    from sklearn.preprocessing import PolynomialFeatures  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.model_selection import train_test_split  # noqa: F401
    from sklearn.decomposition import PCA  # noqa: F401
    from sklearn.metrics import accuracy_score  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    _version = _parse_version(sklearn.__version__)
    HAS_SKLEARN = _version >= (1, 0, 0)
except Exception:  # pragma: no cover - sklearn missing
    HAS_SKLEARN = False


@dataclass
class Template:
    pattern: str
    parameters: Dict[str, Any]
    code: str

    def generate(self, **kwargs: Any) -> str:
        if not HAS_SKLEARN:
            raise UnsupportedLibraryError("sklearn")
        missing = set(self.parameters) - set(kwargs)
        if missing:
            raise MissingParameterError(sorted(missing))
        return self.code.format(**kwargs)


TEMPLATES: Dict[str, Template] = {
    "standard_scaler": Template(
        pattern="scale columns {columns} with StandardScaler",
        parameters={"columns": "list[str]"},
        code="scaler = StandardScaler();\nscaled = scaler.fit_transform(df[{columns!r}])",
    ),
    "train_test_split": Template(
        pattern="split data into train and test sets",
        parameters={"test_size": "float", "random_state": "int"},
        code="X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})",
    ),
    "pca": Template(
        pattern="apply PCA with {n_components} components",
        parameters={"n_components": "int"},
        code="pca = PCA(n_components={n_components});\ntransformed = pca.fit_transform(X)",
    ),
    "polynomial_features": Template(
        pattern="generate polynomial features of degree {degree}",
        parameters={"degree": "int"},
        code="poly = PolynomialFeatures(degree={degree});\nfeatures = poly.fit_transform(X)",
    ),
    "logistic_regression": Template(
        pattern="fit logistic regression",
        parameters={},
        code="model = LogisticRegression();\nmodel.fit(X, y)",
    ),
    "random_forest": Template(
        pattern="fit random forest classifier",
        parameters={},
        code="model = RandomForestClassifier();\nmodel.fit(X, y)",
    ),
    "accuracy_score": Template(
        pattern="compute accuracy score",
        parameters={},
        code="score = accuracy_score(y_true, y_pred)",
    ),
}

__all__ = ["TEMPLATES", "Template", "HAS_SKLEARN"]
