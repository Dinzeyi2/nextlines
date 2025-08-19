"""scikit-learn templates for preprocessing and modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


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
    from sklearn.linear_model import LogisticRegression  # noqa: F401
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
        missing = set(self.parameters) - set(kwargs)
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        return self.code.format(**kwargs)


TEMPLATES: Dict[str, Template] = {
    "standard_scaler": Template(
        pattern="scale columns {columns} with StandardScaler",
        parameters={"columns": "list[str]"},
        code="scaler = StandardScaler();\nscaled = scaler.fit_transform(df[{columns!r}])",
    ),
    "logistic_regression": Template(
        pattern="fit logistic regression",
        parameters={},
        code="model = LogisticRegression();\nmodel.fit(X, y)",
    ),
}

__all__ = ["TEMPLATES", "Template", "HAS_SKLEARN"]
