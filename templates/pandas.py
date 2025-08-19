"""Pandas templates for data loading operations with version guards."""
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
    import pandas as pd  # type: ignore
    _version = _parse_version(pd.__version__)
    HAS_PANDAS = _version >= (1, 0, 0)
except Exception:  # pragma: no cover - pandas missing
    pd = None  # type: ignore
    HAS_PANDAS = False


@dataclass
class Template:
    """Simple code generation template."""

    pattern: str
    parameters: Dict[str, Any]
    code: str

    def generate(self, **kwargs: Any) -> str:
        missing = set(self.parameters) - set(kwargs)
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        return self.code.format(**kwargs)


TEMPLATES: Dict[str, Template] = {
    "read_csv": Template(
        pattern="load CSV file at {path}",
        parameters={"path": "str", "sep": "str"},
        code="pd.read_csv({path!r}, sep={sep!r})",
    ),
    "dropna": Template(
        pattern="drop missing values",
        parameters={},
        code="df.dropna()",
    ),
    "merge": Template(
        pattern="merge df1 with df2 on {on}",
        parameters={"on": "str"},
        code="df1.merge(df2, on={on!r})",
    ),
    "groupby": Template(
        pattern="group by {column} with {agg}",
        parameters={"column": "str", "agg": "str"},
        code="df.groupby({column!r}).{agg}()",
    ),
    "one_hot_encode": Template(
        pattern="one-hot encode columns {columns}",
        parameters={"columns": "list[str]"},
        code="pd.get_dummies(df, columns={columns!r})",
    ),
}

__all__ = ["TEMPLATES", "Template", "HAS_PANDAS"]
