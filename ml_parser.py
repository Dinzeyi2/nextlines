from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from collections import Counter
import math

from logging_config import configure_logging
from monitoring import REQUEST_COUNT, REQUEST_LATENCY, init_monitoring

configure_logging()
init_monitoring(os.getenv("SENTRY_DSN"))
logger = logging.getLogger(__name__)


@dataclass
class MLCodeGenerator:
    """Simple ML-based code generator using bag-of-words cosine similarity."""

    queries: List[List[str]]
    codes: List[str]

    @classmethod
    def train(cls, dataset: List[dict]) -> "MLCodeGenerator":
        queries: List[List[str]] = []
        codes: List[str] = []
        for item in dataset:
            tokens = item["query"].lower().split()
            queries.append(tokens)
            codes.append(item["code"])
        return cls(queries=queries, codes=codes)

    def _vectorize(self, tokens: List[str]) -> Counter:
        return Counter(tokens)

    def _cosine(self, v1: Counter, v2: Counter) -> float:
        inter = set(v1) & set(v2)
        num = sum(v1[t] * v2[t] for t in inter)
        denom = (
            math.sqrt(sum(c * c for c in v1.values()))
            * math.sqrt(sum(c * c for c in v2.values()))
        )
        return num / denom if denom else 0.0

    def predict_with_score(self, text: str) -> Tuple[str, float]:
        REQUEST_COUNT.inc()
        with REQUEST_LATENCY.time():
            try:
                tokens = text.lower().split()
                vec = self._vectorize(tokens)
                best_score = -1.0
                best_code = ""
                for q_tokens, code in zip(self.queries, self.codes):
                    score = self._cosine(vec, self._vectorize(q_tokens))
                    if score > best_score:
                        best_score = score
                        best_code = code
                distance = 1.0 - best_score
                logger.info(
                    "prediction",
                    extra={
                        "request": text,
                        "generated_code": best_code,
                        "distance": distance,
                    },
                )
                return best_code, distance
            except Exception:
                logger.exception("prediction_failed", extra={"request": text})
                raise

    def predict(self, text: str) -> str:
        return self.predict_with_score(text)[0]

    def save(self, path: str | Path) -> None:
        data = {
            "queries": [" ".join(q) for q in self.queries],
            "codes": self.codes,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "MLCodeGenerator":
        data = json.loads(Path(path).read_text())
        queries = [q.split() for q in data["queries"]]
        return cls(queries=queries, codes=data["codes"])


def load_corpus(path: str | Path) -> List[dict]:
    return json.loads(Path(path).read_text())
