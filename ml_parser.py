from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import math

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - handled during runtime
    SentenceTransformer = None


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class MLCodeGenerator:
    """Code generator using sentence-transformer embeddings."""

    model_name: str
    embeddings: List[List[float]]
    codes: List[str]
    model: SentenceTransformer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if SentenceTransformer is None:  # pragma: no cover - requires optional dep
            raise ImportError("sentence-transformers package is required for MLCodeGenerator")
        self.model = SentenceTransformer(self.model_name)

    @classmethod
    def train(cls, dataset: List[dict], model_name: str = DEFAULT_MODEL_NAME) -> "MLCodeGenerator":
        if SentenceTransformer is None:  # pragma: no cover - requires optional dep
            raise ImportError("sentence-transformers package is required for training")
        model = SentenceTransformer(model_name)
        queries = [item["query"] for item in dataset]
        codes = [item["code"] for item in dataset]
        embeddings = [list(vec) for vec in model.encode(queries)]
        return cls(model_name=model_name, embeddings=embeddings, codes=codes)

    def _cosine(self, v1: List[float], v2: List[float]) -> float:
        num = sum(a * b for a, b in zip(v1, v2))
        denom = math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2))
        return num / denom if denom else 0.0

    def predict_with_score(self, text: str) -> Tuple[str, float]:
        vec = list(self.model.encode([text])[0])
        best_score = -1.0
        best_code = ""
        for emb, code in zip(self.embeddings, self.codes):
            score = self._cosine(vec, emb)
            if score > best_score:
                best_score = score
                best_code = code
        distance = 1.0 - best_score
        return best_code, distance

    def predict(self, text: str) -> str:
        return self.predict_with_score(text)[0]

    def save(self, path: str | Path) -> None:
        data = {
            "model_name": self.model_name,
            "embeddings": self.embeddings,
            "codes": self.codes,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "MLCodeGenerator":
        data = json.loads(Path(path).read_text())
        return cls(
            model_name=data["model_name"],
            embeddings=data["embeddings"],
            codes=data["codes"],
        )


def load_corpus(path: str | Path) -> List[dict]:
    return json.loads(Path(path).read_text())
