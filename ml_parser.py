from __future__ import annotations
import base64
import json
import pickle
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
    """Code generator using sentence-transformer embeddings or TF-IDF vectors."""

    model_name: str
    embeddings: List[List[float]]
    codes: List[str]
    tfidf_data: str | None = field(default=None, repr=False)
    model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer  # pragma: no cover - requires sklearn
            if self.tfidf_data is None:
                raise ValueError("TF-IDF vectorizer data missing")
            self.model = pickle.loads(base64.b64decode(self.tfidf_data))
        else:
            if SentenceTransformer is None:  # pragma: no cover - requires optional dep
                raise ImportError(
                    "sentence-transformers package is required for MLCodeGenerator"
                )
            self.model = SentenceTransformer(self.model_name)

    @classmethod
    def train(cls, dataset: List[dict], model_name: str = DEFAULT_MODEL_NAME) -> "MLCodeGenerator":
        queries = [item["query"] for item in dataset]
        codes = [item["code"] for item in dataset]
        if SentenceTransformer is not None:
            model = SentenceTransformer(model_name)
            embeddings = [list(vec) for vec in model.encode(queries)]
            return cls(model_name=model_name, embeddings=embeddings, codes=codes)
        else:  # pragma: no cover - requires sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            embeddings = vectorizer.fit_transform(queries).toarray().tolist()
            tfidf_data = base64.b64encode(pickle.dumps(vectorizer)).decode("utf-8")
            return cls(model_name="tfidf", embeddings=embeddings, codes=codes, tfidf_data=tfidf_data)

    def _cosine(self, v1: List[float], v2: List[float]) -> float:
        num = sum(a * b for a, b in zip(v1, v2))
        denom = math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2))
        return num / denom if denom else 0.0

    def predict_with_score(self, text: str) -> Tuple[str, float]:
        if self.model_name == "tfidf":  # pragma: no cover - requires sklearn
            vec = list(self.model.transform([text]).toarray()[0])
        else:
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
        if self.model_name == "tfidf" and self.tfidf_data is not None:
            data["tfidf_data"] = self.tfidf_data
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "MLCodeGenerator":
        data = json.loads(Path(path).read_text())
        return cls(
            model_name=data["model_name"],
            embeddings=data["embeddings"],
            codes=data["codes"],
            tfidf_data=data.get("tfidf_data"),
        )


def load_corpus(path: str | Path) -> List[dict]:
    return json.loads(Path(path).read_text())
