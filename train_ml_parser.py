from __future__ import annotations
import json
import random
from pathlib import Path

from ml_parser import MLCodeGenerator, load_corpus, DEFAULT_MODEL_NAME

DATA_PATH = Path("data/nl2code_corpus.json")
MODEL_PATH = Path("models/ml_parser.json")
TEST_PATH = Path("data/test_corpus.json")


def main() -> None:
    dataset = load_corpus(DATA_PATH)
    random.Random(42).shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]
    parser = MLCodeGenerator.train(train_data, model_name=DEFAULT_MODEL_NAME)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    parser.save(MODEL_PATH)
    TEST_PATH.write_text(json.dumps(test_data, indent=2))


if __name__ == "__main__":
    main()
