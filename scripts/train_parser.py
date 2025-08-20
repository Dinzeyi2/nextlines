#!/usr/bin/env python
"""Train the ML parser model from a dataset of command/code pairs."""
from __future__ import annotations

import argparse

from ml_parser import DEFAULT_MODEL_NAME, MLCodeGenerator, load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML code parser")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/commands.json",
        help="Path to JSON dataset with 'query' and 'code' entries",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ml_parser.json",
        help="Where to store the trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model to use",
    )
    args = parser.parse_args()

    dataset = load_corpus(args.dataset)
    model = MLCodeGenerator.train(dataset, model_name=args.model_name)
    model.save(args.output)

    # Evaluate training accuracy
    total = len(dataset)
    correct = sum(1 for item in dataset if model.predict(item["query"]) == item["code"])
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
