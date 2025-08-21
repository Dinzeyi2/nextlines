#!/usr/bin/env python
"""Train the ML parser model from a dataset of command/code pairs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest matches to save",
    )
    parser.add_argument(
        "--neighbors",
        type=str,
        default=None,
        help="Optional JSON file to store nearest matches for debugging",
    )
    args = parser.parse_args()

    dataset = load_corpus(args.dataset)
    random.shuffle(dataset)
    split = int(len(dataset) * args.test_split)
    test, train = dataset[:split], dataset[split:]
    if not train:
        raise ValueError("Training set is empty; adjust --test-split")
    model = MLCodeGenerator.train(train, model_name=args.model_name)
    model.save(args.output)

    # Evaluate on held-out set
    total = len(test)
    correct_top1 = 0
    correct_topk = 0
    neighbors_dump = []
    for item in test:
        pred = model.predict(item["query"])
        if pred == item["code"]:
            correct_top1 += 1
        nearest = model.nearest_neighbors(item["query"], args.top_k)
        if item["code"] in [c for c, _ in nearest]:
            correct_topk += 1
        if args.neighbors:
            neighbors_dump.append(
                {
                    "query": item["query"],
                    "expected": item["code"],
                    "neighbors": [{"code": c, "distance": d} for c, d in nearest],
                }
            )
    accuracy = correct_top1 / total if total else 0.0
    topk_acc = correct_topk / total if total else 0.0
    print(f"Accuracy: {accuracy:.2%} ({correct_top1}/{total})")
    print(f"Top-{args.top_k} Accuracy: {topk_acc:.2%} ({correct_topk}/{total})")
    if args.neighbors:
        Path(args.neighbors).write_text(json.dumps(neighbors_dump, indent=2))
        print(f"Saved nearest matches to {args.neighbors}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
