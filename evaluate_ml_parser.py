from __future__ import annotations
import ast
import json
from pathlib import Path

from ml_parser import MLCodeGenerator

MODEL_PATH = Path("models/ml_parser.json")
TEST_PATH = Path("data/test_corpus.json")


def main() -> None:
    parser = MLCodeGenerator.load(MODEL_PATH)
    test_data = json.loads(TEST_PATH.read_text())
    total = len(test_data)
    correct_intent = 0
    correct_code = 0
    for item in test_data:
        predicted = parser.predict(item["query"])
        if predicted == item["code"]:
            correct_intent += 1
        try:
            ast.parse(predicted)
            correct_code += 1
        except SyntaxError:
            pass
    intent_acc = correct_intent / total if total else 0.0
    code_corr = correct_code / total if total else 0.0
    print(f"Intent accuracy: {intent_acc:.2f}")
    print(f"Code correctness (syntax): {code_corr:.2f}")


if __name__ == "__main__":
    main()
