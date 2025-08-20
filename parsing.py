import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None


class ParameterExtractionError(Exception):
    """Raised when a parameter cannot be parsed"""
    pass


class ParameterType(Enum):
    IDENTIFIER = "identifier"
    VALUE = "value"
    CONDITION = "condition"
    EXPRESSION = "expression"
    TYPE = "type"
    STATEMENT = "statement"
    COLLECTION = "collection"


@dataclass
class ExecutionTemplate:
    pattern: str
    execution_func: str  # Function to call for execution
    code_template: str = ""  # Python code template for real execution
    parameters: Dict[str, ParameterType] | None = None
    priority: int = 1

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}


class ParameterExtractor:
    def __init__(self, context: "ExecutionContext") -> None:
        self.context = context
        self.number_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10,
        }
        self.ordinal_words = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        }
        self.boolean_words = {
            "true": True, "false": False, "yes": True, "no": False,
        }
        self.identifier_synonyms = {
            "ids": "id",
            "identifiers": "id",
            "identifier": "id",
        }

        self.value_synonyms = {
            "couple": 2,
            "pair": 2,
            "dozen": 12,
        }
        self.unit_multipliers = {
            "dozen": 12,
            "score": 20,
        }

        if spacy:
            try:  # pragma: no cover - runtime dependency
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:  # Model not installed or other error
                try:
                    self.nlp = spacy.blank("en")
                    self.nlp.add_pipe("lemmatizer", config={"mode": "rule"})
                    self.nlp.initialize()
                except Exception:
                    self.nlp = None
        else:
            self.nlp = None

    def _tokenize(self, text: str) -> List[str]:
        try:
            return shlex.split(text)
        except Exception:
            return text.split()

    def _resolve_pronoun(self, text: str) -> str | None:
        if not self.nlp:
            return None
        doc = self.nlp(text)
        if len(doc) != 1:
            return None
        token = doc[0]
        if token.pos_ != "PRON":
            return None
        pronoun = token.lemma_.lower()
        if pronoun in {"it", "this", "that"}:
            return self.context.last_assigned_variable
        if pronoun in {"they", "them", "those", "these"}:
            return self.context.last_collection
        return None

    def extract_identifier(self, text: str) -> str:
        """Convert natural language to Python identifier"""
        pronoun = self._resolve_pronoun(text)
        if pronoun:
            return pronoun

        resolved = self.context.resolve_reference(text)
        if resolved and resolved != text:
            return resolved

        if self.nlp:
            doc = self.nlp(text)
            tokens = [t for t in doc if not t.is_space]
            if not tokens:
                raise ParameterExtractionError(f"Cannot resolve identifier from '{text}'")
            normalized: List[str] = []
            for tok in tokens:
                lemma = tok.lemma_.lower()
                lemma = ''.join(ch for ch in lemma if ch.isalnum() or ch == '_')
                lemma = self.identifier_synonyms.get(lemma, lemma)
                normalized.append(lemma)
            identifier = '_'.join(normalized)
        else:
            tokens = [t.lower() for t in self._tokenize(text)]
            if not tokens:
                raise ParameterExtractionError(f"Cannot resolve identifier from '{text}'")
            normalized = []
            for tok in tokens:
                tok = ''.join(ch for ch in tok if ch.isalnum() or ch == '_')
                tok = self.identifier_synonyms.get(tok, tok)
                normalized.append(tok)
            identifier = '_'.join(normalized)
        if not identifier:
            raise ParameterExtractionError(f"Cannot resolve identifier from '{text}'")
        return identifier

    def extract_value(self, text: str) -> Any:
        """Extract and convert values to appropriate Python types"""
        text = text.strip()
        pronoun = self._resolve_pronoun(text)
        if pronoun:
            if pronoun in self.context.variables:
                return self.context.variables[pronoun]
            return pronoun

        tokens = self._tokenize(text)
        if not tokens:
            raise ParameterExtractionError(f"Cannot resolve value from '{text}'")

        if len(tokens) >= 2:
            first, second = tokens[0], tokens[1]
            first_low = first.lower()
            second_low = second.lower()
            num = None
            if first_low in ("a", "an"):
                num = 1
            elif first_low.isdigit():
                num = int(first_low)
            elif first_low in self.number_words:
                num = self.number_words[first_low]
            if num is not None and second_low in self.unit_multipliers:
                return num * self.unit_multipliers[second_low]

        if len(tokens) == 1:
            token = tokens[0]
            lower = token.lower()

            if lower.isdigit():
                return int(lower)
            if lower in self.number_words:
                return self.number_words[lower]
            if lower in self.ordinal_words:
                return self.ordinal_words[lower]
            if lower in self.value_synonyms:
                return self.value_synonyms[lower]
            if lower in self.unit_multipliers:
                return self.unit_multipliers[lower]
            try:
                return float(lower)
            except ValueError:
                pass
            if lower in self.boolean_words:
                return self.boolean_words[lower]
            if lower in self.context.variables:
                return self.context.variables[lower]
            return token

        return ' '.join(tokens)

    def extract_condition_parts(self, text: str) -> Tuple[Any, str, Any]:
        """Extract condition parts for evaluation"""
        tokens = [t.lower() for t in self._tokenize(text)]
        operator_map = {
            ("is", "equal", "to"): "==",
            ("equals",): "==",
            ("is", "greater", "than"): ">",
            ("is", "less", "than"): "<",
            ("is", "greater", "than", "or", "equal", "to"): ">=",
            ("is", "less", "than", "or", "equal", "to"): "<=",
            ("is", "not", "equal", "to"): "!=",
            ("contains",): "in",
            ("is", "in"): "in",
            ("is", "not", "in"): "not in",
        }

        for phrase, op in operator_map.items():
            length = len(phrase)
            for i in range(len(tokens) - length + 1):
                if tokens[i:i + length] == list(phrase):
                    left_tokens = tokens[:i]
                    right_tokens = tokens[i + length:]
                    left = self.extract_value(' '.join(left_tokens))
                    right = self.extract_value(' '.join(right_tokens))
                    return left, op, right

        raise ParameterExtractionError(f"Cannot parse condition '{text}'")

    def extract_collection(self, text: str) -> List[Any]:
        """Extract collection from natural language"""
        text = text.strip()

        if text.startswith('[') and text.endswith(']'):
            try:
                return eval(text)
            except Exception:
                pass

        if ',' in text:
            items = []
            for item in text.split(','):
                items.append(self.extract_value(item.strip()))
            return items

        return [self.extract_value(text)]
