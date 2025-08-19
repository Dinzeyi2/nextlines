# Natural language DS/ML helper

This prototype exposes a tiny HTTP API for issuing controlled data-science
commands in natural language. Only a small set of high‑level actions is allowed
(e.g. load, clean, encode, scale, split, build, fit, transform, evaluate, save).
Commands are sent to the `/parse` endpoint, which behaves the same as the
legacy `/execute` endpoint kept for backward compatibility.

## Requirements

The project relies on a few Python packages:

* **Mandatory**: [`pandas`](https://pandas.pydata.org/) and [`scikit-learn`](https://scikit-learn.org/)
  for dataframe operations and ML utilities.
* **Optional**: [`sentence_transformers`](https://www.sbert.net/) enables an
  embedding-based ML fallback when rule-based parsing fails.

## Installation

Install the mandatory packages first:

```bash
pip install pandas scikit-learn
```

To enable the ML fallback, install the additional dependency:

```bash
pip install sentence-transformers
```

## Setup

Clone the repository and start a local API server:

```bash
git clone <repo-url>
cd nextlines
python -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn fastapi uvicorn
```

Run the HTTP service with:

```bash
uvicorn api:app --reload
```

Commands can then be posted to `http://localhost:8000/parse` or the
legacy `/execute` endpoint.

## ML fallback example

With the optional model installed, unknown commands fall back to an
embedding-based parser:

```python
from execution import NaturalLanguageExecutor

executor = NaturalLanguageExecutor()
result = executor.execute("fit a linear regression model using X and y")
print(result)
```

The example above generates and executes Python code even when no
rule-based template matches the query.

## Prompt design

The parser expects concise, imperative instructions:

* One operation per line.
* Reference variables explicitly (e.g. `df`, `X`, `y`).
* Use lowercase verbs like "load", "scale" or "fit".
* Avoid extra punctuation or prose.

## How to phrase commands

Copy and adapt the examples below:

```text
load csv file /path/to/data.csv into df
clean remove rows with missing values from df
encode one hot encode column city in df
scale standard scale columns age and income in df
split df into train and test sets
build pipeline with standard scaler and logistic regression
fit pipeline on train
evaluate pipeline on test
save pipeline to model.joblib
reset session
```

Send a POST request to `/parse` (or `/execute` for backward compatibility):

```bash
curl -X POST localhost:8000/parse -H 'Content-Type: application/json' \
     -d '{"command": "load csv file data.csv into df"}'
```

## Example queries and generated code

* Query: `load CSV file at data.csv`

  ```python
  pd.read_csv("data.csv", sep=",")
  ```

* Query: `drop missing values`

  ```python
  df.dropna()
  ```

* Query: `scale columns ['age', 'income'] with StandardScaler`

  ```python
  scaler = StandardScaler()
  scaled = scaler.fit_transform(df[['age', 'income']])
  ```

* Query: `fit logistic regression`

  ```python
  model = LogisticRegression()
  model.fit(X, y)
  ```

## Security model and sandboxing

The `PythonExecutor` executes generated Python code in a restricted
environment designed to prevent misuse:

* The abstract syntax tree is validated to block `import` and `from` statements
  and to disallow calls to `eval`, `exec`, `__import__`, and `open`.
* Only a small whitelist of safe built-ins (e.g. `print`, `len`, `range`) is
  exposed to executed code; all other built-ins are unavailable.
* Execution happens inside a sandboxed subprocess with no network access and
  limited visibility of the filesystem.
* The sandbox has CPU time limited to one second and address space capped at
  roughly 50 MB via `resource` limits.
* The subprocess is terminated if execution exceeds a one‑second timeout.

These safeguards make code execution best‑effort safe while still supporting
basic educational and data‑science snippets; the sandbox is not a perfect
security boundary, so avoid running it with elevated privileges.
