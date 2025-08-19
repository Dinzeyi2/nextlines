# Natural language DS/ML helper

This prototype exposes a tiny HTTP API for issuing controlled data-science
commands in natural language. Only a small set of high‑level actions is allowed
(e.g. load, clean, encode, scale, split, build, fit, transform, evaluate, save).

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

Send a POST request to `/execute`:

```bash
curl -X POST localhost:8000/execute -H 'Content-Type: application/json' \
     -d '{"command": "load csv file data.csv into df"}'
```
