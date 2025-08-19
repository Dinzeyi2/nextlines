# Natural language DS/ML helper

This prototype exposes a tiny HTTP API for issuing controlled data-science
commands in natural language. Only a small set of high‑level actions is allowed
(e.g. load, clean, encode, scale, split, build, fit, transform, evaluate, save).
Additional dataset management helpers are exposed via dedicated commands to
load a dataframe into a `DatasetManager`, validate it against an inferred
schema and create train/validation/test splits.
Commands are sent to the `/parse` endpoint, which behaves the same as the
legacy `/execute` endpoint kept for backward compatibility.

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

### Dataset management

```text
dataset_load df target target
dataset_validate df
dataset_split df
```

Send a POST request to `/parse` (or `/execute` for backward compatibility):

```bash
curl -X POST localhost:8000/parse -H 'Content-Type: application/json' \
     -d '{"command": "load csv file data.csv into df"}'
```
