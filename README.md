# DREAM challenge

This repository contains an exploratory solution to the [DREAM challenge](https://dreamchallenges.org/), which models expression from promoter sequences.

## Setup

Install [uv](https://docs.astral.sh/uv/) and create the locked environment:

```sh
uv sync --locked
```

## Data

Challenge data is not committed. Place `train_sequences.txt` and `test_sequences.txt` in `data/`. Each row must contain a sequence and expression value separated by a tab. The sequence includes the challenge prefix and suffix used by the feature pipeline.

## Commands

Run preprocessing with explicit, non-interactive defaults:

```sh
uv run python process.py
uv run python dream.py
```

The baseline writes `pred.json` using the keys in `sample_submission.json`. For small experiments, call `vectors.tokenize` or `dream.train_and_predict` with custom paths and limits.

## Notebooks

`dream.ipynb` and `notebooks/lgbm.ipynb` preserve the original exploratory work. Matching readable Python scripts are stored beside them as `dream.py` and `notebooks/lgbm.py`; the scripts are synchronized representations for review and reuse.

## Development

Run the checks used by CI:

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Tests use small synthetic sequences and never require challenge-scale data. Generated data, models, results, predictions, and notebook checkpoints are ignored by Git.
