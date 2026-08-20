"""Train the baseline model and create a submission."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from vectors import tokenize


def dump_predictions(predictions: dict[str, float], output_path: str | Path) -> None:
    """Write predictions as JSON."""
    with Path(output_path).open("w") as handle:
        json.dump(predictions, handle)


def submit(
    predictions: np.ndarray,
    sample_path="sample_submission.json",
    output_path="pred.json",
) -> None:
    """Align predictions with the sample submission keys."""
    with Path(sample_path).open() as handle:
        sample = json.load(handle)
    ordered = OrderedDict((key, float(predictions[int(key)])) for key in sample)
    dump_predictions(ordered, output_path)


def train_and_predict(
    train_path, test_path, sample_path, output_path, size=4, limit=None
) -> None:
    """Train the baseline and write test predictions."""
    train_vectors, labels, _ = tokenize(size, train_path, limit=limit)
    model = RandomForestRegressor(
        n_estimators=100, bootstrap=False, random_state=0, n_jobs=-1
    )
    model.fit(train_vectors, labels)
    test_vectors, _, _ = tokenize(size, test_path, submission=True, limit=limit)
    test_vectors = test_vectors.reindex(columns=train_vectors.columns, fill_value=0)
    submit(model.predict(test_vectors), sample_path, output_path)


def main() -> None:
    """Run the baseline pipeline."""
    train_and_predict(
        "data/train_sequences.txt",
        "data/test_sequences.txt",
        "sample_submission.json",
        "pred.json",
    )


if __name__ == "__main__":
    main()
