"""Prepare challenge data for model training."""

from pathlib import Path

import numpy as np

from vectors import parse_rows, remove_prefix, remove_suffix


def one_hot_encode(sequence: str, max_length: int = 112) -> np.ndarray:
    """Encode a wrapped sequence as a flattened one-hot array."""
    core = remove_suffix(remove_prefix(sequence))
    if len(core) > max_length:
        raise ValueError("Sequence is longer than the configured maximum")
    mapping = {nucleotide: index for index, nucleotide in enumerate("ACTG")}
    encoded = np.zeros((max_length, 4), dtype=np.float16)
    for index, nucleotide in enumerate(core):
        if nucleotide in mapping:
            encoded[index, mapping[nucleotide]] = 1 / max_length
    return encoded.ravel()


def create_one_hot_file(
    input_path: str | Path, output_path: str | Path, limit=None
) -> None:
    """Write one-hot features and labels to a compressed NumPy file."""
    rows = parse_rows(input_path)
    if limit is not None:
        rows = rows[:limit]
    features = np.stack([one_hot_encode(sequence) for sequence, _ in rows])
    labels = np.array([expression for _, expression in rows], dtype=np.float16)
    np.savez(output_path, x=features, y=labels)


def main() -> None:
    """Run one-hot preprocessing with explicit paths."""
    create_one_hot_file("data/train_sequences.txt", "data/onehot.npz")


if __name__ == "__main__":
    main()
