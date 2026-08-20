"""Sequence parsing and k-mer feature generation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PREFIX = "TGCATTTTTTTCACATC"
SUFFIX = "GGTTACGGCTGTT"
NUCLEOTIDES = frozenset("ACTGN")


def remove_prefix(text: str, prefix: str = PREFIX) -> str:
    """Remove a required sequence prefix."""
    if not text.startswith(prefix):
        raise ValueError(f"Sequence does not start with {prefix}")
    return text[len(prefix) :]


def remove_suffix(text: str, suffix: str = SUFFIX) -> str:
    """Remove a required sequence suffix."""
    if not text.endswith(suffix):
        raise ValueError(f"Sequence does not end with {suffix}")
    return text[: -len(suffix)]


def parse_rows(filename: str | Path) -> list[tuple[str, float]]:
    """Read tab separated sequence and expression rows."""
    rows = []
    with Path(filename).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Expected two fields on line {line_number}")
            sequence, expression = fields
            rows.append((sequence, float(expression)))
    return rows


def kmerize_sequence(sequence: str, size: int = 4, stride: int = 1) -> list[str]:
    """Return valid overlapping k-mers from a wrapped sequence."""
    core = remove_suffix(remove_prefix(sequence))
    if not set(core) <= NUCLEOTIDES:
        raise ValueError("Sequence contains an invalid nucleotide")
    if size < 1 or stride < 1:
        raise ValueError("Size and stride must be positive")
    return [core[i : i + size] for i in range(0, len(core) - size + 1, stride)]


def kmerize(
    filename: str | Path, stride: int = 1, size: int = 4, limit: int | None = None
):
    """Generate k-mer strings and expression values."""
    rows = parse_rows(filename)
    if limit is not None:
        rows = rows[:limit]
    sequences = [sequence for sequence, _ in rows]
    kmers = [
        ",".join(kmerize_sequence(sequence, size, stride)) for sequence in sequences
    ]
    expressions = np.array([expression for _, expression in rows], dtype=np.float32)
    return pd.DataFrame({"sequence": sequences, "kmers": kmers}), expressions


def tokenize(
    size=4, input_path="data/train_sequences.txt", submission=False, limit=None
):
    """Create a k-mer frequency matrix."""
    database, expressions = kmerize(input_path, size=size, limit=limit)
    vectorizer = CountVectorizer(
        tokenizer=lambda value: value.split(","), token_pattern=None
    )
    matrix = vectorizer.fit_transform(database["kmers"])
    vectors = pd.DataFrame(
        matrix.toarray(),
        index=database["sequence"],
        columns=vectorizer.get_feature_names_out(),
    )
    return vectors, expressions, submission
