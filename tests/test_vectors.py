import json

import numpy as np
import pytest

from dream import submit
from process import one_hot_encode
from vectors import kmerize, kmerize_sequence, parse_rows, remove_prefix, remove_suffix

PREFIX = "TGCATTTTTTTCACATC"
SUFFIX = "GGTTACGGCTGTT"


def sequence(core="ACGTAC"):
    return PREFIX + core + SUFFIX


def test_prefix_and_suffix_validation():
    assert remove_prefix(PREFIX + "A") == "A"
    assert remove_suffix("A" + SUFFIX) == "A"
    with pytest.raises(ValueError):
        remove_prefix("A")
    with pytest.raises(ValueError):
        remove_suffix("A")


def test_kmers_support_size_and_stride():
    assert kmerize_sequence(sequence(), size=3, stride=2) == ["ACG", "GTA"]


def test_kmers_reject_invalid_input():
    with pytest.raises(ValueError):
        kmerize_sequence(sequence("ACX"))
    with pytest.raises(ValueError):
        kmerize_sequence(sequence(), size=0)


def test_parse_and_kmerize(tmp_path):
    path = tmp_path / "sequences.txt"
    path.write_text(f"{sequence()}\t1.5\n")
    assert parse_rows(path) == [(sequence(), 1.5)]
    database, labels = kmerize(path, size=2)
    assert database.shape == (1, 2)
    assert np.allclose(labels, [1.5])


def test_parse_rejects_malformed_rows(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("not a tab separated row\n")
    with pytest.raises(ValueError):
        parse_rows(path)


def test_one_hot_encoding():
    encoded = one_hot_encode(sequence("ACN"), max_length=4).reshape(4, 4)
    assert encoded.shape == (4, 4)
    assert encoded[0, 0] > 0
    assert encoded[1, 1] > 0
    assert encoded[2].sum() == 0


def test_submission_preserves_sample_order(tmp_path):
    sample = tmp_path / "sample.json"
    output = tmp_path / "pred.json"
    sample.write_text(json.dumps({"2": 0, "0": 0}))
    submit(np.array([1.25, 2.5, 3.75]), sample, output)
    assert list(json.loads(output.read_text())) == ["2", "0"]
    assert json.loads(output.read_text())["2"] == 3.75
