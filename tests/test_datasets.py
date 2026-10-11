"""Tests for dataset loading and the synthetic generator."""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.datasets import SIFTSMALL_PATH, Dataset, gaussian_mixture, load_siftsmall


def test_vendored_archive_is_present():
    """The repository is meant to be usable offline from a fresh clone, so the
    corpus has to actually be in it."""
    assert SIFTSMALL_PATH.exists(), f"missing vendored dataset at {SIFTSMALL_PATH}"


def test_siftsmall_shapes_and_dtypes():
    ds = load_siftsmall()
    assert ds.base.shape == (10_000, 128)
    assert ds.queries.shape == (100, 128)
    assert ds.train.shape == (25_000, 128)
    assert ds.groundtruth.shape == (100, 100)
    for array in (ds.base, ds.queries, ds.train):
        assert array.dtype == np.float32
        assert array.flags["C_CONTIGUOUS"]
    assert ds.groundtruth.dtype == np.int32


def test_siftsmall_is_flagged_as_real_data():
    """The synthetic flag is what the README's honesty rests on; if this ever
    flipped, the benchmark would be describing generated data as measured."""
    ds = load_siftsmall()
    assert ds.synthetic is False
    assert ds.metric == "euclidean"
    assert "real" in ds.describe()


def test_siftsmall_values_are_valid_sift_descriptors():
    """SIFT components are integral and fit in a byte. This is the property the
    lossless uint8 vendoring depends on, so it is verified on load rather than
    assumed."""
    ds = load_siftsmall()
    for name, array in (("base", ds.base), ("queries", ds.queries), ("train", ds.train)):
        assert np.all(array == np.round(array)), name
        assert array.min() >= 0 and array.max() <= 255, name


def test_siftsmall_groundtruth_ids_are_in_range():
    ds = load_siftsmall()
    assert ds.groundtruth.min() >= 0
    assert ds.groundtruth.max() < ds.n


def test_siftsmall_train_set_is_disjoint_from_base():
    """PQ codebooks must not be fitted on the vectors they will encode. The SIFT
    learn set is a separate capture, and this confirms the vendored copy kept
    them separate."""
    ds = load_siftsmall()
    base_rows = {row.tobytes() for row in ds.base}
    train_rows = {row.tobytes() for row in ds.train}
    overlap = base_rows & train_rows
    assert not overlap, f"{len(overlap)} training vectors also appear in the base set"


def test_siftsmall_helpers():
    ds = load_siftsmall()
    assert ds.n == 10_000
    assert ds.dim == 128


def test_missing_archive_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_siftsmall"):
        load_siftsmall(tmp_path / "nope.npz")


# ------------------------------------------------------------------ synthetic


def test_gaussian_mixture_shapes():
    ds = gaussian_mixture(n=100, dim=8, n_queries=7, n_train=50, n_clusters=4, seed=0)
    assert ds.base.shape == (100, 8)
    assert ds.queries.shape == (7, 8)
    assert ds.train.shape == (50, 8)
    assert ds.groundtruth is None


def test_gaussian_mixture_is_flagged_synthetic():
    ds = gaussian_mixture(n=50, dim=4, seed=0)
    assert ds.synthetic is True
    assert "synthetic" in ds.describe()


def test_gaussian_mixture_is_deterministic():
    a = gaussian_mixture(n=80, dim=6, seed=9)
    b = gaussian_mixture(n=80, dim=6, seed=9)
    assert np.array_equal(a.base, b.base)
    assert np.array_equal(a.queries, b.queries)
    assert np.array_equal(a.train, b.train)


def test_gaussian_mixture_different_seeds_differ():
    a = gaussian_mixture(n=80, dim=6, seed=1)
    b = gaussian_mixture(n=80, dim=6, seed=2)
    assert not np.array_equal(a.base, b.base)


def test_gaussian_mixture_is_actually_clustered():
    """A uniform cloud would give an inverted file nothing to exploit, so the
    generator has to produce real cluster structure. Checked by comparing
    within-cluster spread against the spread of the component means."""
    ds = gaussian_mixture(n=4000, dim=16, n_clusters=10, spread=10.0, seed=0)
    from tinyvec.kmeans import kmeans

    clustered = kmeans(ds.base, 10, n_iter=30, seed=0).inertia
    unclustered = float(np.sum((ds.base - ds.base.mean(axis=0)) ** 2))
    assert clustered < 0.2 * unclustered


def test_gaussian_mixture_rejects_bad_sizes():
    with pytest.raises(ValueError, match="must be positive"):
        gaussian_mixture(n=0)


def test_dataset_is_frozen():
    ds = gaussian_mixture(n=10, dim=4, seed=0)
    assert isinstance(ds, Dataset)
    with pytest.raises(Exception):
        ds.name = "mutated"
