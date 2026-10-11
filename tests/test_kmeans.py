"""Tests for the k-means implementation."""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.kmeans import assign, kmeans, squared_distances


def test_squared_distances_matches_naive_loop():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(17, 11)).astype(np.float32)
    b = rng.normal(size=(23, 11)).astype(np.float32)
    got = squared_distances(a, b)
    want = np.empty((17, 23), dtype=np.float64)
    for i in range(17):
        for j in range(23):
            want[i, j] = np.sum((a[i].astype(np.float64) - b[j].astype(np.float64)) ** 2)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_squared_distances_never_negative_on_identical_rows():
    """The |a|^2 - 2ab + |b|^2 expansion can underflow below zero; it is clipped."""
    x = np.full((50, 64), 3.7, dtype=np.float32)
    d = squared_distances(x, x)
    assert np.all(d >= 0.0)
    assert np.allclose(np.diag(d), 0.0)


def test_squared_distances_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        squared_distances(np.zeros((3, 4), np.float32), np.zeros((3, 5), np.float32))


# Lloyd is exactly monotone in theory, and measured exactly monotone here: the
# worst relative increase across 500 seeds of the two fixtures below was 0.0.
# The bound is nevertheless relative rather than exactly zero, because centroid
# means are rounded to float32 on store, so a centroid is not precisely its
# cluster's inertia minimizer and an increase of order float32 epsilon is
# mechanically possible. 1e-9 is tight enough to catch any real regression while
# leaving room for that. An earlier absolute bound of 1e-3 was, on an inertia of
# 0.03, a 3 percent allowance.
MONOTONICITY_RELATIVE_SLACK = 1e-9


def test_inertia_is_non_increasing():
    """Pure Lloyd must never increase inertia. This is the invariant that an
    empty-cluster 'repair' would silently break."""
    rng = np.random.default_rng(1)
    x = np.concatenate(
        [rng.normal(loc=loc, size=(200, 8)) for loc in (-6.0, 0.0, 6.0, 12.0)]
    ).astype(np.float32)
    result = kmeans(x, 12, n_iter=40, seed=0, tol=0.0)
    history = np.array(result.inertia_history)
    assert len(history) >= 2
    deltas = np.diff(history)
    slack = MONOTONICITY_RELATIVE_SLACK * max(history[0], 1.0)
    assert np.all(deltas <= slack), f"inertia increased: {deltas[deltas > 0]}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_inertia_is_non_increasing_when_clusters_go_empty(seed):
    """Many centroids over few clumps is the case that strains the empty-cluster
    path. Monotonicity must still hold.

    The clumps sit near the origin with noise that is a real fraction of their
    separation, which keeps the assignment step inside the regime where it can
    actually resolve the nearest centroid. An earlier version of this test used
    clumps of radius 0.01 centred 20 units out, where squared_distances carries
    71 percent relative error on the distances that matter; Lloyd's monotonicity
    does not hold when the argmin is that unreliable, and the test passed on one
    BLAS and failed on another. See test_precision_limit_of_the_expansion below,
    which pins that behaviour deliberately instead.
    """
    rng = np.random.default_rng(seed)
    clumps = rng.normal(size=(3, 5)).astype(np.float32) * 2.0
    x = np.repeat(clumps, 40, axis=0) + rng.normal(scale=0.3, size=(120, 5)).astype(np.float32)
    result = kmeans(x, 30, n_iter=30, seed=seed + 7, tol=0.0)
    history = np.array(result.inertia_history)
    deltas = np.diff(history)
    # Inertia is computed from direct float64 residuals, so the only slack
    # needed is for an argmin that float32 may resolve differently; scale it to
    # the inertia itself rather than using an absolute epsilon.
    slack = MONOTONICITY_RELATIVE_SLACK * max(history[0], 1.0)
    assert np.all(deltas <= slack), deltas[deltas > 0]


def test_precision_limit_of_the_expansion():
    """Pin the documented failure regime rather than leaving it implicit.

    Tight clusters far from the origin make the |a|^2 - 2a.b + |b|^2 expansion
    lose essentially all relative accuracy on small distances, because it
    subtracts quantities of size |a|^2 to produce a result of size |a - b|^2.
    This asserts the limit exists and that the well-conditioned case is fine, so
    that anyone reading squared_distances' docstring can see it is measured.
    """
    rng = np.random.default_rng(0)

    def relative_error(offset: float, noise: float) -> float:
        centers = (rng.normal(size=(3, 5)) * offset).astype(np.float32)
        x = (
            np.repeat(centers, 40, axis=0) + rng.normal(scale=noise, size=(120, 5))
        ).astype(np.float32)
        got = squared_distances(x, centers)
        want = (
            (x.astype(np.float64)[:, None, :] - centers.astype(np.float64)[None, :, :]) ** 2
        ).sum(-1)
        near = want < np.percentile(want, 30)
        return float(np.abs(got - want)[near].max() / max(want[near].max(), 1e-12))

    pathological = relative_error(offset=20.0, noise=0.01)
    well_conditioned = relative_error(offset=0.0, noise=1.0)
    assert pathological > 0.1, pathological
    assert well_conditioned < 1e-6, well_conditioned


def test_inertia_is_computed_exactly_not_from_the_expansion():
    """Inertia must come from direct residuals, so it stays accurate even where
    the assignment distances do not. Checked against an independent float64
    computation on deliberately ill-conditioned data."""
    rng = np.random.default_rng(2)
    clumps = rng.normal(size=(3, 5)).astype(np.float32) * 20.0
    x = np.repeat(clumps, 40, axis=0) + rng.normal(scale=0.01, size=(120, 5)).astype(np.float32)
    result = kmeans(x, 30, n_iter=30, seed=3, tol=0.0)
    independent = float(
        (
            (x.astype(np.float64) - result.centroids[result.labels].astype(np.float64)) ** 2
        ).sum()
    )
    assert result.inertia == pytest.approx(independent, rel=1e-9)
    # The expanded distances disagree with the truth by an amount comparable to
    # the inertia itself here, which is exactly why they are not used for it.
    _, expanded = assign(x, result.centroids)
    assert abs(float(expanded.sum()) - independent) > 1e-4


def test_converges_to_planted_clusters():
    """Well-separated clusters should be recovered: inertia must collapse to
    roughly the within-cluster noise."""
    rng = np.random.default_rng(4)
    centers = np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 40.0], [40.0, 40.0]], dtype=np.float32)
    x = np.repeat(centers, 150, axis=0) + rng.normal(scale=0.5, size=(600, 2)).astype(np.float32)
    result = kmeans(x, 4, n_iter=50, seed=0)
    assert result.centroids.shape == (4, 2)
    assert len(np.unique(result.labels)) == 4
    # Each recovered centroid should sit on top of a planted one.
    d = squared_distances(centers, result.centroids)
    assert np.all(d.min(axis=1) < 1.0)
    assert result.inertia < 600 * 2 * 0.5**2 * 2.0


def test_k_larger_than_n_clamps_to_distinct_rows():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    result = kmeans(x, 25, seed=0)
    assert result.centroids.shape[0] == 3
    assert result.inertia == pytest.approx(0.0, abs=1e-5)


def test_duplicate_points_do_not_produce_duplicate_centroids():
    """A PQ codebook with duplicate entries wastes code space, so k is clamped
    to the number of distinct rows rather than padded with repeats."""
    x = np.repeat(np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32), 50, axis=0)
    result = kmeans(x, 16, seed=0)
    assert result.centroids.shape[0] == 2
    assert len(np.unique(result.centroids, axis=0)) == 2
    assert result.inertia == pytest.approx(0.0, abs=1e-5)


def test_single_point():
    x = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    result = kmeans(x, 1, seed=0)
    assert result.centroids.shape == (1, 3)
    assert np.allclose(result.centroids[0], x[0])
    assert result.inertia == pytest.approx(0.0)


def test_all_identical_points():
    x = np.full((64, 7), 2.5, dtype=np.float32)
    result = kmeans(x, 8, seed=0)
    assert result.centroids.shape[0] == 1
    assert result.inertia == pytest.approx(0.0)


def test_same_seed_is_deterministic():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(400, 12)).astype(np.float32)
    a = kmeans(x, 10, seed=7)
    b = kmeans(x, 10, seed=7)
    assert np.array_equal(a.centroids, b.centroids)
    assert np.array_equal(a.labels, b.labels)
    assert a.inertia_history == b.inertia_history


def test_different_seeds_differ():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(400, 12)).astype(np.float32)
    a = kmeans(x, 10, seed=1)
    b = kmeans(x, 10, seed=2)
    assert not np.array_equal(a.centroids, b.centroids)


def test_labels_are_nearest_centroid():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 9)).astype(np.float32)
    result = kmeans(x, 11, seed=0)
    expected = np.argmin(squared_distances(x, result.centroids), axis=1)
    assert np.array_equal(result.labels, expected)


def test_assign_chunking_is_transparent():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(500, 6)).astype(np.float32)
    c = rng.normal(size=(13, 6)).astype(np.float32)
    whole_labels, whole_d = assign(x, c, chunk=10_000)
    chunked_labels, chunked_d = assign(x, c, chunk=7)
    assert np.array_equal(whole_labels, chunked_labels)
    assert np.allclose(whole_d, chunked_d)


@pytest.mark.parametrize("bad_k", [0, -3])
def test_rejects_non_positive_k(bad_k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        kmeans(np.zeros((5, 2), np.float32), bad_k)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="zero points"):
        kmeans(np.zeros((0, 4), np.float32), 3)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        kmeans(np.zeros(10, np.float32), 3)
