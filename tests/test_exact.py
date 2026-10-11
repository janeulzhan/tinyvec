"""Tests for brute-force ground truth, recall, and the tie-margin guard."""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.datasets import load_siftsmall
from tinyvec.exact import brute_force, recall_at_k, tie_margin


def naive_knn(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Deliberately naive independent implementation: explicit Python loops,
    float64, full sort. Shares no code with brute_force, which is the point."""
    out = np.empty((queries.shape[0], k), dtype=np.int32)
    for qi in range(queries.shape[0]):
        dists = []
        for bi in range(base.shape[0]):
            acc = 0.0
            for j in range(base.shape[1]):
                diff = float(base[bi, j]) - float(queries[qi, j])
                acc += diff * diff
            dists.append((acc, bi))
        dists.sort()
        out[qi] = [bi for _, bi in dists[:k]]
    return out


def test_matches_naive_independent_implementation():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(120, 6)).astype(np.float32)
    queries = rng.normal(size=(8, 6)).astype(np.float32)
    got, _ = brute_force(base, queries, 5)
    assert np.array_equal(got, naive_knn(base, queries, 5))


def test_returned_distances_are_correct_squared_euclidean():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 9)).astype(np.float32)
    queries = rng.normal(size=(6, 9)).astype(np.float32)
    ids, dists = brute_force(base, queries, 7)
    for qi in range(6):
        for rank in range(7):
            want = float(((base[ids[qi, rank]] - queries[qi]) ** 2).sum())
            assert dists[qi, rank] == pytest.approx(want, rel=1e-5, abs=1e-4)


def test_distances_are_sorted_ascending():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(300, 5)).astype(np.float32)
    queries = rng.normal(size=(10, 5)).astype(np.float32)
    _, dists = brute_force(base, queries, 20)
    assert np.all(np.diff(dists, axis=1) >= -1e-5)


def test_query_that_is_in_the_base_set_finds_itself_first():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(150, 8)).astype(np.float32)
    ids, dists = brute_force(base, base[[7, 42, 99]], 1)
    assert ids[:, 0].tolist() == [7, 42, 99]
    assert np.allclose(dists[:, 0], 0.0, atol=1e-4)


def test_k_larger_than_n_clamps():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(5, 3)).astype(np.float32)
    ids, dists = brute_force(base, base[:2], 50)
    assert ids.shape == (2, 5)
    assert dists.shape == (2, 5)


def test_single_base_vector():
    base = np.array([[1.0, 2.0]], dtype=np.float32)
    ids, dists = brute_force(base, np.array([[0.0, 0.0]], np.float32), 3)
    assert ids.shape == (1, 1)
    assert ids[0, 0] == 0
    assert dists[0, 0] == pytest.approx(5.0)


def test_accepts_a_1d_query():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(40, 4)).astype(np.float32)
    ids, _ = brute_force(base, base[3], 2)
    assert ids.shape == (1, 2)
    assert ids[0, 0] == 3


def test_chunking_is_transparent():
    rng = np.random.default_rng(6)
    base = rng.normal(size=(400, 7)).astype(np.float32)
    queries = rng.normal(size=(37, 7)).astype(np.float32)
    a, da = brute_force(base, queries, 10, chunk=1000)
    b, db = brute_force(base, queries, 10, chunk=4)
    assert np.array_equal(a, b)
    assert np.allclose(da, db)


def test_rejects_empty_base():
    with pytest.raises(ValueError, match="empty base"):
        brute_force(np.zeros((0, 4), np.float32), np.zeros((1, 4), np.float32), 1)


def test_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        brute_force(np.zeros((5, 2), np.float32), np.zeros((1, 2), np.float32), 0)


# --------------------------------------------------------------------- recall


def test_recall_is_one_for_a_perfect_result():
    truth = np.arange(30, dtype=np.int32).reshape(3, 10)
    assert recall_at_k(truth, truth, 10) == 1.0


def test_recall_is_zero_for_a_disjoint_result():
    truth = np.arange(20, dtype=np.int32).reshape(2, 10)
    assert recall_at_k(truth + 1000, truth, 10) == 0.0


def test_recall_is_order_insensitive():
    truth = np.arange(10, dtype=np.int32).reshape(1, 10)
    assert recall_at_k(truth[:, ::-1], truth, 10) == 1.0


def test_recall_counts_partial_overlap():
    truth = np.arange(10, dtype=np.int32).reshape(1, 10)
    found = np.array([[0, 1, 2, 3, 99, 98, 97, 96, 95, 94]], dtype=np.int32)
    assert recall_at_k(found, truth, 10) == pytest.approx(0.4)


def test_recall_ignores_padding_beyond_k():
    truth = np.arange(10, dtype=np.int32).reshape(1, 10)
    found = np.concatenate([truth, np.full((1, 5), -1, np.int32)], axis=1)
    assert recall_at_k(found, truth, 10) == 1.0


def test_recall_rejects_too_few_columns():
    """Silently scoring against fewer than k columns would deflate the number
    without anyone noticing, so it is an error."""
    truth = np.arange(10, dtype=np.int32).reshape(1, 10)
    with pytest.raises(ValueError, match="at least k"):
        recall_at_k(truth[:, :4], truth, 10)


def test_recall_rejects_query_count_mismatch():
    with pytest.raises(ValueError, match="query count mismatch"):
        recall_at_k(np.zeros((2, 10), np.int32), np.zeros((3, 10), np.int32), 10)


# ----------------------------------------------------------------- tie margin


def test_tie_margin_is_zero_when_neighbours_are_equidistant():
    dists = np.array([[1.0, 2.0, 2.0, 5.0]])
    assert tie_margin(dists, 2) == 0.0


def test_tie_margin_is_positive_when_unambiguous():
    dists = np.array([[1.0, 2.0, 7.0, 9.0]])
    assert tie_margin(dists, 2) == pytest.approx(5.0)


def test_tie_margin_takes_the_worst_query():
    dists = np.array([[1.0, 2.0, 9.0], [1.0, 2.0, 2.5]])
    assert tie_margin(dists, 2) == pytest.approx(0.5)


def test_tie_margin_needs_a_spare_column():
    with pytest.raises(ValueError, match="more than k"):
        tie_margin(np.zeros((2, 3)), 3)


# ------------------------------------------------- cross-check vs real corpus


def test_reproduces_published_siftsmall_groundtruth():
    """The strongest correctness evidence available: our exact search is checked
    against ground truth published by someone else, computed independently.

    Two claims, both portable. Neighbour sets agree at every k up to 100, and
    the sequence of *distances* is identical, which is the stronger statement
    and the one that survives ties.

    Exact id ordering is deliberately not asserted. Where two base vectors are
    equidistant from a query, which one sorts first depends on float32
    accumulation order and therefore on the BLAS build; an earlier version of
    this test pinned the ordering to k=20, passed on macOS Accelerate, and
    failed in CI on Linux. Distance-sequence equality says everything the
    ordering claim was trying to say, without depending on how ties break.
    """
    ds = load_siftsmall()
    ids, _ = brute_force(ds.base, ds.queries, 100)
    for k in (1, 5, 10, 20, 50, 100):
        for qi in range(ds.queries.shape[0]):
            assert set(ids[qi, :k].tolist()) == set(ds.groundtruth[qi, :k].tolist()), (k, qi)

    base = ds.base.astype(np.float64)
    for qi in range(ds.queries.shape[0]):
        query = ds.queries[qi].astype(np.float64)
        ours = ((base[ids[qi]] - query) ** 2).sum(axis=1)
        theirs = ((base[ds.groundtruth[qi]] - query) ** 2).sum(axis=1)
        assert np.array_equal(ours, theirs), qi


# Upper bound on how many of the 100 x 100 ground-truth cells may sit in a
# different order than the published file. Exactly 18 permute on the authoring
# machine; tie-break order varies by BLAS build so the exact count is not
# portable, but a pathological reordering should still fail loudly. 200 cells is
# 2 percent of the grid, roughly 11x the observed value.
MAX_PERMUTED_CELLS = 200


def test_every_disagreement_with_published_groundtruth_is_a_distance_tie():
    """Any id we return that differs from the published file must be a vector at
    exactly the same distance, never a genuinely different neighbour.

    Exact equality is the right comparison here, not a tolerance, and SIFT makes
    that provable rather than hopeful. Descriptors are uint8, so squared
    distances are integers; the largest is 513,065, comfortably under the 2^24
    beyond which float32 stops representing integers exactly. Measured float32
    error against a float64 reference on this dataset is 0.0, and the smallest
    non-zero gap between adjacent distances in the top 100 is exactly 1.0. Two
    distances here are therefore either identical or at least 1.0 apart, with no
    near-tie band in between, so a tie is unambiguous and a tolerance would only
    blur a distinction the data does not actually contain.

    The permuted-cell count is asserted as a bound and reported on failure, so a
    regression that reorders genuinely unequal neighbours cannot hide behind the
    tie exemption.
    """
    ds = load_siftsmall()
    ids, _ = brute_force(ds.base, ds.queries, 100)
    base = ds.base.astype(np.float64)
    mismatches = np.argwhere(ids != ds.groundtruth)

    genuine: list[str] = []
    for qi, col in mismatches:
        q = ds.queries[qi].astype(np.float64)
        ours = ((base[ids[qi, col]] - q) ** 2).sum()
        theirs = ((base[ds.groundtruth[qi, col]] - q) ** 2).sum()
        if ours != theirs:
            genuine.append(
                f"query {qi} rank {col}: ours {ids[qi, col]} at {ours:.1f}, "
                f"published {ds.groundtruth[qi, col]} at {theirs:.1f}"
            )

    assert not genuine, (
        f"{len(genuine)} of {len(mismatches)} positional differences are NOT "
        f"distance ties, so the search returned genuinely wrong neighbours:\n  "
        + "\n  ".join(genuine[:10])
    )
    assert len(mismatches) <= MAX_PERMUTED_CELLS, (
        f"{len(mismatches)} of {ids.size} ground-truth cells are permuted "
        f"against the published file, over the bound of {MAX_PERMUTED_CELLS}. "
        f"Every one is an exact distance tie, so this is not a correctness "
        f"failure, but the jump is large enough to warrant a look at the "
        f"sort's tie-breaking before raising the bound."
    )


def test_siftsmall_distances_are_integers_so_ties_are_unambiguous():
    """Underpins the exact-equality comparison used above.

    If this ever stops holding, tie detection needs a tolerance and the
    reasoning in this file has to be revisited.
    """
    ds = load_siftsmall()
    _, dists = brute_force(ds.base, ds.queries, 100)
    exact = np.stack(
        [
            ((ds.base.astype(np.float64) - ds.queries[qi].astype(np.float64)) ** 2).sum(axis=1)
            for qi in range(ds.queries.shape[0])
        ]
    )
    assert np.all(exact == np.round(exact)), "squared distances are not integral"
    assert exact.max() < 2**24, "distances exceed exact float32 integer range"
    # float32 computation is exact on this data, so returned distances match.
    assert np.array_equal(dists.astype(np.float64), np.sort(exact, axis=1)[:, :100])
    gaps = np.diff(np.sort(exact, axis=1)[:, :101], axis=1)
    assert gaps[gaps > 0].min() >= 1.0, "found a sub-unit non-zero gap"


def test_siftsmall_top10_groundtruth_is_unambiguous():
    """Guards the recall numbers in the README: if this margin were zero, recall
    at 10 would not be a well-defined quantity on this dataset."""
    ds = load_siftsmall()
    _, dists = brute_force(ds.base, ds.queries, 11)
    assert tie_margin(dists, 10) > 0.0
