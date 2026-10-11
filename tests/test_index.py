"""IVF-PQ index behaviour, edge cases, and failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.datasets import gaussian_mixture
from tinyvec.exact import brute_force, recall_at_k
from tinyvec.index import IVFPQIndex


@pytest.fixture(scope="module")
def small():
    return gaussian_mixture(n=1200, dim=32, n_queries=40, n_train=1200, n_clusters=12, seed=0)


@pytest.fixture(scope="module")
def built(small):
    return IVFPQIndex(nlist=16, m=8, nprobe=4, seed=0, kmeans_iter=10).fit(small.base, small.train)


def test_repr_before_and_after_fit():
    index = IVFPQIndex(nlist=4, m=4)
    assert "unfitted" in repr(index)
    x = gaussian_mixture(n=200, dim=16, seed=0).base
    index.fit(x)
    assert "ntotal=200" in repr(index)


def test_every_vector_lands_in_exactly_one_list(built, small):
    sizes = built.list_sizes()
    assert sizes.sum() == small.n
    all_ids = np.concatenate([entry.ids for entry in built.lists])
    assert np.array_equal(np.sort(all_ids), np.arange(small.n))


def test_codes_have_the_right_shape(built):
    for entry in built.lists:
        assert entry.codes.shape == (entry.ids.size, 8)
        assert entry.codes.dtype == np.uint8


def test_search_returns_requested_shape(built, small):
    ids, dists = built.search(small.queries, 10)
    assert ids.shape == (40, 10)
    assert dists.shape == (40, 10)
    assert ids.dtype == np.int32


def test_search_accepts_a_1d_query(built, small):
    ids, _ = built.search(small.queries[0], 5)
    assert ids.shape == (1, 5)


def test_results_are_sorted_by_distance(built, small):
    _, dists = built.search(small.queries, 10, nprobe=8)
    finite = np.isfinite(dists)
    for row, mask in zip(dists, finite):
        assert np.all(np.diff(row[mask]) >= -1e-4)


def test_recall_increases_with_nprobe(built, small):
    """The defining property of the nprobe dial. Monotone non-decreasing,
    because probing more lists can only add candidates."""
    truth, _ = brute_force(small.base, small.queries, 10)
    recalls = []
    for nprobe in (1, 2, 4, 8, 16):
        ids, _ = built.search(small.queries, 10, nprobe=nprobe)
        recalls.append(recall_at_k(ids, truth, 10))
    assert recalls == sorted(recalls), recalls
    assert recalls[-1] > recalls[0]


def test_candidates_scanned_increases_with_nprobe(built, small):
    counts = []
    for nprobe in (1, 2, 4, 8, 16):
        built.search(small.queries, 10, nprobe=nprobe)
        counts.append(built.last_stats.candidates_scanned)
    assert counts == sorted(counts)
    # At nprobe == nlist every vector is a candidate for every query.
    assert counts[-1] == small.n * small.queries.shape[0]


def test_full_scan_with_rerank_is_exact(built, small):
    """nprobe == nlist removes IVF error and a large rerank removes PQ ranking
    error, so the index must degenerate to exact search."""
    truth, _ = brute_force(small.base, small.queries, 10)
    ids, _ = built.search(small.queries, 10, nprobe=16, rerank=small.n)
    assert recall_at_k(ids, truth, 10) == 1.0


def test_reranked_distances_are_exact(built, small):
    ids, dists = built.search(small.queries, 5, nprobe=16, rerank=200)
    for qi in range(small.queries.shape[0]):
        for rank in range(5):
            want = float(((small.base[ids[qi, rank]] - small.queries[qi]) ** 2).sum())
            assert dists[qi, rank] == pytest.approx(want, rel=1e-4, abs=1e-3)


def test_rerank_never_lowers_recall(built, small):
    """Reranking only reorders candidates ADC already selected, so for
    rerank >= k it cannot evict a true neighbour that was found."""
    truth, _ = brute_force(small.base, small.queries, 10)
    for nprobe in (1, 2, 4, 8):
        plain, _ = built.search(small.queries, 10, nprobe=nprobe, rerank=0)
        ranked, _ = built.search(small.queries, 10, nprobe=nprobe, rerank=200)
        assert recall_at_k(ranked, truth, 10) >= recall_at_k(plain, truth, 10)


def test_stats_report_deterministic_counters(built, small):
    built.search(small.queries, 10, nprobe=3)
    stats = built.last_stats
    assert stats.n_queries == 40
    assert stats.lists_probed == 40 * 3
    assert stats.tables_built <= stats.lists_probed
    assert stats.candidates_per_query == pytest.approx(stats.candidates_scanned / 40)
    assert stats.candidates_reranked == 0


def test_stats_are_identical_across_repeated_identical_searches(built, small):
    built.search(small.queries, 10, nprobe=5)
    first = built.last_stats
    built.search(small.queries, 10, nprobe=5)
    second = built.last_stats
    assert first == second


def test_memory_report_arithmetic(built, small):
    report = built.memory_report()
    assert report["original_float32_bytes"] == 4 * small.n * 32
    assert report["pq_code_bytes"] == small.n * 8
    assert report["compression_ratio"] == pytest.approx(4 * 32 / 8)
    # Folding in codebooks and coarse centroids can only make the ratio worse.
    assert report["compression_ratio_with_overhead"] < report["compression_ratio"]


def test_compression_ratio_matches_real_code_bytes(built):
    actual = sum(entry.codes.nbytes for entry in built.lists)
    assert actual == built.memory_report()["pq_code_bytes"]


# --------------------------------------------------------------- determinism


def test_same_seed_gives_byte_identical_index(small):
    a = IVFPQIndex(nlist=16, m=8, seed=3, kmeans_iter=8).fit(small.base, small.train)
    b = IVFPQIndex(nlist=16, m=8, seed=3, kmeans_iter=8).fit(small.base, small.train)
    assert np.array_equal(a.centroids, b.centroids)
    assert np.array_equal(a.pq.codebooks, b.pq.codebooks)
    assert len(a.lists) == len(b.lists)
    for ea, eb in zip(a.lists, b.lists):
        assert np.array_equal(ea.ids, eb.ids)
        assert np.array_equal(ea.codes, eb.codes)


def test_same_seed_gives_identical_search_results(small):
    kwargs = dict(nlist=16, m=8, nprobe=4, seed=5, kmeans_iter=8)
    a = IVFPQIndex(**kwargs).fit(small.base, small.train)
    b = IVFPQIndex(**kwargs).fit(small.base, small.train)
    ia, da = a.search(small.queries, 10)
    ib, db = b.search(small.queries, 10)
    assert np.array_equal(ia, ib)
    assert np.array_equal(da, db)


def test_different_seeds_give_different_indexes(small):
    a = IVFPQIndex(nlist=16, m=8, seed=1, kmeans_iter=8).fit(small.base, small.train)
    b = IVFPQIndex(nlist=16, m=8, seed=2, kmeans_iter=8).fit(small.base, small.train)
    assert not np.array_equal(a.centroids, b.centroids)


# --------------------------------------------------------------- edge cases


def test_search_before_fit_raises_cleanly():
    index = IVFPQIndex(nlist=4, m=4)
    with pytest.raises(RuntimeError, match="not built"):
        index.search(np.zeros((1, 16), np.float32), 5)
    with pytest.raises(RuntimeError, match="not built"):
        index.memory_report()
    with pytest.raises(RuntimeError, match="not built"):
        index.list_sizes()
    with pytest.raises(RuntimeError, match="not built"):
        index.exact_search(np.zeros((1, 16), np.float32))


def test_fit_on_empty_base_raises():
    with pytest.raises(ValueError, match="zero vectors"):
        IVFPQIndex(nlist=4, m=4).fit(np.zeros((0, 16), np.float32))


def test_fit_on_non_2d_base_raises():
    with pytest.raises(ValueError, match="2-D"):
        IVFPQIndex(nlist=4, m=4).fit(np.zeros(16, np.float32))


def test_train_dimension_mismatch_raises():
    base = np.zeros((50, 16), np.float32)
    with pytest.raises(ValueError, match="incompatible"):
        IVFPQIndex(nlist=4, m=4).fit(base, np.zeros((50, 8), np.float32))


def test_single_vector_index():
    """n=1 forces nlist down to one distinct centroid and makes every query
    return the same answer. It must not crash."""
    x = np.arange(16, dtype=np.float32).reshape(1, 16)
    index = IVFPQIndex(nlist=8, m=4, nprobe=8, seed=0, kmeans_iter=5).fit(x)
    assert index.ntotal == 1
    assert len(index.lists) == 1
    ids, dists = index.search(np.zeros((3, 16), np.float32), 10)
    assert ids.shape == (3, 1)
    assert np.all(ids == 0)
    assert np.all(np.isfinite(dists))


def test_k_larger_than_dataset_clamps():
    x = gaussian_mixture(n=30, dim=16, n_queries=5, n_train=30, n_clusters=3, seed=0)
    index = IVFPQIndex(nlist=4, m=4, nprobe=4, seed=0, kmeans_iter=5).fit(x.base, x.train)
    ids, dists = index.search(x.queries, 500)
    assert ids.shape == (5, 30)
    assert dists.shape == (5, 30)


def test_nprobe_larger_than_nlist_clamps(built, small):
    a, _ = built.search(small.queries, 10, nprobe=16)
    b, _ = built.search(small.queries, 10, nprobe=9999)
    assert np.array_equal(a, b)


def test_query_dimension_mismatch_raises(built):
    with pytest.raises(ValueError, match="expected"):
        built.search(np.zeros((2, 31), np.float32), 5)


def test_non_positive_k_raises(built, small):
    with pytest.raises(ValueError, match="k must be at least 1"):
        built.search(small.queries, 0)


def test_non_positive_nprobe_raises(built, small):
    with pytest.raises(ValueError, match="nprobe must be at least 1"):
        built.search(small.queries, 10, nprobe=0)


def test_negative_rerank_raises(built, small):
    with pytest.raises(ValueError, match="rerank must be non-negative"):
        built.search(small.queries, 10, rerank=-1)


def test_rerank_without_stored_vectors_is_rejected_at_construction():
    with pytest.raises(ValueError, match="rerank requires store_vectors"):
        IVFPQIndex(nlist=4, m=4, rerank=10, store_vectors=False)


def test_rerank_without_stored_vectors_is_rejected_at_search_time(small):
    index = IVFPQIndex(
        nlist=8, m=8, nprobe=4, store_vectors=False, seed=0, kmeans_iter=5
    ).fit(small.base, small.train)
    assert index.vectors is None
    with pytest.raises(RuntimeError, match="store_vectors=True"):
        index.search(small.queries, 10, rerank=50)
    with pytest.raises(RuntimeError, match="store_vectors=True"):
        index.exact_search(small.queries, 10)


def test_store_vectors_false_still_searches(small):
    """Dropping the originals is the configuration that actually realizes the
    compression ratio, so it has to work."""
    index = IVFPQIndex(
        nlist=8, m=8, nprobe=8, store_vectors=False, seed=0, kmeans_iter=5
    ).fit(small.base, small.train)
    ids, _ = index.search(small.queries, 10)
    assert ids.shape == (40, 10)
    assert np.all(ids >= 0)


def test_invalid_constructor_arguments():
    with pytest.raises(ValueError, match="nlist must be at least 1"):
        IVFPQIndex(nlist=0)
    with pytest.raises(ValueError, match="nprobe must be at least 1"):
        IVFPQIndex(nprobe=0)
    with pytest.raises(ValueError, match="rerank must be non-negative"):
        IVFPQIndex(rerank=-5)


def test_nlist_larger_than_n_is_tolerated():
    """k-means clamps to the distinct-row count, so asking for more lists than
    there are vectors yields fewer lists rather than an error."""
    x = gaussian_mixture(n=20, dim=16, n_queries=3, n_train=20, n_clusters=2, seed=0)
    index = IVFPQIndex(nlist=500, m=4, nprobe=500, seed=0, kmeans_iter=5).fit(x.base, x.train)
    assert len(index.lists) <= 20
    ids, _ = index.search(x.queries, 5)
    assert ids.shape == (3, 5)
    assert np.all(ids >= 0)


def test_exact_search_matches_brute_force(built, small):
    a, _ = built.exact_search(small.queries, 10)
    b, _ = brute_force(small.base, small.queries, 10)
    assert np.array_equal(a, b)
