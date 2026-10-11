"""The precomputed-table optimization must be a pure rearrangement.

The optimization folds two of the three terms of the expanded IVF-PQ distance
into a build-time table. That is valid algebra, but it is algebra applied to
float32, so the only thing worth testing is that it changes results by rounding
and nothing more. If it ever changes which neighbours come back, the speedup is
buying accuracy loss and the benchmark comparison is invalid.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.datasets import gaussian_mixture
from tinyvec.exact import brute_force, recall_at_k
from tinyvec.index import IVFPQIndex

PARAMS = dict(nlist=16, m=8, nprobe=4, seed=0, kmeans_iter=10)


@pytest.fixture(scope="module")
def dataset():
    return gaussian_mixture(n=1500, dim=32, n_queries=50, n_train=1500, n_clusters=12, seed=0)


@pytest.fixture(scope="module")
def pair(dataset):
    fast = IVFPQIndex(precompute_tables=True, **PARAMS).fit(dataset.base, dataset.train)
    slow = IVFPQIndex(precompute_tables=False, **PARAMS).fit(dataset.base, dataset.train)
    return fast, slow


def test_both_paths_build_the_same_index(pair):
    """The optimization must not perturb the index itself, only the scoring
    route, so centroids and codes have to be bit-identical."""
    fast, slow = pair
    assert np.array_equal(fast.centroids, slow.centroids)
    assert np.array_equal(fast.pq.codebooks, slow.pq.codebooks)
    for a, b in zip(fast.lists, slow.lists):
        assert np.array_equal(a.ids, b.ids)
        assert np.array_equal(a.codes, b.codes)


@pytest.mark.parametrize("nprobe", [1, 2, 4, 8, 16])
def test_returned_ids_are_identical(pair, dataset, nprobe):
    fast, slow = pair
    fast_ids, _ = fast.search(dataset.queries, 10, nprobe=nprobe)
    slow_ids, _ = slow.search(dataset.queries, 10, nprobe=nprobe)
    assert np.array_equal(fast_ids, slow_ids)


@pytest.mark.parametrize("nprobe", [1, 4, 16])
def test_distances_agree_to_float32_rounding(pair, dataset, nprobe):
    fast, slow = pair
    _, fast_d = fast.search(dataset.queries, 10, nprobe=nprobe)
    _, slow_d = slow.search(dataset.queries, 10, nprobe=nprobe)
    scale = max(float(np.max(slow_d[np.isfinite(slow_d)])), 1.0)
    assert np.max(np.abs(fast_d - slow_d)) / scale < 1e-4


def test_recall_is_identical(pair, dataset):
    fast, slow = pair
    truth, _ = brute_force(dataset.base, dataset.queries, 10)
    for nprobe in (1, 2, 4, 8, 16):
        a = recall_at_k(fast.search(dataset.queries, 10, nprobe=nprobe)[0], truth, 10)
        b = recall_at_k(slow.search(dataset.queries, 10, nprobe=nprobe)[0], truth, 10)
        assert a == b, f"nprobe={nprobe}: {a} vs {b}"


def test_candidate_counts_are_identical(pair, dataset):
    """Same work, just arranged differently, so the deterministic counters that
    CI gates on must not move either."""
    fast, slow = pair
    for nprobe in (1, 4, 16):
        fast.search(dataset.queries, 10, nprobe=nprobe)
        slow.search(dataset.queries, 10, nprobe=nprobe)
        assert fast.last_stats == slow.last_stats


def test_rerank_is_unaffected(pair, dataset):
    fast, slow = pair
    a, _ = fast.search(dataset.queries, 10, nprobe=8, rerank=100)
    b, _ = slow.search(dataset.queries, 10, nprobe=8, rerank=100)
    assert np.array_equal(a, b)


def test_precompute_table_size_matches_the_formula(pair):
    """nlist * m * 256 * 4 bytes. This term is independent of n, which is the
    whole reason it has to be reported separately from the compression ratio."""
    fast, slow = pair
    report = fast.memory_report()
    assert report["precomputed_table_bytes"] == 16 * 8 * 256 * 4
    assert slow.memory_report()["precomputed_table_bytes"] == 0.0


def test_precompute_worsens_the_reported_memory_ratio(pair):
    """Honesty check on memory_report: the optimization costs real bytes and the
    accounting has to show it rather than quoting the codes-only figure."""
    fast, slow = pair
    assert fast.memory_report()["compression_ratio"] == slow.memory_report()["compression_ratio"]
    assert (
        fast.memory_report()["compression_ratio_with_overhead"]
        < slow.memory_report()["compression_ratio_with_overhead"]
    )


def test_index_byte_total_is_the_sum_of_its_parts(pair):
    for index in pair:
        report = index.memory_report()
        total = (
            report["pq_code_bytes"]
            + report["id_bytes"]
            + report["codebook_bytes"]
            + report["coarse_centroid_bytes"]
            + report["precomputed_table_bytes"]
        )
        assert report["index_bytes_excluding_vectors"] == pytest.approx(total)


def test_repr_reports_the_mode(pair):
    fast, slow = pair
    assert "precompute=True" in repr(fast)
    assert "precompute=False" in repr(slow)


# ------------------------------------------------- the decomposition itself


def test_decomposition_identity_holds_term_by_term(dataset):
    """Verify the algebra directly rather than only through the index:

        d(q, c + r) = ||q - c||^2 + sum_j ( ||C_j[k]||^2
                                            + 2 c_j . C_j[k]
                                            - 2 q_j . C_j[k] )
    """
    from tinyvec.pq import ProductQuantizer

    rng = np.random.default_rng(0)
    pq = ProductQuantizer(8, seed=0, n_iter=10).fit(dataset.train)
    centroid = rng.normal(size=32).astype(np.float32)
    queries = rng.normal(size=(4, 32)).astype(np.float32)
    codes = pq.encode(dataset.base[:200] - centroid)
    decoded = pq.decode(codes).astype(np.float64)

    table = (
        pq.codebook_sq_norms
        + 2.0 * pq.subspace_dots(centroid)[0]
        - 2.0 * pq.subspace_dots(queries)
    )
    for i in range(4):
        got = float(((queries[i] - centroid) ** 2).sum()) + pq.adc(codes, table[i])
        want = ((decoded + centroid.astype(np.float64) - queries[i].astype(np.float64)) ** 2).sum(
            axis=1
        )
        assert np.max(np.abs(got - want) / np.maximum(want, 1e-9)) < 1e-4


def test_subspace_dots_matches_direct_inner_products(dataset):
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(4, seed=0, n_iter=5).fit(dataset.train)
    vectors = dataset.base[:6]
    dots = pq.subspace_dots(vectors)
    assert dots.shape == (6, 4, 256)
    for i in range(6):
        for j in range(4):
            sub = vectors[i, j * 8 : (j + 1) * 8]
            assert np.allclose(dots[i, j], pq.codebooks[j] @ sub, rtol=1e-4, atol=1e-3)


def test_codebook_sq_norms_matches_direct_computation(dataset):
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(4, seed=0, n_iter=5).fit(dataset.train)
    norms = pq.codebook_sq_norms
    assert norms.shape == (4, 256)
    for j in range(4):
        assert np.allclose(norms[j], (pq.codebooks[j] ** 2).sum(axis=1), rtol=1e-5)


def test_subspace_dots_accepts_a_1d_vector(dataset):
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(4, seed=0, n_iter=5).fit(dataset.train)
    assert pq.subspace_dots(dataset.base[0]).shape == (1, 4, 256)


def test_precompute_helpers_require_a_fitted_quantizer():
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(4)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = pq.codebook_sq_norms
    with pytest.raises(RuntimeError, match="not fitted"):
        pq.subspace_dots(np.zeros((1, 16), np.float32))


# ------------------------------------------------------- adc method agreement


def test_adc_methods_agree(dataset):
    """Both gather implementations are selected automatically by list length, so
    they must produce the same numbers at every length."""
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(8, seed=0, n_iter=10).fit(dataset.train)
    table = pq.distance_tables(dataset.queries[:1])[0]
    for n in (1, 7, 255, 256, 511, 512, 1500):
        codes = pq.encode(dataset.base[:n])
        loop = pq.adc(codes, table, method="loop")
        fancy = pq.adc(codes, table, method="fancy")
        auto = pq.adc(codes, table)
        assert np.allclose(loop, fancy, rtol=1e-5, atol=1e-3), n
        assert np.allclose(auto, loop, rtol=1e-5, atol=1e-3), n


def test_adc_auto_picks_the_measured_winner(dataset):
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(8, seed=0, n_iter=5).fit(dataset.train)
    assert pq.ADC_LOOP_THRESHOLD == 512


def test_adc_rejects_an_unknown_method(dataset):
    from tinyvec.pq import ProductQuantizer

    pq = ProductQuantizer(8, seed=0, n_iter=5).fit(dataset.train)
    table = pq.distance_tables(dataset.queries[:1])[0]
    codes = pq.encode(dataset.base[:10])
    with pytest.raises(ValueError, match="method must be"):
        pq.adc(codes, table, method="turbo")
