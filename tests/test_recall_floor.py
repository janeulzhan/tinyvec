"""Sealed-seed recall regression test.

This is what turns the benchmark from a one-off claim into something CI can
defend. The dataset is fixed (vendored SIFT-small, not generated), the index
parameters are fixed, and the seed is fixed, so recall at 10 is a deterministic
function of the committed code. If a refactor degrades retrieval quality, this
fails.

The floors are set a few points below the values measured on the authoring
machine, which are recorded alongside each floor. The margin is not decoration:
recall here depends on k-means centroids, which depend on float32 GEMM
accumulation order, which differs between BLAS implementations. A test asserting
exact equality would pass locally and flake on any runner with a different
numpy build. A floor with a stated margin is the strongest assertion that is
actually true across platforms.

Only deterministic quantities are asserted on. There is no timing assertion
anywhere in this file by design: see the README section on what CI gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.datasets import load_siftsmall
from tinyvec.exact import brute_force, recall_at_k, tie_margin
from tinyvec.index import IVFPQIndex
from tinyvec.kmeans import assign

# The sealed configuration. Changing any of these invalidates the floors below.
SEED = 42
NLIST = 128
M = 16
KMEANS_ITER = 25
K = 10

# Measured on Apple M4, Python 3.11.9, numpy 2.4.6.
# Floors sit roughly 3 points below measured to absorb BLAS variation.
MEASURED_NO_RERANK = {1: 0.403, 2: 0.548, 4: 0.648, 8: 0.700, 16: 0.723, 32: 0.724}
FLOOR_NO_RERANK = {1: 0.37, 2: 0.51, 4: 0.61, 8: 0.66, 16: 0.69, 32: 0.69}

MEASURED_RERANK_100 = {1: 0.444, 2: 0.642, 4: 0.808, 8: 0.925, 16: 0.988, 32: 0.999}
FLOOR_RERANK_100 = {1: 0.40, 2: 0.60, 4: 0.77, 8: 0.89, 16: 0.96, 32: 0.98}

# Measured 32.0 exactly; it is 4 * dim / m and carries no float dependence.
EXPECTED_COMPRESSION = 32.0
# Measured mean squared residual reconstruction error: 12071.06.
RECONSTRUCTION_ERROR_CEILING = 13_500.0


@pytest.fixture(scope="module")
def sealed():
    dataset = load_siftsmall()
    index = IVFPQIndex(
        nlist=NLIST, m=M, nprobe=8, seed=SEED, kmeans_iter=KMEANS_ITER
    ).fit(dataset.base, dataset.train)
    truth, truth_dists = brute_force(dataset.base, dataset.queries, K + 1)
    return dataset, index, truth, truth_dists


def test_dataset_is_the_expected_real_corpus(sealed):
    """Guards against the floors below being silently re-interpreted against a
    different or regenerated dataset."""
    dataset, _, _, _ = sealed
    assert dataset.name == "sift-small"
    assert dataset.synthetic is False
    assert dataset.base.shape == (10_000, 128)
    assert dataset.queries.shape == (100, 128)
    assert dataset.train.shape == (25_000, 128)
    assert dataset.metric == "euclidean"


def test_groundtruth_top10_is_unambiguous(sealed):
    """Recall at 10 is only well defined if the 10th and 11th neighbours are
    distinguishable. Checked before any recall is asserted on."""
    _, _, _, truth_dists = sealed
    assert tie_margin(truth_dists, K) > 0.0


@pytest.mark.parametrize("nprobe", sorted(FLOOR_NO_RERANK))
def test_recall_floor_without_rerank(sealed, nprobe):
    dataset, index, truth, _ = sealed
    ids, _ = index.search(dataset.queries, K, nprobe=nprobe, rerank=0)
    recall = recall_at_k(ids, truth, K)
    floor = FLOOR_NO_RERANK[nprobe]
    assert recall >= floor, (
        f"recall@{K} at nprobe={nprobe} fell to {recall:.4f}, below the declared "
        f"floor {floor} (measured {MEASURED_NO_RERANK[nprobe]} when sealed)"
    )


@pytest.mark.parametrize("nprobe", sorted(FLOOR_RERANK_100))
def test_recall_floor_with_rerank(sealed, nprobe):
    dataset, index, truth, _ = sealed
    ids, _ = index.search(dataset.queries, K, nprobe=nprobe, rerank=100)
    recall = recall_at_k(ids, truth, K)
    floor = FLOOR_RERANK_100[nprobe]
    assert recall >= floor, (
        f"recall@{K} at nprobe={nprobe}, rerank=100 fell to {recall:.4f}, below the "
        f"declared floor {floor} (measured {MEASURED_RERANK_100[nprobe]} when sealed)"
    )


def test_recall_is_monotone_in_nprobe(sealed):
    dataset, index, truth, _ = sealed
    recalls = [
        recall_at_k(index.search(dataset.queries, K, nprobe=p, rerank=0)[0], truth, K)
        for p in sorted(FLOOR_NO_RERANK)
    ]
    assert recalls == sorted(recalls), recalls


def test_rerank_dominates_plain_adc_at_every_nprobe(sealed):
    dataset, index, truth, _ = sealed
    for nprobe in sorted(FLOOR_NO_RERANK):
        plain = recall_at_k(index.search(dataset.queries, K, nprobe=nprobe, rerank=0)[0], truth, K)
        ranked = recall_at_k(
            index.search(dataset.queries, K, nprobe=nprobe, rerank=100)[0], truth, K
        )
        assert ranked >= plain, f"rerank hurt recall at nprobe={nprobe}"


def test_pq_alone_caps_recall_below_one(sealed):
    """At nprobe=nlist the inverted file scans everything, so it contributes no
    error at all. Recall is still well under 1.0, which localizes the remaining
    loss entirely to PQ misranking. This is the measurement that justifies
    shipping a reranking stage, so it is worth pinning."""
    dataset, index, truth, _ = sealed
    ids, _ = index.search(dataset.queries, K, nprobe=NLIST, rerank=0)
    full_scan_recall = recall_at_k(ids, truth, K)
    assert index.last_stats.candidates_scanned == dataset.n * dataset.queries.shape[0]
    assert 0.60 < full_scan_recall < 0.90, full_scan_recall

    ranked, _ = index.search(dataset.queries, K, nprobe=NLIST, rerank=100)
    assert recall_at_k(ranked, truth, K) > 0.98


def test_compression_ratio_is_exactly_as_declared(sealed):
    _, index, _, _ = sealed
    report = index.memory_report()
    assert report["compression_ratio"] == pytest.approx(EXPECTED_COMPRESSION)
    assert report["pq_code_bytes"] == 10_000 * M
    assert report["original_float32_bytes"] == 4 * 10_000 * 128


def test_residual_reconstruction_error_ceiling(sealed):
    """Deterministic quality metric for the quantizer itself, independent of any
    retrieval measurement."""
    dataset, index, _, _ = sealed
    labels, _ = assign(dataset.base, index.centroids)
    residuals = dataset.base - index.centroids[labels]
    error = index.pq.reconstruction_error(residuals)
    assert error < RECONSTRUCTION_ERROR_CEILING, error


def test_inverted_lists_are_not_pathologically_unbalanced(sealed):
    """A single list holding most of the database would make nprobe meaningless
    while still reporting plausible recall, so partition balance is pinned too."""
    dataset, index, _, _ = sealed
    sizes = index.list_sizes()
    assert sizes.sum() == dataset.n
    assert sizes.max() < 0.10 * dataset.n


def test_search_is_reproducible_within_a_process(sealed):
    dataset, index, _, _ = sealed
    first = index.search(dataset.queries, K, nprobe=8, rerank=100)
    second = index.search(dataset.queries, K, nprobe=8, rerank=100)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


@pytest.mark.slow
def test_rebuilding_the_sealed_index_reproduces_it_exactly(sealed):
    """Determinism across a full independent rebuild, not just across searches."""
    dataset, index, truth, _ = sealed
    rebuilt = IVFPQIndex(
        nlist=NLIST, m=M, nprobe=8, seed=SEED, kmeans_iter=KMEANS_ITER
    ).fit(dataset.base, dataset.train)
    assert np.array_equal(rebuilt.centroids, index.centroids)
    assert np.array_equal(rebuilt.pq.codebooks, index.pq.codebooks)
    for a, b in zip(rebuilt.lists, index.lists):
        assert np.array_equal(a.ids, b.ids)
        assert np.array_equal(a.codes, b.codes)
