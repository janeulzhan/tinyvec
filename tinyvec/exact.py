"""Exact brute-force search and the recall metric.

This module is the measuring stick for everything else, so it is deliberately
the dullest code in the repository. It computes true nearest neighbours by
scoring every query against every base vector. The only concession to
performance is chunking, which does not change the result.

One correctness note that matters for recall accounting: ties. If the k-th and
(k+1)-th true neighbours sit at exactly the same distance, then which of them
belongs in "the" ground-truth top-k is arbitrary, and an index that returns the
other one gets scored as a miss despite being equally correct. :func:`recall_at_k`
does the standard thing and intersects id sets, so it inherits that ambiguity;
:func:`tie_margin` is the guard that tells you whether the ambiguity is actually
present in your data. Measure it once per dataset and report it. A recall number
published without that check is a recall number you cannot defend.
"""

from __future__ import annotations

import numpy as np

from tinyvec.kmeans import squared_distances

__all__ = ["brute_force", "recall_at_k", "tie_margin"]


def brute_force(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours under squared Euclidean distance.

    Args:
        base: ``(n, d)`` database vectors.
        queries: ``(nq, d)`` query vectors.
        k: Number of neighbours. Clamped to ``n``.
        chunk: Queries scored per block.

    Returns:
        ``(ids, dists)`` with shapes ``(nq, min(k, n))``, sorted nearest first.
        ``dists`` are squared distances.

    Raises:
        ValueError: If ``base`` is empty or ``k < 1``.
    """
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError("base and queries must both be 2-D")
    if base.shape[0] == 0:
        raise ValueError("cannot search an empty base set")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    k = min(k, base.shape[0])

    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int32)
    dists = np.empty((nq, k), dtype=np.float32)

    for start in range(0, nq, chunk):
        stop = min(start + chunk, nq)
        d = squared_distances(queries[start:stop], base)
        # argpartition is O(n) per query to isolate the top k, then only those k
        # get sorted. A full argsort would be O(n log n) for a result that throws
        # away all but k of it.
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part, axis=1)
        order = np.argsort(part_d, axis=1, kind="stable")
        ids[start:stop] = np.take_along_axis(part, order, axis=1)
        dists[start:stop] = np.take_along_axis(part_d, order, axis=1)

    return ids, dists


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of the true top-k that a result set recovered.

    Averaged over queries, which is the ann-benchmarks convention. With equal
    ``k`` on both sides this is identical to precision at k, which is why the
    literature often writes the two interchangeably for this measurement.

    Args:
        found: ``(nq, kf)`` returned ids. Only the first ``k`` columns are used.
        truth: ``(nq, kt)`` exact ids. Only the first ``k`` columns are used.
        k: Cutoff.

    Returns:
        Recall in ``[0, 1]``.

    Raises:
        ValueError: On a query-count mismatch, or if either side has fewer than
            ``k`` columns, which would silently deflate the score.
    """
    found = np.asarray(found)
    truth = np.asarray(truth)
    if found.ndim != 2 or truth.ndim != 2:
        raise ValueError("found and truth must both be 2-D")
    if found.shape[0] != truth.shape[0]:
        raise ValueError(f"query count mismatch: {found.shape[0]} vs {truth.shape[0]}")
    if found.shape[1] < k or truth.shape[1] < k:
        raise ValueError(
            f"need at least k={k} columns on both sides, "
            f"got found={found.shape[1]} truth={truth.shape[1]}"
        )
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    hits = 0
    for row_found, row_truth in zip(found[:, :k], truth[:, :k]):
        hits += len(set(row_found.tolist()) & set(row_truth.tolist()))
    return hits / (found.shape[0] * k)


def tie_margin(dists: np.ndarray, k: int) -> float:
    """Smallest gap between the k-th and (k+1)-th exact distance, over queries.

    A value at or below zero means at least one query has an ambiguous top-k, so
    recall at k is not uniquely defined for it. Call this before trusting a
    recall number on an unfamiliar dataset.

    Args:
        dists: ``(nq, kd)`` sorted exact squared distances with ``kd > k``.
        k: Cutoff to check.

    Returns:
        The minimum gap across queries.
    """
    dists = np.asarray(dists, dtype=np.float64)
    if dists.shape[1] <= k:
        raise ValueError(f"need more than k={k} distance columns to measure a tie margin")
    return float(np.min(dists[:, k] - dists[:, k - 1]))
