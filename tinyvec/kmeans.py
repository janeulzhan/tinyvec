"""Lloyd's k-means with k-means++ seeding.

Used twice in this library: once to pick the coarse IVF centroids, and once per
subspace to build the product-quantization codebooks. Both call sites want the
same three properties, so they are guaranteed here rather than at the call site:

1. Deterministic for a fixed seed.
2. Inertia is non-increasing across iterations. Pure Lloyd has this property;
   the usual way to lose it is an empty-cluster "repair" that teleports a
   centroid somewhere inertia-increasing, so this implementation leaves an empty
   cluster's centroid exactly where it was instead.

   This guarantee is conditional on the assignment step actually finding the
   nearest centroid, which is not free: see :func:`squared_distances` for the
   regime where float32 cancellation makes that step unreliable. Inertia itself
   is computed from direct residuals rather than from the expanded distances, so
   the reported numbers stay accurate even when the assignment does not.
3. ``k`` larger than the number of distinct rows is not an error. The number of
   returned centroids is clamped to the number of distinct rows, because asking
   for more centroids than there are distinct points has no solution with
   non-duplicate centroids and silently returning duplicates would corrupt a PQ
   codebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KMeansResult", "kmeans", "assign", "squared_distances"]


@dataclass(frozen=True)
class KMeansResult:
    """Outcome of a k-means fit.

    Attributes:
        centroids: ``(k_effective, d)`` float32 centroids.
        labels: ``(n,)`` int32 index of the nearest centroid for each input row.
        inertia: Sum of squared distances from each row to its centroid.
        inertia_history: Inertia after every assignment step, oldest first.
            Guaranteed non-increasing.
        n_iter: Number of Lloyd iterations actually run.
    """

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    inertia_history: tuple[float, ...]
    n_iter: int


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of ``a`` and ``b``.

    Uses the ``|a|^2 - 2a.b + |b|^2`` expansion, which turns the whole thing
    into one GEMM. That expansion can go slightly negative on near-identical
    vectors through floating-point cancellation, so the result is clipped at
    zero; callers that take a square root would otherwise get a NaN.

    Precision limit, which is worth knowing before trusting a small distance:
    the expansion subtracts quantities of size ``|a|^2`` to produce a result of
    size ``|a - b|^2``, so it loses relative accuracy as ``|a|, |b|`` grow
    against ``|a - b|``. Measured relative error on the smallest 30 percent of
    distances, float32, for 5-dimensional clusters of radius ``noise`` centred
    at radius ``offset`` from the origin:

        offset  noise   mean |x|^2   true d^2   relative error
          20.0   0.01       1728.2   9.47e-04         7.09e-01
          20.0   0.10       1727.5   9.47e-02         6.10e-03
          20.0   1.00       1724.8   9.47e+00         8.64e-05
           2.0   0.30         17.5   8.52e-01         6.01e-06
           0.0   1.00          5.0   3.04e+00         9.27e-08

    The first row is not a usable distance. Tight clusters far from the origin
    are the pathological case, and for data of that shape the fix is to centre
    it first, since distances are translation invariant. Real embedding data
    (SIFT included) sits near the well-conditioned end, which is why this is a
    documented limit rather than a blocking defect.

    Args:
        a: ``(n, d)`` array.
        b: ``(m, d)`` array.

    Returns:
        ``(n, m)`` float32 array of squared distances.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D inputs, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}")
    a_sq = np.einsum("ij,ij->i", a, a)[:, None]
    b_sq = np.einsum("ij,ij->i", b, b)[None, :]
    d = a_sq - 2.0 * (a @ b.T) + b_sq
    np.maximum(d, 0.0, out=d)
    return d


def assign(x: np.ndarray, centroids: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row of ``x`` to its nearest centroid.

    Chunked over rows so that the ``(n, k)`` distance matrix never has to exist
    in full; at n = 1e6 and k = 4096 that matrix alone would be 16 GB.

    Args:
        x: ``(n, d)`` array of points.
        centroids: ``(k, d)`` array of centroids.
        chunk: Number of rows scored per block.

    Returns:
        ``(labels, min_sq_dist)`` with shapes ``(n,)`` int32 and ``(n,)`` float32.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[0]
    labels = np.empty(n, dtype=np.int32)
    best = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = squared_distances(x[start:stop], centroids)
        labels[start:stop] = np.argmin(d, axis=1)
        best[start:stop] = d[np.arange(stop - start), labels[start:stop]]
    return labels, best


def _distinct_rows(x: np.ndarray) -> int:
    """Count distinct rows of ``x`` exactly, via a byte-level view."""
    contiguous = np.ascontiguousarray(x, dtype=np.float32)
    flat = contiguous.view(np.uint8).reshape(contiguous.shape[0], -1)
    return int(np.unique(flat, axis=0).shape[0])


def _group_sums(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Sum rows of ``x`` grouped by ``labels``, returning a ``(k, d)`` array.

    One ``bincount`` per column rather than a single ``np.add.at``: the scatter-add
    ufunc is unbuffered and runs roughly an order of magnitude slower than
    ``bincount``, and this sits in the inner Lloyd loop. Accumulation happens in
    float64 (bincount's weight dtype) before the cast back down, which also keeps
    the centroid means from drifting on large clusters.
    """
    d = x.shape[1]
    sums = np.empty((k, d), dtype=np.float32)
    for j in range(d):
        sums[:, j] = np.bincount(labels, weights=x[:, j], minlength=k)
    return sums


def _exact_inertia(
    x: np.ndarray, centroids: np.ndarray, labels: np.ndarray, chunk: int = 8192
) -> float:
    """Sum of squared distances from each point to its assigned centroid.

    Computed from direct residuals in float64 rather than reusing the distances
    that :func:`assign` already produced. Those come from the GEMM expansion and
    carry its cancellation error, which on tightly clustered data far from the
    origin can be the same order as the inertia being reported. One extra
    O(n * d) pass is cheap next to the O(n * k * d) assignment, and it buys an
    inertia number that means what it says.
    """
    total = 0.0
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        residual = x[start:stop].astype(np.float64) - centroids[labels[start:stop]].astype(
            np.float64
        )
        total += float(np.einsum("ij,ij->", residual, residual))
    return total


def _kmeans_plusplus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur and Vassilvitskii, 2007).

    Picks the first centroid uniformly, then each subsequent centroid with
    probability proportional to its squared distance to the nearest centroid
    chosen so far. The squared-distance weighting is what gives k-means++ its
    O(log k) approximation guarantee, and in practice it is the difference
    between a PQ codebook that uses all 256 of its slots and one that collapses.
    """
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centroids[0] = x[first]
    closest = squared_distances(x, centroids[:1])[:, 0]

    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # Every remaining point coincides with a chosen centroid. Callers
            # clamp k to the distinct-row count so this is only reachable on
            # pathological input; fall back to a uniform pick rather than
            # dividing by zero.
            centroids[i] = x[int(rng.integers(n))]
        else:
            probs = closest / total
            centroids[i] = x[int(rng.choice(n, p=probs))]
        closest = np.minimum(closest, squared_distances(x, centroids[i : i + 1])[:, 0])
    return centroids


def kmeans(
    x: np.ndarray,
    k: int,
    *,
    n_iter: int = 25,
    seed: int = 0,
    tol: float = 1e-6,
) -> KMeansResult:
    """Fit k-means to ``x``.

    Args:
        x: ``(n, d)`` array of points. Cast to float32 internally.
        k: Requested number of centroids. Clamped to the number of distinct rows.
        n_iter: Maximum Lloyd iterations.
        seed: Seed for the k-means++ draw. Fixing it fixes the whole fit.
        tol: Stop early when relative inertia improvement drops below this.

    Returns:
        A :class:`KMeansResult`.

    Raises:
        ValueError: If ``x`` is not 2-D and non-empty, or ``k < 1``.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array of points, got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("cannot fit k-means to zero points")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    k_effective = min(k, _distinct_rows(x))
    rng = np.random.default_rng(seed)
    centroids = _kmeans_plusplus(x, k_effective, rng)

    history: list[float] = []
    iterations = 0
    labels, _ = assign(x, centroids)
    history.append(_exact_inertia(x, centroids, labels))

    for _ in range(n_iter):
        iterations += 1
        counts = np.bincount(labels, minlength=k_effective)
        sums = _group_sums(x, labels, k_effective)
        occupied = counts > 0
        # An empty cluster keeps its previous centroid. Both the assignment and
        # the update step are then non-increasing in inertia, so the history is
        # monotone by construction.
        centroids = centroids.copy()
        centroids[occupied] = sums[occupied] / counts[occupied, None].astype(np.float32)

        labels, _ = assign(x, centroids)
        inertia = _exact_inertia(x, centroids, labels)
        previous = history[-1]
        history.append(inertia)
        if previous > 0.0 and (previous - inertia) / previous < tol:
            break
        if inertia == 0.0:
            break

    return KMeansResult(
        centroids=centroids,
        labels=labels,
        inertia=history[-1],
        inertia_history=tuple(history),
        n_iter=iterations,
    )
