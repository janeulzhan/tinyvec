"""Product quantization with asymmetric distance computation.

Product quantization (Jegou, Douze and Schmid, 2011) splits a d-dimensional
vector into ``m`` contiguous subvectors and quantizes each one independently
against its own codebook of ``2**nbits`` centroids. A vector is then stored as
``m`` codebook indices. With the usual ``nbits = 8`` that is ``m`` bytes, so a
128-dimensional float32 vector (512 bytes) stored with ``m = 16`` becomes 16
bytes: a 32x reduction, and the compression ratio is exactly ``4 * d / m``
independent of the data.

The reason PQ is worth implementing rather than just compressing is asymmetric
distance computation. The query is *not* quantized. Instead, for a query ``q``
we precompute, once per query, the squared distance from each of ``q``'s ``m``
subvectors to each of the ``2**nbits`` centroids in that subspace. That is an
``(m, 2**nbits)`` lookup table. Because squared L2 decomposes as a sum over
disjoint coordinate blocks, the approximate distance from ``q`` to any stored
code is then an ``m``-entry table gather and sum:

    d(q, x_hat)^2 = sum over j of table[j, code[j]]

So scoring a list of N vectors costs N*m table lookups and no arithmetic over
the original d dimensions at all. At d = 128 and m = 16 that is 8x fewer
memory touches per candidate than an exact distance, on data that is also 32x
smaller and therefore far more cache-resident. Keeping the query unquantized is
what makes the estimate asymmetric, and it is strictly more accurate than
quantizing both sides: only one of the two vectors carries quantization error.
"""

from __future__ import annotations

import numpy as np

from tinyvec.kmeans import kmeans, squared_distances

__all__ = ["ProductQuantizer"]


class ProductQuantizer:
    """Trains PQ codebooks and computes asymmetric distances against them.

    Args:
        m: Number of subspaces. Must divide the training dimension.
        nbits: Bits per subquantizer code. Only 8 is supported, which is what
            lets codes live in a ``uint8`` array and keeps each lookup table at
            256 entries, small enough to stay in L1 while a list is scanned.
        n_iter: Lloyd iterations per subspace codebook.
        seed: Base seed. Subspace ``j`` is trained with ``seed + j`` so that the
            codebooks are independent but the whole fit is reproducible.

    Attributes:
        codebooks: ``(m, ksub, dsub)`` float32, available after :meth:`fit`.
    """

    def __init__(self, m: int = 8, *, nbits: int = 8, n_iter: int = 25, seed: int = 0) -> None:
        if m < 1:
            raise ValueError(f"m must be at least 1, got {m}")
        if nbits != 8:
            raise ValueError(f"only nbits=8 is supported, got {nbits}")
        self.m = int(m)
        self.nbits = int(nbits)
        self.ksub = 1 << int(nbits)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.codebooks: np.ndarray | None = None
        self.dim: int | None = None
        self.dsub: int | None = None
        self._subspace_index = np.arange(self.m)

    @property
    def is_fitted(self) -> bool:
        return self.codebooks is not None

    @property
    def code_size(self) -> int:
        """Bytes of storage per encoded vector."""
        return self.m

    def compression_ratio(self, dim: int | None = None) -> float:
        """Ratio of float32 storage to PQ-code storage, ``4 * dim / m``."""
        d = self.dim if dim is None else dim
        if d is None:
            raise RuntimeError("compression_ratio needs a fitted quantizer or an explicit dim")
        return 4.0 * d / self.m

    def _require_fitted(self) -> np.ndarray:
        if self.codebooks is None:
            raise RuntimeError("ProductQuantizer is not fitted; call fit() first")
        return self.codebooks

    def fit(self, x: np.ndarray) -> "ProductQuantizer":
        """Train one codebook per subspace.

        Args:
            x: ``(n, d)`` training vectors. ``d`` must be divisible by ``m``.

        Returns:
            ``self``.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected a 2-D training array, got shape {x.shape}")
        n, d = x.shape
        if n == 0:
            raise ValueError("cannot train a product quantizer on zero vectors")
        if d % self.m != 0:
            raise ValueError(f"m={self.m} does not divide dim={d}")

        self.dim = d
        self.dsub = d // self.m
        codebooks = np.zeros((self.m, self.ksub, self.dsub), dtype=np.float32)

        for j in range(self.m):
            sub = x[:, j * self.dsub : (j + 1) * self.dsub]
            result = kmeans(sub, self.ksub, n_iter=self.n_iter, seed=self.seed + j)
            got = result.centroids.shape[0]
            codebooks[j, :got] = result.centroids
            if got < self.ksub:
                # Fewer distinct subvectors than codebook slots. The unused rows
                # would otherwise sit at the origin and attract encodings, so
                # fill them with a real centroid and let them stay unused.
                codebooks[j, got:] = result.centroids[0]

        self.codebooks = codebooks
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Quantize vectors to PQ codes.

        Args:
            x: ``(n, d)`` vectors.

        Returns:
            ``(n, m)`` uint8 codes.
        """
        codebooks = self._require_fitted()
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) input, got shape {x.shape}")
        assert self.dsub is not None
        codes = np.empty((x.shape[0], self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = x[:, j * self.dsub : (j + 1) * self.dsub]
            codes[:, j] = np.argmin(squared_distances(sub, codebooks[j]), axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate vectors from PQ codes.

        Args:
            codes: ``(n, m)`` uint8 codes.

        Returns:
            ``(n, d)`` float32 reconstructions.
        """
        codebooks = self._require_fitted()
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.m:
            raise ValueError(f"expected (n, {self.m}) codes, got shape {codes.shape}")
        assert self.dim is not None and self.dsub is not None
        out = np.empty((codes.shape[0], self.dim), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self.dsub : (j + 1) * self.dsub] = codebooks[j][codes[:, j]]
        return out

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Mean squared reconstruction error of ``x`` under this quantizer."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        approx = self.decode(self.encode(x))
        return float(np.mean(np.sum((x - approx) ** 2, axis=1)))

    @property
    def codebook_sq_norms(self) -> np.ndarray:
        """``(m, ksub)`` squared norms of every codebook entry.

        One of the three terms the precomputed-table decomposition in
        :mod:`tinyvec.index` needs. Depends on neither the query nor the coarse
        centroid, so it is computed on demand and reused.
        """
        codebooks = self._require_fitted()
        return np.ascontiguousarray(np.einsum("jkl,jkl->jk", codebooks, codebooks))

    def subspace_dots(self, vectors: np.ndarray) -> np.ndarray:
        """Inner products between each subvector and every codebook entry.

        Args:
            vectors: ``(n, d)`` or ``(d,)`` vectors.

        Returns:
            ``(n, m, ksub)`` float32 where ``out[i, j, c]`` is
            ``vectors[i, block j] . codebooks[j, c]``.

        This is the only query-dependent term in the precomputed-table
        decomposition, which is what lets the per-list table build collapse to a
        single array addition.
        """
        codebooks = self._require_fitted()
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) input, got shape {vectors.shape}")
        assert self.dsub is not None
        out = np.empty((vectors.shape[0], self.m, self.ksub), dtype=np.float32)
        for j in range(self.m):
            sub = vectors[:, j * self.dsub : (j + 1) * self.dsub]
            out[:, j, :] = sub @ codebooks[j].T
        return out

    def distance_tables(self, queries: np.ndarray) -> np.ndarray:
        """Precompute asymmetric distance lookup tables.

        Args:
            queries: ``(nq, d)`` unquantized query vectors.

        Returns:
            ``(nq, m, ksub)`` float32 tables where ``out[i, j, c]`` is the
            squared distance from query ``i``'s ``j``-th subvector to centroid
            ``c`` of subspace ``j``.
        """
        codebooks = self._require_fitted()
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"expected (nq, {self.dim}) queries, got shape {queries.shape}")
        assert self.dsub is not None
        nq = queries.shape[0]
        tables = np.empty((nq, self.m, self.ksub), dtype=np.float32)
        for j in range(self.m):
            sub = queries[:, j * self.dsub : (j + 1) * self.dsub]
            tables[:, j, :] = squared_distances(sub, codebooks[j])
        return tables

    # Measured gather crossover on Apple M4 / numpy 2.4.6, m=16, single-threaded.
    # Microseconds per scan of n candidates, best of 7 trials:
    #
    #        n     loop   fancy          n     loop   fancy
    #       32    10.09    2.88        512    19.25   30.18
    #       64    11.16    4.34       1024    27.59   61.08
    #      128    12.45    7.53       2048    44.92  102.43
    #      256    14.82   14.04      20000   359.79  945.25
    #
    # Short lists are dominated by per-call numpy overhead, so the single
    # fancy-index gather (one call) wins. Long lists are dominated by memory
    # traffic, so the per-subspace loop wins because it never materializes the
    # (n, m) gathered temporary. The crossover sat between 256 and 512 in every
    # repetition, so the threshold is placed at 512. IVF lists are usually far
    # below it, which is why the short-list path is the one that matters here.
    ADC_LOOP_THRESHOLD = 512

    def adc(
        self, codes: np.ndarray, table: np.ndarray, *, method: str = "auto"
    ) -> np.ndarray:
        """Score PQ codes against one query's lookup table.

        This is the hot loop of the whole index. Two implementations of the same
        arithmetic are kept because their costs cross over with list length; see
        ``ADC_LOOP_THRESHOLD`` above for the measurements.

        ``"loop"`` accumulates one subspace at a time. Each step is a 1-D gather
        against a 256-entry float32 table, which is 1 KB and stays in L1 for the
        whole pass, and the running sum is updated in place so no ``(n, m)``
        temporary is ever allocated.

        ``"fancy"`` does the whole gather in a single ``table[arange(m), codes]``
        and then one row sum. It allocates that temporary, but it is one numpy
        call instead of ``m`` of them, which is what matters on a short list.

        Args:
            codes: ``(n, m)`` uint8 codes.
            table: ``(m, ksub)`` table for a single query, from
                :meth:`distance_tables`.
            method: ``"auto"``, ``"loop"``, or ``"fancy"``. The two explicit
                modes exist so tests can assert they agree.

        Returns:
            ``(n,)`` float32 approximate squared distances.
        """
        self._require_fitted()
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.m:
            raise ValueError(f"expected (n, {self.m}) codes, got shape {codes.shape}")
        if table.shape != (self.m, self.ksub):
            raise ValueError(f"expected ({self.m}, {self.ksub}) table, got {table.shape}")
        if method not in ("auto", "loop", "fancy"):
            raise ValueError(f"method must be auto, loop or fancy, got {method!r}")
        n = codes.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)

        table = np.ascontiguousarray(table, dtype=np.float32)
        if method == "auto":
            method = "loop" if n >= self.ADC_LOOP_THRESHOLD else "fancy"

        if method == "fancy":
            return table[self._subspace_index, codes].sum(axis=1, dtype=np.float32)
        out = table[0][codes[:, 0]].astype(np.float32, copy=True)
        for j in range(1, self.m):
            out += table[j][codes[:, j]]
        return out
