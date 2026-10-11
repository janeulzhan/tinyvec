"""IVF-PQ: an inverted file index over product-quantized residuals.

Two independent approximations compose here, and they trade off against
different knobs.

**The inverted file** partitions the base set into ``nlist`` Voronoi cells
around k-means centroids. A query is compared against the ``nlist`` centroids,
and only the ``nprobe`` nearest cells are scanned. That cuts the candidate set to
roughly ``nprobe / nlist`` of the database. The error it introduces is a *missed*
neighbour: if the true nearest vector lives in an unprobed cell, no amount of
scoring accuracy recovers it. ``nprobe`` is the dial, and sweeping it is what
produces a recall-versus-throughput frontier rather than a single point.

**Product quantization** compresses whatever does get scanned. The error it
introduces is a *misranked* neighbour: the candidate is in hand but its distance
is only estimated, so it may be ordered below a worse vector. ``m`` is the dial.

The detail that makes this a real IVF-PQ rather than two techniques stapled
together: PQ encodes the **residual** ``x - centroid(x)``, not ``x`` itself.
Residuals are much smaller in magnitude and far more isotropic than raw vectors,
so a fixed codebook budget buys substantially lower quantization error. The cost
is that the query's distance table is no longer query-global: the table depends
on ``q - centroid_l`` and must be recomputed for each probed list. That is
``nprobe`` table builds per query instead of one, which is why ``nlist`` cannot
be made arbitrarily large for free.

Encoding residuals costs a per-list table rebuild, and naively that rebuild is
``m`` small matrix products per probed list, which at ``nprobe`` lists per query
dominates everything else. It is avoidable. Expand the distance to a stored
vector ``c_l + r`` where ``r`` is the decoded residual:

    d(q, c_l + r) = ||q - c_l||^2 + sum_j ( ||C_j[code_j]||^2
                                            + 2 c_l,j . C_j[code_j]
                                            - 2 q_j . C_j[code_j] )

The first term is already known: it is the coarse distance computed when
choosing which lists to probe. Of the three terms in the sum, the first two
depend only on the codebooks and the coarse centroids, so they can be computed
once at build time into an ``(nlist, m, 256)`` table. The third depends only on
the query, so it is one ``(m, 256)`` build per query rather than per list. The
per-list table is then a single array addition instead of ``m`` matrix products.
That is what ``precompute_tables=True`` does, and it costs
``nlist * m * 256 * 4`` bytes: 2.1 MB at nlist=128, m=16. It changes no
arithmetic, so recall is identical either way, which the tests assert directly.

An optional reranking stage closes most of the PQ ranking gap. ADC picks the top
``rerank`` candidates, then those few are rescored with exact float32 distances
against the original vectors. Because ``rerank`` is tiny next to the scanned set,
this is cheap, and it means the PQ codes only have to get the candidate *set*
right rather than the final order. It does require keeping the original vectors
around, so it trades PQ's memory win back; ``store_vectors=False`` gives up
reranking and keeps the compression.

All distances are squared Euclidean. Squaring is monotone on non-negative
values, so the ordering is identical to Euclidean and the square root is wasted
work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tinyvec.exact import brute_force
from tinyvec.kmeans import assign, kmeans, squared_distances
from tinyvec.pq import ProductQuantizer

__all__ = ["IVFPQIndex", "SearchStats"]


@dataclass
class SearchStats:
    """Deterministic counters from the most recent search.

    Wall-clock numbers are deliberately absent. These counts are a function of
    the index and the query alone, so CI can assert on them; timings cannot be
    asserted on without producing a flaky test.

    Attributes:
        n_queries: Queries in the batch.
        lists_probed: Total inverted lists scanned across the batch.
        candidates_scanned: Total PQ codes scored by ADC across the batch.
        candidates_reranked: Total candidates rescored exactly.
        tables_built: Total ``(m, 256)`` distance tables constructed.
    """

    n_queries: int = 0
    lists_probed: int = 0
    candidates_scanned: int = 0
    candidates_reranked: int = 0
    tables_built: int = 0

    @property
    def candidates_per_query(self) -> float:
        return self.candidates_scanned / self.n_queries if self.n_queries else 0.0


@dataclass
class _InvertedList:
    """One Voronoi cell's contents."""

    ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    codes: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))


class IVFPQIndex:
    """An IVF index whose postings are product-quantized residuals.

    Args:
        nlist: Number of inverted lists, i.e. coarse k-means centroids.
        m: PQ subspaces. Must divide the data dimension.
        nbits: Bits per PQ code. Only 8 is supported.
        nprobe: Default lists to scan per query. Overridable per search call.
        rerank: Default number of ADC candidates to rescore exactly. 0 disables.
            Requires ``store_vectors=True``.
        store_vectors: Keep a float32 copy of the base vectors to enable exact
            reranking. Set False to keep only the PQ codes.
        precompute_tables: Trade ``nlist * m * 256 * 4`` bytes for a per-list
            table build that is one array addition instead of ``m`` matrix
            products. Mathematically identical results either way.
        kmeans_iter: Lloyd iterations for both the coarse and the PQ codebooks.
        seed: Fixes the entire build. Two fits with the same seed and the same
            input produce byte-identical codes and centroids.

    Raises:
        ValueError: On out-of-range parameters.
    """

    def __init__(
        self,
        *,
        nlist: int = 64,
        m: int = 16,
        nbits: int = 8,
        nprobe: int = 8,
        rerank: int = 0,
        store_vectors: bool = True,
        precompute_tables: bool = True,
        kmeans_iter: int = 25,
        seed: int = 0,
    ) -> None:
        if nlist < 1:
            raise ValueError(f"nlist must be at least 1, got {nlist}")
        if nprobe < 1:
            raise ValueError(f"nprobe must be at least 1, got {nprobe}")
        if rerank < 0:
            raise ValueError(f"rerank must be non-negative, got {rerank}")
        if rerank > 0 and not store_vectors:
            raise ValueError("rerank requires store_vectors=True to rescore exactly")

        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.rerank = int(rerank)
        self.store_vectors = bool(store_vectors)
        self.precompute_tables = bool(precompute_tables)
        self.kmeans_iter = int(kmeans_iter)
        self.seed = int(seed)
        self.pq = ProductQuantizer(m, nbits=nbits, n_iter=kmeans_iter, seed=seed + 1_000)

        self.centroids: np.ndarray | None = None
        self.lists: list[_InvertedList] = []
        self.vectors: np.ndarray | None = None
        self._precomputed: np.ndarray | None = None
        self.dim: int | None = None
        self.ntotal: int = 0
        self.last_stats = SearchStats()

    # ------------------------------------------------------------------ build

    @property
    def is_fitted(self) -> bool:
        return self.centroids is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("index is not built; call fit() before searching")

    def fit(self, base: np.ndarray, train: np.ndarray | None = None) -> "IVFPQIndex":
        """Build the index over ``base``.

        Args:
            base: ``(n, d)`` vectors to index.
            train: ``(nt, d)`` vectors for fitting the coarse centroids and the
                PQ codebooks. Defaults to ``base``. Passing a held-out learn set
                avoids fitting the quantizers on the vectors they must encode.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``base`` is empty, not 2-D, or ``train`` has a
                different dimension.
        """
        base = np.ascontiguousarray(base, dtype=np.float32)
        if base.ndim != 2:
            raise ValueError(f"expected a 2-D base array, got shape {base.shape}")
        if base.shape[0] == 0:
            raise ValueError("cannot build an index over zero vectors")

        training = base if train is None else np.ascontiguousarray(train, dtype=np.float32)
        if training.ndim != 2 or training.shape[1] != base.shape[1]:
            raise ValueError(
                f"train dimension {training.shape} is incompatible with base {base.shape}"
            )

        self.dim = int(base.shape[1])
        self.ntotal = int(base.shape[0])

        coarse = kmeans(training, self.nlist, n_iter=self.kmeans_iter, seed=self.seed)
        self.centroids = coarse.centroids
        n_lists = self.centroids.shape[0]

        # Train PQ on residuals of the *training* vectors against their own
        # assigned centroid, so the codebooks see the same residual distribution
        # they will have to encode at build time.
        train_labels, _ = assign(training, self.centroids)
        train_residuals = training - self.centroids[train_labels]
        self.pq.fit(train_residuals)

        labels, _ = assign(base, self.centroids)
        residuals = base - self.centroids[labels]
        codes = self.pq.encode(residuals)

        # Group by list with a single argsort rather than nlist boolean masks,
        # which would each be a full pass over n.
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        bounds = np.searchsorted(sorted_labels, np.arange(n_lists + 1))
        self.lists = [
            _InvertedList(
                ids=np.ascontiguousarray(order[bounds[i] : bounds[i + 1]].astype(np.int32)),
                codes=np.ascontiguousarray(codes[order[bounds[i] : bounds[i + 1]]]),
            )
            for i in range(n_lists)
        ]

        if self.precompute_tables:
            # PRE[l, j, c] = ||C_j[c]||^2 + 2 * centroid_l,j . C_j[c]
            self._precomputed = np.ascontiguousarray(
                self.pq.codebook_sq_norms[None, :, :]
                + 2.0 * self.pq.subspace_dots(self.centroids)
            )
        else:
            self._precomputed = None

        self.vectors = base.copy() if self.store_vectors else None
        return self

    # ----------------------------------------------------------------- search

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        *,
        nprobe: int | None = None,
        rerank: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find approximate k nearest neighbours.

        Args:
            queries: ``(nq, d)`` or ``(d,)`` query vectors.
            k: Neighbours per query. Clamped to ``ntotal``.
            nprobe: Lists to scan. Defaults to the constructor value. Clamped to
                the number of lists.
            rerank: Candidates to rescore exactly. Defaults to the constructor
                value. Clamped up to at least ``k`` when non-zero.

        Returns:
            ``(ids, dists)`` of shape ``(nq, min(k, ntotal))``, nearest first.
            ``dists`` are squared Euclidean. They are ADC estimates unless the
            entry was reranked, in which case they are exact. Entries are padded
            with id ``-1`` and ``inf`` if a query's probed lists held fewer than
            ``k`` candidates.

        Raises:
            RuntimeError: If the index has not been built.
            ValueError: On a dimension mismatch or ``k < 1``.
        """
        self._require_fitted()
        assert self.centroids is not None

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"expected (nq, {self.dim}) queries, got shape {queries.shape}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        k = min(k, self.ntotal)
        n_lists = len(self.lists)
        probe = self.nprobe if nprobe is None else int(nprobe)
        if probe < 1:
            raise ValueError(f"nprobe must be at least 1, got {probe}")
        probe = min(probe, n_lists)
        rr = self.rerank if rerank is None else int(rerank)
        if rr < 0:
            raise ValueError(f"rerank must be non-negative, got {rr}")
        if rr > 0:
            if self.vectors is None:
                raise RuntimeError("rerank requires an index built with store_vectors=True")
            rr = max(rr, k)

        nq = queries.shape[0]
        out_ids = np.full((nq, k), -1, dtype=np.int32)
        out_dists = np.full((nq, k), np.inf, dtype=np.float32)
        stats = SearchStats(n_queries=nq)

        # One GEMM for the whole batch against the coarse centroids, then pick
        # each query's nprobe nearest cells.
        coarse_d = squared_distances(queries, self.centroids)
        if probe >= n_lists:
            probe_ids = np.argsort(coarse_d, axis=1)[:, :probe]
        else:
            part = np.argpartition(coarse_d, probe - 1, axis=1)[:, :probe]
            part_d = np.take_along_axis(coarse_d, part, axis=1)
            probe_ids = np.take_along_axis(part, np.argsort(part_d, axis=1), axis=1)

        # The only query-dependent term of the precomputed decomposition, built
        # once per query for the whole batch rather than once per probed list.
        query_terms = None
        if self._precomputed is not None:
            query_terms = -2.0 * self.pq.subspace_dots(queries)

        for qi in range(nq):
            query = queries[qi]
            cand_ids: list[np.ndarray] = []
            cand_dists: list[np.ndarray] = []

            for list_id in probe_ids[qi]:
                entry = self.lists[list_id]
                if entry.ids.size == 0:
                    continue
                if query_terms is not None:
                    # One array addition, plus the already-known coarse term.
                    table = self._precomputed[list_id] + query_terms[qi]
                    offset = coarse_d[qi, list_id]
                else:
                    # Reference path: build the residual query's table directly.
                    # Same arithmetic, m matrix products instead of one add.
                    table = self.pq.distance_tables(
                        (query - self.centroids[list_id])[None, :]
                    )[0]
                    offset = 0.0
                stats.tables_built += 1
                scored = self.pq.adc(entry.codes, table)
                if offset:
                    scored = scored + offset
                cand_dists.append(scored)
                cand_ids.append(entry.ids)
                stats.candidates_scanned += int(entry.ids.size)
            stats.lists_probed += int(probe_ids[qi].size)

            if not cand_ids:
                continue

            ids = np.concatenate(cand_ids)
            dists = np.concatenate(cand_dists)

            if rr > 0 and self.vectors is not None:
                take = min(rr, ids.size)
                keep = np.argpartition(dists, take - 1)[:take] if take < ids.size else np.arange(ids.size)
                ids = ids[keep]
                exact = squared_distances(query[None, :], self.vectors[ids])[0]
                stats.candidates_reranked += int(ids.size)
                dists = exact

            top = min(k, ids.size)
            if top < ids.size:
                sel = np.argpartition(dists, top - 1)[:top]
            else:
                sel = np.arange(ids.size)
            order = np.argsort(dists[sel], kind="stable")
            sel = sel[order]
            out_ids[qi, :top] = ids[sel]
            out_dists[qi, :top] = dists[sel]

        self.last_stats = stats
        return out_ids, out_dists

    # ------------------------------------------------------------ diagnostics

    def exact_search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Exact search over the stored vectors, for ground truth.

        Raises:
            RuntimeError: If the index was built with ``store_vectors=False``.
        """
        self._require_fitted()
        if self.vectors is None:
            raise RuntimeError("exact_search requires an index built with store_vectors=True")
        return brute_force(self.vectors, queries, k)

    def memory_report(self) -> dict[str, float]:
        """Byte-level accounting of the index.

        Returns:
            A dict with the float32 size of the original vectors, the PQ code
            size, the codebook and centroid overhead, and the resulting
            compression ratio. ``compression_ratio`` counts codes only, which is
            the asymptotic figure; ``compression_ratio_with_overhead`` folds in
            the codebooks and coarse centroids, which is what actually sits in
            memory and is the honest number at small n.
        """
        self._require_fitted()
        assert self.centroids is not None and self.dim is not None
        original = 4.0 * self.ntotal * self.dim
        codes = float(self.ntotal * self.pq.code_size)
        ids = 4.0 * self.ntotal
        codebooks = float(self.pq.codebooks.nbytes) if self.pq.codebooks is not None else 0.0
        centroids = float(self.centroids.nbytes)
        precomputed = float(self._precomputed.nbytes) if self._precomputed is not None else 0.0
        overhead = codebooks + centroids + precomputed
        return {
            "n": float(self.ntotal),
            "dim": float(self.dim),
            "original_float32_bytes": original,
            "pq_code_bytes": codes,
            "id_bytes": ids,
            "codebook_bytes": codebooks,
            "coarse_centroid_bytes": centroids,
            "precomputed_table_bytes": precomputed,
            "index_bytes_excluding_vectors": codes + ids + overhead,
            "compression_ratio": original / codes,
            "compression_ratio_with_overhead": original / (codes + ids + overhead),
        }

    def list_sizes(self) -> np.ndarray:
        """Length of each inverted list, for checking partition balance."""
        self._require_fitted()
        return np.array([entry.ids.size for entry in self.lists], dtype=np.int32)

    def __repr__(self) -> str:
        state = f"ntotal={self.ntotal}, dim={self.dim}" if self.is_fitted else "unfitted"
        return (
            f"IVFPQIndex(nlist={self.nlist}, m={self.pq.m}, nbits={self.pq.nbits}, "
            f"nprobe={self.nprobe}, rerank={self.rerank}, "
            f"precompute={self.precompute_tables}, {state})"
        )
