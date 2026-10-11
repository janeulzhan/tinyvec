"""Dataset loaders.

The benchmark in this repository runs on **SIFT-small**, a real public dataset:
10,000 128-dimensional SIFT descriptors with 100 queries and published exact
ground truth, distributed by INRIA as part of the TEXMEX corpus
(http://corpus-texmex.irisa.fr/). The metric is Euclidean.

The data is vendored in ``data/siftsmall.npz`` so that a fresh clone can run the
full benchmark and the recall regression test with no network access. Vendoring
is lossless and not a subsample: every SIFT descriptor component is an integer
in [0, 255], so the original float32 ``.fvecs`` files are stored as ``uint8`` and
cast back to float32 on load. That is bit-identical to parsing the upstream
files, and it is what gets the corpus under 4 MB instead of 18 MB.
``scripts/download_siftsmall.py`` regenerates the archive from the upstream
tarball and checks its SHA-256, so the vendored copy is reproducible rather than
merely asserted.

:func:`gaussian_mixture` is also provided, but it is **synthetic** and exists
only for tests that need a cheap dataset of an arbitrary shape. No benchmark
number in the README comes from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Dataset", "load_siftsmall", "gaussian_mixture", "SIFTSMALL_PATH"]

SIFTSMALL_PATH = Path(__file__).resolve().parent.parent / "data" / "siftsmall.npz"

# SHA-256 of siftsmall.tar.gz as published at
# ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
SIFTSMALL_TARBALL_SHA256 = "b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e"


@dataclass(frozen=True)
class Dataset:
    """A benchmark dataset.

    Attributes:
        name: Human-readable identifier.
        base: ``(n, d)`` float32 vectors to index.
        queries: ``(nq, d)`` float32 query vectors.
        train: ``(nt, d)`` float32 vectors for fitting quantizers. Held out from
            ``base`` where the source dataset provides a separate learn set.
        groundtruth: ``(nq, kt)`` int32 exact neighbour ids, or None.
        metric: Distance name.
        synthetic: True if the vectors were generated rather than measured.
    """

    name: str
    base: np.ndarray
    queries: np.ndarray
    train: np.ndarray
    groundtruth: np.ndarray | None
    metric: str = "euclidean"
    synthetic: bool = False

    @property
    def n(self) -> int:
        return int(self.base.shape[0])

    @property
    def dim(self) -> int:
        return int(self.base.shape[1])

    def describe(self) -> str:
        kind = "synthetic" if self.synthetic else "real"
        return (
            f"{self.name} ({kind}): n={self.n} dim={self.dim} "
            f"queries={self.queries.shape[0]} train={self.train.shape[0]} metric={self.metric}"
        )


def load_siftsmall(path: Path | str | None = None) -> Dataset:
    """Load the vendored SIFT-small corpus.

    Args:
        path: Override for the ``.npz`` location.

    Returns:
        A :class:`Dataset` with ``synthetic=False``. ``train`` is the 25,000-vector
        SIFT learn set, which is disjoint from ``base``, so PQ codebooks are never
        fitted on the vectors they will be asked to encode.

    Raises:
        FileNotFoundError: If the archive is missing.
    """
    target = Path(path) if path is not None else SIFTSMALL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"SIFT-small archive not found at {target}. "
            "Run scripts/download_siftsmall.py to regenerate it."
        )
    with np.load(target) as handle:
        base = handle["base"].astype(np.float32)
        queries = handle["query"].astype(np.float32)
        train = handle["learn"].astype(np.float32)
        groundtruth = handle["groundtruth"].astype(np.int32)
    return Dataset(
        name="sift-small",
        base=np.ascontiguousarray(base),
        queries=np.ascontiguousarray(queries),
        train=np.ascontiguousarray(train),
        groundtruth=groundtruth,
        metric="euclidean",
        synthetic=False,
    )


def gaussian_mixture(
    n: int = 2000,
    dim: int = 32,
    *,
    n_queries: int = 50,
    n_train: int = 2000,
    n_clusters: int = 16,
    spread: float = 8.0,
    seed: int = 0,
) -> Dataset:
    """Generate a deterministic Gaussian-mixture dataset.

    Synthetic, and flagged as such on the returned :class:`Dataset`. Clustered
    rather than uniform, because a uniform cloud in high dimensions has no
    structure for an inverted file to exploit and would make IVF look worse than
    it is on real data.

    Args:
        n: Base vectors.
        dim: Dimension.
        n_queries: Query vectors, drawn from the same mixture.
        n_train: Separate training vectors for quantizer fitting.
        n_clusters: Mixture components.
        spread: Standard deviation of the component means. Unit-variance
            components means this controls how separated the clusters are.
        seed: Fixes every draw.

    Returns:
        A :class:`Dataset` with ``synthetic=True`` and ``groundtruth=None``.
    """
    if min(n, dim, n_queries, n_train, n_clusters) < 1:
        raise ValueError("all size parameters must be positive")
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=spread, size=(n_clusters, dim)).astype(np.float32)

    def draw(count: int) -> np.ndarray:
        which = rng.integers(n_clusters, size=count)
        return (centers[which] + rng.normal(size=(count, dim))).astype(np.float32)

    return Dataset(
        name=f"gaussian-mixture(n={n},dim={dim},clusters={n_clusters},seed={seed})",
        base=np.ascontiguousarray(draw(n)),
        queries=np.ascontiguousarray(draw(n_queries)),
        train=np.ascontiguousarray(draw(n_train)),
        groundtruth=None,
        metric="euclidean",
        synthetic=True,
    )
