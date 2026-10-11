"""tinyvec: an IVF plus product-quantization vector index built on numpy."""

from tinyvec.datasets import Dataset, gaussian_mixture, load_siftsmall
from tinyvec.exact import brute_force, recall_at_k, tie_margin
from tinyvec.index import IVFPQIndex, SearchStats
from tinyvec.kmeans import KMeansResult, assign, kmeans, squared_distances
from tinyvec.pq import ProductQuantizer

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "IVFPQIndex",
    "KMeansResult",
    "ProductQuantizer",
    "SearchStats",
    "assign",
    "brute_force",
    "gaussian_mixture",
    "kmeans",
    "load_siftsmall",
    "recall_at_k",
    "squared_distances",
    "tie_margin",
    "__version__",
]
