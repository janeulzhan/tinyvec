"""tinyvec: an IVF plus product-quantization vector index built on numpy."""

from tinyvec.kmeans import KMeansResult, assign, kmeans, squared_distances

__version__ = "0.1.0"

__all__ = [
    "KMeansResult",
    "assign",
    "kmeans",
    "squared_distances",
    "__version__",
]
