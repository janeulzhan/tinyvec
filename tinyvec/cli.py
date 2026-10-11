"""Command line interface: build, query, bench, info.

An index built here is saved as a single ``.npz`` so that ``build`` and ``query``
can be separate invocations, which is the only way to tell build cost from query
cost by hand.

Examples:
    tinyvec info
    tinyvec build --out /tmp/sift.tinyvec --nlist 128 --m 16
    tinyvec query --index /tmp/sift.tinyvec --nprobe 8 --rerank 100 -k 10
    tinyvec bench --nprobe 1,4,16,64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tinyvec.datasets import gaussian_mixture, load_siftsmall
from tinyvec.exact import brute_force, recall_at_k
from tinyvec.index import IVFPQIndex


def _load_dataset(name: str):
    if name == "siftsmall":
        return load_siftsmall()
    if name == "synthetic":
        return gaussian_mixture(n=10_000, dim=128, n_queries=100, n_train=10_000, seed=0)
    raise ValueError(f"unknown dataset {name!r}")


def save_index(index: IVFPQIndex, path: Path) -> None:
    """Serialize an index to a single compressed ``.npz``.

    Inverted lists are stored flat with a boundary array rather than as one
    array per list, which keeps the archive to a fixed number of members no
    matter how large ``nlist`` is.
    """
    sizes = np.array([entry.ids.size for entry in index.lists], dtype=np.int64)
    payload = {
        "centroids": index.centroids,
        "codebooks": index.pq.codebooks,
        "list_sizes": sizes,
        "ids": np.concatenate([entry.ids for entry in index.lists])
        if index.lists
        else np.empty(0, np.int32),
        "codes": np.concatenate([entry.codes for entry in index.lists])
        if index.lists
        else np.empty((0, index.pq.m), np.uint8),
        "params": np.array(
            [index.nlist, index.pq.m, index.pq.nbits, index.nprobe, index.rerank, index.seed],
            dtype=np.int64,
        ),
        "flags": np.array(
            [int(index.store_vectors), int(index.precompute_tables)], dtype=np.int64
        ),
    }
    if index.vectors is not None:
        payload["vectors"] = index.vectors
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pass an open handle rather than the path: np.savez_compressed appends
    # ".npz" to a path that lacks it, which would silently write somewhere other
    # than where the caller asked.
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def load_index(path: Path) -> IVFPQIndex:
    """Rebuild an index from a ``.npz`` written by :func:`save_index`."""
    from tinyvec.index import _InvertedList

    with np.load(path) as handle:
        nlist, m, nbits, nprobe, rerank, seed = (int(v) for v in handle["params"])
        store_vectors, precompute = (bool(v) for v in handle["flags"])
        index = IVFPQIndex(
            nlist=nlist,
            m=m,
            nbits=nbits,
            nprobe=nprobe,
            rerank=rerank,
            store_vectors=store_vectors,
            precompute_tables=precompute,
            seed=seed,
        )
        index.centroids = handle["centroids"]
        index.pq.codebooks = handle["codebooks"]
        index.pq.dim = int(index.centroids.shape[1])
        index.pq.dsub = index.pq.dim // m
        index.dim = index.pq.dim

        sizes = handle["list_sizes"]
        ids = handle["ids"]
        codes = handle["codes"]
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        index.lists = [
            _InvertedList(
                ids=np.ascontiguousarray(ids[bounds[i] : bounds[i + 1]]),
                codes=np.ascontiguousarray(codes[bounds[i] : bounds[i + 1]]),
            )
            for i in range(len(sizes))
        ]
        index.ntotal = int(sizes.sum())
        index.vectors = handle["vectors"] if "vectors" in handle.files else None

    if precompute:
        index._precomputed = np.ascontiguousarray(
            index.pq.codebook_sq_norms[None, :, :] + 2.0 * index.pq.subspace_dots(index.centroids)
        )
    return index


def cmd_info(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    print(dataset.describe())
    print(f"  synthetic: {dataset.synthetic}")
    print(f"  base    {dataset.base.shape} {dataset.base.dtype}")
    print(f"  queries {dataset.queries.shape}")
    print(f"  train   {dataset.train.shape}")
    if dataset.groundtruth is not None:
        print(f"  shipped ground truth {dataset.groundtruth.shape}")
    print(f"  base size {dataset.base.nbytes / 1e6:.2f} MB as float32")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    print(f"building over {dataset.describe()}")
    start = time.perf_counter()
    index = IVFPQIndex(
        nlist=args.nlist,
        m=args.m,
        nprobe=args.nprobe,
        rerank=args.rerank,
        store_vectors=not args.no_store_vectors,
        precompute_tables=not args.no_precompute,
        kmeans_iter=args.kmeans_iter,
        seed=args.seed,
    ).fit(dataset.base, dataset.train)
    elapsed = time.perf_counter() - start

    report = index.memory_report()
    sizes = index.list_sizes()
    print(f"built in {elapsed:.2f} s: {index!r}")
    print(f"  lists {len(index.lists)} (min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()})")
    print(f"  originals  {report['original_float32_bytes'] / 1e6:8.2f} MB float32")
    print(f"  pq codes   {report['pq_code_bytes'] / 1e6:8.2f} MB uint8")
    print(f"  codebooks  {report['codebook_bytes'] / 1e6:8.2f} MB")
    print(f"  centroids  {report['coarse_centroid_bytes'] / 1e6:8.2f} MB")
    print(f"  precompute {report['precomputed_table_bytes'] / 1e6:8.2f} MB")
    print(f"  compression {report['compression_ratio']:.1f}x codes only, "
          f"{report['compression_ratio_with_overhead']:.1f}x whole index")

    if args.out is not None:
        save_index(index, args.out)
        print(f"  saved to {args.out} ({args.out.stat().st_size / 1e6:.2f} MB on disk)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    if args.index is not None:
        index = load_index(args.index)
        print(f"loaded {index!r}")
    else:
        print("no --index given, building a fresh one")
        index = IVFPQIndex(
            nlist=args.nlist, m=args.m, seed=args.seed, kmeans_iter=args.kmeans_iter
        ).fit(dataset.base, dataset.train)

    queries = dataset.queries[: args.n_queries]
    start = time.perf_counter()
    ids, dists = index.search(queries, args.k, nprobe=args.nprobe, rerank=args.rerank)
    elapsed = time.perf_counter() - start

    stats = index.last_stats
    print(f"\n{queries.shape[0]} queries, k={args.k}, nprobe={args.nprobe}, rerank={args.rerank}")
    print(f"  {elapsed * 1000:.2f} ms total, {elapsed / queries.shape[0] * 1000:.3f} ms per query")
    print(f"  {stats.candidates_per_query:.1f} candidates per query "
          f"({stats.candidates_per_query / index.ntotal * 100:.1f}% of the database)")

    if args.recall:
        truth, _ = brute_force(dataset.base, queries, args.k)
        print(f"  recall@{args.k} {recall_at_k(ids, truth, args.k):.4f} "
              f"(against ground truth computed in-repo)")

    for qi in range(min(args.show, queries.shape[0])):
        pairs = ", ".join(
            f"{int(i)}:{float(d):.1f}" for i, d in zip(ids[qi][:5], dists[qi][:5]) if i >= 0
        )
        print(f"  query {qi}: {pairs}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args.dataset)
    nprobes = [int(v) for v in args.nprobe.split(",")]
    index = IVFPQIndex(
        nlist=args.nlist, m=args.m, seed=args.seed, kmeans_iter=args.kmeans_iter
    ).fit(dataset.base, dataset.train)
    truth, _ = brute_force(dataset.base, dataset.queries, args.k)

    print(f"{'nprobe':>7s} {'rerank':>7s} {f'recall@{args.k}':>11s} {'QPS':>10s} {'cand/q':>9s}")
    for nprobe in nprobes:
        for rerank in (0, args.rerank) if args.rerank else (0,):
            ids, _ = index.search(dataset.queries, args.k, nprobe=nprobe, rerank=rerank)
            timings = []
            for _ in range(args.runs):
                started = time.perf_counter()
                index.search(dataset.queries, args.k, nprobe=nprobe, rerank=rerank)
                timings.append(time.perf_counter() - started)
            qps = dataset.queries.shape[0] / min(timings)
            print(f"{nprobe:7d} {rerank:7d} {recall_at_k(ids, truth, args.k):11.4f} "
                  f"{qps:10.1f} {index.last_stats.candidates_per_query:9.1f}")
    print("\nTimings here are a quick single-process estimate. The disclosed "
          "benchmark is bench/bench.py.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyvec",
        description="IVF plus product-quantization vector index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dataset", default="siftsmall", choices=("siftsmall", "synthetic"))
        sub.add_argument("--nlist", type=int, default=128)
        sub.add_argument("--m", type=int, default=16)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--kmeans-iter", type=int, default=25)

    info = subparsers.add_parser("info", help="describe a dataset")
    info.add_argument("--dataset", default="siftsmall", choices=("siftsmall", "synthetic"))
    info.set_defaults(func=cmd_info)

    build = subparsers.add_parser("build", help="build an index and optionally save it")
    add_common(build)
    build.add_argument("--out", type=Path, default=None, help="save the index here")
    build.add_argument("--nprobe", type=int, default=8)
    build.add_argument("--rerank", type=int, default=0)
    build.add_argument("--no-store-vectors", action="store_true",
                       help="drop the float32 originals, realizing the full compression")
    build.add_argument("--no-precompute", action="store_true",
                       help="disable precomputed per-list tables")
    build.set_defaults(func=cmd_build)

    query = subparsers.add_parser("query", help="search an index")
    add_common(query)
    query.add_argument("--index", type=Path, default=None, help="load an index saved by build")
    query.add_argument("-k", type=int, default=10)
    query.add_argument("--nprobe", type=int, default=8)
    query.add_argument("--rerank", type=int, default=0)
    query.add_argument("--n-queries", type=int, default=100)
    query.add_argument("--show", type=int, default=3, help="print this many result rows")
    query.add_argument("--recall", action="store_true", help="also compute recall against exact search")
    query.set_defaults(func=cmd_query)

    bench = subparsers.add_parser("bench", help="quick recall and QPS sweep")
    add_common(bench)
    bench.add_argument("--nprobe", default="1,2,4,8,16,32", help="comma-separated sweep")
    bench.add_argument("-k", type=int, default=10)
    bench.add_argument("--rerank", type=int, default=100)
    bench.add_argument("--runs", type=int, default=5)
    bench.set_defaults(func=cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from tinyvec import __version__

        print(f"tinyvec {__version__}")
        return 0
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"tinyvec: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
