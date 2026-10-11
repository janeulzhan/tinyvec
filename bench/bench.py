"""Recall-versus-throughput benchmark for the IVF-PQ index.

Sweeps nprobe and reports the precision-performance frontier rather than a
single operating point, which is the only form in which an approximate-search
number means anything: any index can be made to look good by picking the nprobe
that flatters it. Follows the ann-benchmarks convention of holding the dataset,
the metric, k, and the query set fixed across every row, varying only the dial
under study, and reporting all of it.

Timing protocol: one warmup pass over the full query set to fault in pages and
let the BLAS threadpool settle, then 10 timed passes, reporting mean and
population standard deviation of queries per second. Everything is
single-threaded at the tinyvec level; numpy's own BLAS threading is pinned to
one thread by default so that the QPS figures describe one core.

Usage:
    python3 bench/bench.py                        # table to stdout, CSV to bench/results.csv
    python3 bench/bench.py --json bench/metrics.json
    python3 bench/bench.py --runs 3 --quick       # fast smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Pin BLAS threading before numpy is imported, so the single-threaded claim in
# the README is enforced by the harness rather than hoped for.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyvec.datasets import load_siftsmall  # noqa: E402
from tinyvec.exact import brute_force, recall_at_k, tie_margin  # noqa: E402
from tinyvec.index import IVFPQIndex  # noqa: E402
from tinyvec.kmeans import assign  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "bench" / "results.csv"

NPROBE_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
SEED = 42
NLIST = 128
M = 16
K = 10
KMEANS_ITER = 25
RERANK = 100


def _sysctl(name: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False, timeout=10
        )
        return out.stdout.strip() or "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def environment() -> dict[str, str]:
    """Everything a reader needs to decide whether these timings transfer."""
    memsize = _sysctl("hw.memsize")
    try:
        mem_gib = f"{int(memsize) / 2**30:.0f} GiB ({memsize} bytes)"
    except ValueError:
        mem_gib = memsize
    return {
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "physical_cores": _sysctl("hw.physicalcpu"),
        "logical_cores": _sysctl("hw.logicalcpu"),
        "memory": mem_gib,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas_threads_pinned_to": os.environ.get("OMP_NUM_THREADS", "unset"),
    }


@dataclass
class Row:
    """One point on the frontier."""

    config: str
    nprobe: int
    rerank: int
    recall_at_10: float
    qps_mean: float
    qps_stddev: float
    latency_ms_mean: float
    candidates_per_query: float
    lists_probed_per_query: float
    tables_built_per_query: float
    fraction_of_database_scanned: float


def time_search(
    index: IVFPQIndex,
    queries: np.ndarray,
    *,
    nprobe: int,
    rerank: int,
    runs: int,
) -> tuple[list[float], np.ndarray]:
    """Warm up once, then time ``runs`` full passes over the query set."""
    ids, _ = index.search(queries, K, nprobe=nprobe, rerank=rerank)
    elapsed = []
    for _ in range(runs):
        start = time.perf_counter()
        index.search(queries, K, nprobe=nprobe, rerank=rerank)
        elapsed.append(time.perf_counter() - start)
    return elapsed, ids


def time_callable(fn, *, runs: int) -> list[float]:
    fn()
    out = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return out


def summarize(elapsed: list[float], n_queries: int) -> tuple[float, float, float]:
    qps = [n_queries / t for t in elapsed]
    mean = statistics.fmean(qps)
    stddev = statistics.pstdev(qps) if len(qps) > 1 else 0.0
    latency = statistics.fmean(elapsed) / n_queries * 1000.0
    return mean, stddev, latency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="timed runs per configuration")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="raw CSV output path")
    parser.add_argument("--json", type=Path, default=None, help="machine-readable metrics output")
    parser.add_argument("--quick", action="store_true", help="short nprobe sweep, fewer iterations")
    args = parser.parse_args()

    nprobe_sweep = (1, 8, 64) if args.quick else NPROBE_SWEEP
    kmeans_iter = 5 if args.quick else KMEANS_ITER

    env = environment()
    print("=" * 78)
    print("tinyvec IVF-PQ benchmark")
    print("=" * 78)
    for key, value in env.items():
        print(f"  {key:24s} {value}")

    dataset = load_siftsmall()
    print(f"\n  dataset                  {dataset.describe()}")
    print(f"  source                   INRIA TEXMEX corpus, real measured SIFT descriptors")
    print(f"  k                        {K}")
    print(f"  index params             nlist={NLIST} m={M} nbits=8 seed={SEED} kmeans_iter={kmeans_iter}")
    print(f"  threading                single-threaded (BLAS pinned to 1 thread)")
    print(f"  timing protocol          1 warmup pass, then {args.runs} timed passes, mean and stddev")

    # Ground truth computed in-repo, not read from the shipped file, so the
    # recall numbers below are self-contained.
    gt_start = time.perf_counter()
    truth, truth_dists = brute_force(dataset.base, dataset.queries, K + 1)
    gt_seconds = time.perf_counter() - gt_start
    agrees = all(
        set(truth[qi, :K].tolist()) == set(dataset.groundtruth[qi, :K].tolist())
        for qi in range(dataset.queries.shape[0])
    )
    print(f"\n  in-repo ground truth     {gt_seconds * 1000:.1f} ms for {dataset.queries.shape[0]} queries")
    print(f"  agrees with published    {agrees}")
    print(f"  tie margin at k={K}       {tie_margin(truth_dists, K):.1f} squared units (>0 means unambiguous)")

    build_start = time.perf_counter()
    index = IVFPQIndex(
        nlist=NLIST, m=M, nprobe=8, seed=SEED, kmeans_iter=kmeans_iter
    ).fit(dataset.base, dataset.train)
    build_seconds = time.perf_counter() - build_start

    # Same index with the precomputed-table optimization disabled. Identical
    # results, so it isolates the cost of that optimization alone.
    reference = IVFPQIndex(
        nlist=NLIST, m=M, nprobe=8, seed=SEED, kmeans_iter=kmeans_iter,
        precompute_tables=False,
    ).fit(dataset.base, dataset.train)

    memory = index.memory_report()
    sizes = index.list_sizes()
    labels, _ = assign(dataset.base, index.centroids)
    residual_error = index.pq.reconstruction_error(dataset.base - index.centroids[labels])

    print(f"\n  index build time         {build_seconds:.2f} s")
    print(f"  inverted lists           {len(index.lists)} (min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()})")
    print(f"  original vectors         {memory['original_float32_bytes'] / 1e6:.2f} MB float32")
    print(f"  PQ codes                 {memory['pq_code_bytes'] / 1e6:.2f} MB uint8")
    print(f"  codebooks + centroids    {(memory['codebook_bytes'] + memory['coarse_centroid_bytes']) / 1e6:.2f} MB")
    print(f"  precomputed tables       {memory['precomputed_table_bytes'] / 1e6:.2f} MB")
    print(f"  compression (codes only) {memory['compression_ratio']:.1f}x")
    print(f"  compression (all index)  {memory['compression_ratio_with_overhead']:.1f}x")
    print(f"  residual recon error     {residual_error:.2f} mean squared")

    rows: list[Row] = []
    n_queries = dataset.queries.shape[0]

    plans = (
        ("adc", 0, index),
        ("adc+rerank100", RERANK, index),
        ("adc-no-precompute", 0, reference),
    )
    for config, rerank, target in plans:
        for nprobe in nprobe_sweep:
            elapsed, ids = time_search(
                target, dataset.queries, nprobe=nprobe, rerank=rerank, runs=args.runs
            )
            stats = target.last_stats
            qps_mean, qps_stddev, latency = summarize(elapsed, n_queries)
            rows.append(
                Row(
                    config=config,
                    nprobe=nprobe,
                    rerank=rerank,
                    recall_at_10=round(recall_at_k(ids, truth, K), 6),
                    qps_mean=round(qps_mean, 2),
                    qps_stddev=round(qps_stddev, 2),
                    latency_ms_mean=round(latency, 4),
                    candidates_per_query=round(stats.candidates_per_query, 2),
                    lists_probed_per_query=round(stats.lists_probed / n_queries, 2),
                    tables_built_per_query=round(stats.tables_built / n_queries, 2),
                    fraction_of_database_scanned=round(
                        stats.candidates_per_query / dataset.n, 6
                    ),
                )
            )

    # Exact numpy brute force at recall 1.0, the matched-recall reference point.
    bf_elapsed = time_callable(
        lambda: brute_force(dataset.base, dataset.queries, K), runs=args.runs
    )
    bf_qps, bf_sd, bf_lat = summarize(bf_elapsed, n_queries)
    rows.append(
        Row(
            config="numpy-brute-force",
            nprobe=0,
            rerank=0,
            recall_at_10=1.0,
            qps_mean=round(bf_qps, 2),
            qps_stddev=round(bf_sd, 2),
            latency_ms_mean=round(bf_lat, 4),
            candidates_per_query=float(dataset.n),
            lists_probed_per_query=0.0,
            tables_built_per_query=0.0,
            fraction_of_database_scanned=1.0,
        )
    )

    print("\n" + "-" * 78)
    print(f"{'config':18s} {'nprobe':>6s} {'recall@10':>10s} {'QPS':>10s} {'+/-':>8s} {'cand/q':>9s} {'scanned':>8s}")
    print("-" * 78)
    for row in rows:
        probe = "-" if row.nprobe == 0 else str(row.nprobe)
        print(
            f"{row.config:18s} {probe:>6s} {row.recall_at_10:10.4f} {row.qps_mean:10.1f} "
            f"{row.qps_stddev:8.1f} {row.candidates_per_query:9.1f} "
            f"{row.fraction_of_database_scanned * 100:7.1f}%"
        )
    print("-" * 78)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# tinyvec IVF-PQ benchmark, raw results"])
        for key, value in env.items():
            writer.writerow([f"# {key}", value])
        writer.writerow(["# dataset", dataset.describe()])
        writer.writerow(["# dataset_source", "INRIA TEXMEX SIFT-small, real measured descriptors"])
        writer.writerow(["# k", K])
        writer.writerow(["# nlist", NLIST, "m", M, "nbits", 8, "seed", SEED, "kmeans_iter", kmeans_iter])
        writer.writerow(["# timed_runs", args.runs, "warmup_runs", 1])
        writer.writerow(["# build_seconds", round(build_seconds, 4)])
        writer.writerow(["# compression_ratio_codes_only", round(memory["compression_ratio"], 4)])
        writer.writerow(
            ["# compression_ratio_with_overhead", round(memory["compression_ratio_with_overhead"], 4)]
        )
        writer.writerow(["# residual_reconstruction_error", round(residual_error, 4)])
        writer.writerow(["# single_threaded", True])
        writer.writerow(list(asdict(rows[0]).keys()))
        for row in rows:
            writer.writerow(list(asdict(row).values()))
    print(f"\nraw results written to {args.csv}")

    if args.json is not None:
        metrics: dict[str, object] = {
            "recall": {f"nprobe{r.nprobe}": r.recall_at_10 for r in rows if r.config == "adc"},
            "recall_rerank": {
                f"nprobe{r.nprobe}": r.recall_at_10 for r in rows if r.config == "adc+rerank100"
            },
            "candidates": {
                f"nprobe{r.nprobe}": r.candidates_per_query for r in rows if r.config == "adc"
            },
            "memory": {
                "compression_ratio": round(memory["compression_ratio"], 4),
                "compression_ratio_with_overhead": round(
                    memory["compression_ratio_with_overhead"], 4
                ),
                "original_mb": round(memory["original_float32_bytes"] / 1e6, 4),
                "codes_mb": round(memory["pq_code_bytes"] / 1e6, 4),
            },
            "quality": {"residual_reconstruction_error": round(residual_error, 2)},
            "dataset": {"n": dataset.n, "dim": dataset.dim, "queries": n_queries, "k": K},
            "timing_not_gated": {
                "build_seconds": round(build_seconds, 3),
                "qps": {f"{r.config}.nprobe{r.nprobe}": r.qps_mean for r in rows},
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"metrics written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
