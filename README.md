# tinyvec

[![ci](https://github.com/janeulzhan/tinyvec/actions/workflows/ci.yml/badge.svg)](https://github.com/janeulzhan/tinyvec/actions/workflows/ci.yml)

An IVF plus product-quantization vector index, written from scratch in Python with numpy.

Approximate nearest-neighbour search where the interesting part is not the index
structure but the arithmetic: product quantization stores each vector as
<!--bench:dataset.dim-->128<!--/bench-->-dimensional data compressed into 16
bytes, and **asymmetric distance computation** then scores candidates against an
unquantized query using nothing but table lookups. No arithmetic touches the
original dimensions at query time.

Everything here is measured on real data. The benchmark runs on SIFT-small from
the INRIA TEXMEX corpus (<!--bench:dataset.n-->10000<!--/bench--> real SIFT
descriptors, <!--bench:dataset.queries-->100<!--/bench--> queries, published
exact ground truth), the dataset is vendored so a fresh clone works offline, and
the ground truth is recomputed in-repo rather than trusted. Every number below
came out of `bench/bench.py`, and the deterministic ones are locked to the
committed metrics by `tools/benchlock.py` so the README cannot quietly drift
away from the code.

## The frontier

A single recall number says nothing, because any approximate index can be tuned
to produce a flattering one. What matters is the whole precision-performance
curve, so `nprobe` is swept across eight values with everything else held fixed.

Two error sources compose independently. The inverted file can **miss** a
neighbour by never looking in its cell, which `nprobe` controls. Product
quantization can **misrank** a neighbour it did find, which reranking fixes.

| `nprobe` | recall@10, ADC | recall@10, ADC + rerank 100 | candidates scanned per query | share of database |
| --- | --- | --- | --- | --- |
| 1 | <!--bench:recall.nprobe1-->0.4030<!--/bench--> | <!--bench:recall_rerank.nprobe1-->0.4440<!--/bench--> | <!--bench:candidates.nprobe1-->92.1<!--/bench--> | 0.9% |
| 2 | <!--bench:recall.nprobe2-->0.5480<!--/bench--> | <!--bench:recall_rerank.nprobe2-->0.6420<!--/bench--> | <!--bench:candidates.nprobe2-->176.9<!--/bench--> | 1.8% |
| 4 | <!--bench:recall.nprobe4-->0.6480<!--/bench--> | <!--bench:recall_rerank.nprobe4-->0.8080<!--/bench--> | <!--bench:candidates.nprobe4-->343.3<!--/bench--> | 3.4% |
| 8 | <!--bench:recall.nprobe8-->0.7000<!--/bench--> | <!--bench:recall_rerank.nprobe8-->0.9250<!--/bench--> | <!--bench:candidates.nprobe8-->665.6<!--/bench--> | 6.7% |
| 16 | <!--bench:recall.nprobe16-->0.7230<!--/bench--> | <!--bench:recall_rerank.nprobe16-->0.9880<!--/bench--> | <!--bench:candidates.nprobe16-->1304.5<!--/bench--> | 13.0% |
| 32 | <!--bench:recall.nprobe32-->0.7240<!--/bench--> | <!--bench:recall_rerank.nprobe32-->0.9990<!--/bench--> | <!--bench:candidates.nprobe32-->2542.2<!--/bench--> | 25.4% |
| 64 | <!--bench:recall.nprobe64-->0.7240<!--/bench--> | <!--bench:recall_rerank.nprobe64-->1.0000<!--/bench--> | <!--bench:candidates.nprobe64-->4843.6<!--/bench--> | 48.4% |
| 128 | <!--bench:recall.nprobe128-->0.7240<!--/bench--> | <!--bench:recall_rerank.nprobe128-->1.0000<!--/bench--> | <!--bench:candidates.nprobe128-->10000.0<!--/bench--> | 100% |

`nlist=128`, `m=16`, `nbits=8`, `seed=42`, `k=10`, squared Euclidean. Candidates
scanned per query is the hardware-independent cost measure, which is why it, and
not throughput, is the column next to recall. Throughput is reported separately
below because it depends on the machine.

The shape of that table is the whole point. **The ADC column saturates at
<!--bench:recall.nprobe128-->0.7240<!--/bench-->.** At `nprobe=128` the index
scans all <!--bench:candidates.nprobe128-->10000.0<!--/bench--> vectors for
every query, so the inverted file has discarded nothing and cannot be
responsible for any loss. The missing 27.6 points of recall are entirely
product-quantization misranking. Reranking the top 100 candidates with exact
distances recovers all of it, reaching
<!--bench:recall_rerank.nprobe64-->1.0000<!--/bench-->, which says the PQ codes
were already selecting the right candidate set and only ordering it wrong. That
is the argument for a rerank stage, and it is a measurement rather than an
assertion.

## Throughput

Separate table, because these are wall-clock numbers. They describe one specific
machine, they are **not** checked by CI, and they should not be read as
portable.

Measured on Apple M4, 16 GiB, macOS 15.5, Python 3.11.9, numpy 2.4.6, pinned to
a single BLAS thread. One warmup pass, then 10 timed passes over all 100
queries; mean and population standard deviation of queries per second.

| `nprobe` | QPS, ADC | QPS, ADC + rerank 100 | QPS without precomputed tables | precompute speedup |
| --- | --- | --- | --- | --- |
| 1 | 59137 ± 13132 | 36140 ± 1055 | 10173 ± 343 | 5.8x |
| 2 | 43029 ± 1712 | 23745 ± 729 | 5255 ± 90 | 8.2x |
| 4 | 25728 ± 566 | 17331 ± 788 | 2565 ± 60 | 10.0x |
| 8 | 14422 ± 196 | 10917 ± 165 | 1331 ± 37 | 10.8x |
| 16 | 7488 ± 149 | 6443 ± 72 | 695 ± 6 | 10.8x |
| 32 | 3810 ± 92 | 3474 ± 39 | 340 ± 14 | 11.2x |
| 64 | 2009 ± 56 | 1884 ± 38 | 167 ± 5 | 12.0x |
| 128 | 1009 ± 20 | 864 ± 77 | 81 ± 5 | 12.5x |

Index build time was 7.97 s. For reference, exact numpy brute force over the
same data reaches 12499 ± 594 QPS at recall 1.0.

Most rows hold their standard deviation under 5 percent of the mean. The
`nprobe=1` ADC row is the exception at 22 percent, and the reason is structural
rather than mysterious: one pass over 100 queries takes about 1.7 ms there, which
is short enough that operating-system jitter is a visible fraction of the
measurement. The slower configurations are correspondingly steadier.

### Reading that reference row honestly

At this scale, exact brute force is hard to beat and tinyvec does not always
beat it. One call to `squared_distances` over 10,000 vectors is a single large
BLAS matrix product, which is about as efficient as float32 arithmetic gets. The
honest summary is:

- Below roughly 0.81 recall, tinyvec is faster. At `nprobe=4` with reranking it
  delivers <!--bench:recall_rerank.nprobe4-->0.8080<!--/bench--> recall at 1.39x
  brute-force throughput, at `nprobe=2` it is 1.90x faster at
  <!--bench:recall_rerank.nprobe2-->0.6420<!--/bench-->, and at `nprobe=1` it is
  4.7x faster at <!--bench:recall.nprobe1-->0.4030<!--/bench--> recall.
- Above that, brute force wins on wall-clock. Reaching
  <!--bench:recall_rerank.nprobe8-->0.9250<!--/bench--> runs at 0.87x brute
  force, and <!--bench:recall_rerank.nprobe16-->0.9880<!--/bench--> at 0.52x.

Those ratios are stable even though the absolute throughput is not. Measured
three more times back to back in a single quiet process, the ratio at
`nprobe=4` with reranking stayed within 1.27x to 1.30x and the ratio at
`nprobe=8` within 0.81x to 0.83x, so the crossover sits between 0.81 and 0.93
recall rather than being an artifact of one noisy run.

That crossover is a property of n = 10,000, not of IVF-PQ. The per-query Python
overhead is a fixed cost paid `nprobe` times, while brute force amortizes a
single GEMM across the whole database, so the comparison moves steadily in
tinyvec's favour as n grows. Demonstrating that would need a dataset far larger
than one that belongs in a git repository, so this README does not claim it.
What it does claim is the memory result, which holds at every scale.

## Memory

Compression is the thing PQ buys that a graph index does not, and the ratio is
exactly `4 * dim / m`, independent of the data.

| component | bytes | note |
| --- | --- | --- |
| original vectors, float32 | 5,120,000 | <!--bench:memory.original_mb-->5.12<!--/bench--> MB |
| PQ codes, uint8 | 160,000 | <!--bench:memory.codes_mb-->0.16<!--/bench--> MB |
| PQ codebooks | 131,072 | 16 subspaces x 256 x 8 dims |
| coarse centroids | 65,536 | 128 x 128 |
| vector ids | 40,000 | int32 |
| precomputed list tables | 2,097,152 | `nlist * m * 256 * 4`, optional |

**Codes alone: <!--bench:memory.compression_ratio-->32.0<!--/bench-->x.** That
is the number that scales, since it is the only term proportional to n.

Counting the whole index including the optional precomputed tables gives
<!--bench:memory.compression_ratio_with_overhead-->2.1<!--/bench-->x, and
quoting only the first number would be misleading at this size. The precomputed
table is 2.1 MB and depends on `nlist` and `m` but **not** on n, so at 10,000
vectors it is larger than the codes it accelerates. Without it the whole index
is 12.9x smaller than the originals. Both configurations ship, both are
benchmarked, and `--no-precompute` selects the smaller one.

Mean squared residual reconstruction error at these settings is
<!--bench:quality.residual_reconstruction_error-->12071.06<!--/bench-->.

## Quickstart

No network access needed; the dataset is in the repository.

```bash
git clone https://github.com/janeulzhan/tinyvec.git
cd tinyvec
python3 -m pip install -e ".[dev]"

python3 -m pytest -m "not slow" -q      # 230 tests
tinyvec info                            # describe the vendored dataset
tinyvec bench --nprobe 1,4,16,64        # recall and QPS sweep
```

Build once, query separately:

```bash
tinyvec build --out /tmp/sift.tinyvec --nlist 128 --m 16
tinyvec query --index /tmp/sift.tinyvec --nprobe 8 --rerank 100 --recall
```

Library use:

```python
from tinyvec import IVFPQIndex, load_siftsmall, brute_force, recall_at_k

data = load_siftsmall()
index = IVFPQIndex(nlist=128, m=16, seed=42).fit(data.base, data.train)

truth, _ = brute_force(data.base, data.queries, 10)
ids, dists = index.search(data.queries, k=10, nprobe=8, rerank=100)

print(recall_at_k(ids, truth, 10))              # 0.925
print(index.memory_report()["compression_ratio"])  # 32.0
print(index.last_stats.candidates_per_query)    # 665.55
```

To reproduce the tables, which rewrites `bench/results.csv` and
`bench/metrics.json`:

```bash
python3 bench/bench.py --runs 10 --json bench/metrics.json
python3 tools/benchlock.py --update     # sync the README to what you measured
```

## How IVF-PQ works

```
                   query q (never quantized)
                           |
        +------------------+-------------------+
        |                                      |
  1. coarse search                       3. build one
     ||q - c_l||^2 for all                  (m x 256) table
     128 centroids, keep                    of q-to-codebook
     the nprobe nearest                     distances
        |                                      |
        v                                      v
  2. inverted lists                      4. for each probed list, score every
     list 0: [id,code][id,code]...          code by m table lookups and a sum.
     list 1: [id,code]...                   16 lookups per candidate, zero
     ...                                    arithmetic over the 128 dims
     list 127: ...
                                               |
                                               v
                                        5. optional: rescore the top 100
                                           exactly, then return k=10

  one stored vector = 16 uint8 codes = 16 bytes   (was 512 bytes of float32)
```

k-means partitions the base set into `nlist` Voronoi cells, and a query only
scans the `nprobe` cells whose centroids are nearest to it. Within a cell, each
vector is stored not as itself but as the PQ code of its **residual** against
that cell's centroid, which is a smaller and more isotropic quantity than the
raw vector and so quantizes to lower error for the same codebook budget. Because
squared L2 decomposes as a sum over disjoint coordinate blocks, the distance
from an unquantized query to any stored code is just a sum of `m` precomputed
table entries, which is what makes ADC strictly more accurate than quantizing
both sides: only one of the two vectors carries error. The returned distances
are squared Euclidean throughout, since squaring is monotone and the square root
would be wasted work.

### The optimization that mattered

Residual coding has a catch. The lookup table depends on `q - c_l`, so it is not
query-global and the naive implementation rebuilds it per probed list, at `m`
small matrix products each. Profiling said that rebuild, not the table gather,
was where the time went. Expanding the distance to a stored vector `c_l + r`
separates it:

```
d(q, c_l + r) = ||q - c_l||^2  +  sum_j ( ||C_j[code_j]||^2
                                          + 2 c_l,j . C_j[code_j]
                                          - 2 q_j . C_j[code_j] )
```

The first term is already known from choosing which lists to probe. Inside the
sum, the first two terms involve only the codebooks and the coarse centroids, so
they fold into one `(nlist, m, 256)` table built once at `fit()` time; only the
third depends on the query, and it is one `(m, 256)` build per query rather than
per list. The per-list table build becomes a single array addition, worth the
5.8x to 12.5x in the throughput table above.

It is pure algebra, so retrieval does not move at all. The benchmark checks this
on every run and reports it: returned ids are byte-identical at
<!--bench:quality.precompute_identical_id_nprobe_values-->8<!--/bench--> of
<!--bench:quality.precompute_nprobe_values_compared-->8<!--/bench--> swept
`nprobe` values. Distance *values* differ only by float32 rounding, since the
two routes sum the same terms in a different order; the measured worst is
<!--bench:quality.precompute_max_scaled_distance_diff-->1.325e-06<!--/bench-->
of the largest returned distance.

What `tests/test_precompute.py` asserts, precisely: that the ids are identical,
and that the scaled distance difference stays below 5e-05. That ceiling is
deliberately close to the worst value ever measured, and a companion test fails
if it ever drifts to more than 10x the measured worst, because a bound far
looser than reality is exactly what lets prose overstate a test.

## Benchmark disclosure

| | |
| --- | --- |
| CPU | Apple M4, 10 physical cores, 10 logical |
| Memory | 16 GiB (17179869184 bytes) |
| Platform | macOS-15.5-arm64-arm-64bit |
| Python / numpy | 3.11.9 / 2.4.6 |
| Threading | single-threaded; harness pins BLAS to 1 thread before importing numpy |
| Dataset | SIFT-small, INRIA TEXMEX corpus, real measured SIFT descriptors |
| Size | n = <!--bench:dataset.n-->10000<!--/bench-->, dim = <!--bench:dataset.dim-->128<!--/bench-->, <!--bench:dataset.queries-->100<!--/bench--> queries, 25,000 held-out training vectors |
| Metric | squared Euclidean |
| k | <!--bench:dataset.k-->10<!--/bench--> |
| Index params | `nlist=128`, `m=16`, `nbits=8`, `kmeans_iter=25`, `seed=42` |
| Build time | 7.97 s |
| Timing protocol | 1 warmup pass, 10 timed passes, mean and population stddev |
| Raw data | [`bench/results.csv`](bench/results.csv), harness at [`bench/bench.py`](bench/bench.py) |

On dataset provenance: SIFT-small ships as float32 `.fvecs`, but every
descriptor component is an integer in [0, 255], so the repository vendors it as
`uint8` and casts back on load. That is bit-identical to parsing the upstream
files, not a subsample, and it gets the corpus to 3.4 MB.
`scripts/download_siftsmall.py` re-derives it from the INRIA tarball and checks
its SHA-256, so the vendored copy is verifiable rather than merely asserted.

The in-repo brute force was validated against the published ground truth on two
counts: the returned neighbour sets agree at every k from 1 to 100, and the
sequence of distances is identical, exactly and not approximately. The second is
the stronger claim, and it is the one that survives ties.

Exact id ordering is not claimed, because it is not portable. Where two base
vectors sit at the same distance from a query, which sorts first depends on
float32 accumulation order and so on the BLAS build; 18 of 10,000 cells permute
on the authoring machine and a different number will on yours.
`tests/test_exact.py` asserts the property rather than the count: any id that
differs from the published file must be a vector at exactly the same distance.
An earlier version pinned the ordering to k=20, which passed locally and failed
in CI on Linux.

The tie margin at k=10 is 6.0 squared units, which is comfortably positive, so
recall@10 on this dataset is a well-defined quantity. That check runs before any
recall number is reported.

## What CI gates, and what it does not

CI asserts only on quantities that are a deterministic function of the code, the
data and the seed: recall@10 at each `nprobe`, candidate counts, byte sizes,
compression ratios, and reconstruction error. `tests/test_recall_floor.py` seals
the dataset, the parameters and the seed, then holds recall above a declared
floor per `nprobe`, so a refactor that degrades retrieval quality fails the
build. `tools/benchlock.py --check` separately verifies that every locked number
in this README still matches `bench/metrics.json`.

**CI never gates wall-clock.** Not throughput, not build time. A timing
tolerance tight enough to catch a real slowdown is also tight enough to fire on
a noisy shared runner, and a gate people learn to ignore is worse than no gate
at all. `benchlock` enforces this structurally: any key under the
`timing_not_gated` prefix is refused outright by both `--check` and `--update`.
Timing claims live in their own table above, labelled with the hardware that
produced them.

The recall floors sit about 3 points below the measured values on purpose.
Recall depends on k-means centroids, which depend on float32 GEMM accumulation
order, which differs between BLAS builds. A test asserting exact equality would
pass on the authoring machine and flake anywhere else, so the floors declare the
margin instead of pretending it is not needed.

## Limitations

Read this before taking any number above as a general claim.

**It is pure single-threaded numpy, and not competitive with FAISS on absolute
speed.** The per-candidate ADC gather is a numpy fancy-index, not a SIMD-packed
kernel; FAISS scores four or eight codes per vector instruction and tinyvec
cannot. There is no FAISS comparison in this README because none was run, and a
borrowed benchmark is not evidence.

**The throughput numbers do not extrapolate.** At n = 10,000 a fixed per-query
Python cost is paid `nprobe` times and a well-tuned GEMM beats the index above
roughly 0.81 recall, as shown above. The direction of that comparison depends on
n and would improve with scale, but this repository does not demonstrate it, so
treat the crossover as a fact about this dataset only.

**The benchmark machine was not quiesced.** A browser was consuming roughly 160%
CPU during development, and microbenchmarks came back visibly bimodal as a
result. That is the reason for the warmup pass, the 10 timed runs, and reporting
standard deviation rather than a best-of-N. The widest remaining spread is the
22 percent on the `nprobe=1` ADC row, explained above. An earlier run of the
same harness produced reranked throughput roughly 40 percent below the committed
numbers with standard deviations near 30 percent; that run was discarded as
load-polluted after the ratios it implied were contradicted by three stable
back-to-back measurements, which is the kind of judgement call a wall-clock
number forces and a deterministic one does not.

This is also why the deterministic columns are the ones CI gates. Measurement
noise has cost this repository two wrong claims so far, both caught and both
recorded here: an early docstring asserted a 2x win for a flattened table gather
when measurement showed it was slower, and a k-means test asserted an inertia
invariant that float32 cancellation makes false in the regime the test itself
constructed. The second one passed locally and failed in CI on a different BLAS.

**Recall is measured on 100 queries.** SIFT-small ships exactly 100, so a recall
of 0.9250 is 925 hits out of 1000 judgements and carries a standard error near
0.8 percentage points. Differences below about 2 points between two rows here
are not meaningful.

**Scope gaps.** No metadata filtering, no deletions, no incremental `add()`
after `fit()`, no on-disk memory-mapped format (the CLI writes a whole-index npz
and loads it eagerly), cosine and inner-product metrics are not implemented,
`nbits` is fixed at 8 by the uint8 code layout, and there is no OPQ rotation,
which typically recovers a few points of recall at these settings by
decorrelating the subspaces before quantization.

**Float32 distance arithmetic has a precision floor.** Distances come from the
`|a|^2 - 2a.b + |b|^2` expansion so the whole thing is one matrix product, and
that expansion loses relative accuracy as `|a|, |b|` grow against `|a - b|`.
Tight clusters far from the origin are the pathological case: at 5-dimensional
clusters of radius 0.01 centred 20 units out, the relative error on the small
distances reaches 0.71, which is not a usable distance. `squared_distances`
documents the measured error across that whole regime, a test pins it, and
k-means sidesteps the worst consequence by computing inertia from direct float64
residuals rather than from the expanded distances. Real embedding data, SIFT
included, sits at the well-conditioned end where the error is near 1e-07, but if
you bring your own data of that shape, centre it first.

**Reranking trades away the headline memory win.** It needs the original float32
vectors, so an index with `rerank` enabled stores them and the 32x compression
is not realized. `store_vectors=False` keeps the compression and gives up
reranking. Both are tested; the reranked recall column above assumes the
originals are present.

**The coarse quantizer is brute-forced.** Choosing lists costs a full scan of
all `nlist` centroids per query, which is fine at 128 and would need its own
index long before `nlist` reached the tens of thousands that a billion-scale
deployment would want.

## Layout

```
tinyvec/kmeans.py    Lloyd's k-means, k-means++ seeding, monotone inertia
tinyvec/pq.py        codebooks, uint8 encoding, ADC and its lookup tables
tinyvec/index.py     IVF-PQ: residual coding, nprobe, rerank, precomputed tables
tinyvec/exact.py     brute-force ground truth, recall@k, tie-margin guard
tinyvec/datasets.py  vendored SIFT-small loader, synthetic generator
tinyvec/cli.py       build / query / bench / info
bench/bench.py       the disclosed benchmark; bench/results.csv is its output
tools/benchlock.py   locks this README's numbers to bench/metrics.json
tests/               230 tests; test_adc.py is the one that matters most
```

`tests/test_adc.py` is worth opening first. The entire index rests on the claim
that a sum of table lookups equals a real distance, so that claim is checked
against a distance recomputed from scratch on the decoded vectors, against a
naive Python double loop over the table, and across four values of `m`. Those
assertions are set at 1e-05 relative.

The benchmark then re-measures the same identity on the real index, scoring 25
SIFT queries against all <!--bench:dataset.n-->10000<!--/bench--> codes with a
float64 reference, and reports the worst relative error:
<!--bench:quality.adc_max_relative_error-->2.95e-07<!--/bench-->. That is
float32 rounding rather than an approximation gap, and it comes out of
`bench/bench.py` on every run rather than from a figure typed in once.

## References

- Jegou, Douze and Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011
- Arthur and Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, SODA 2007
- Aumuller, Bernhardsson and Faithfull, *ANN-Benchmarks*, Information Systems 2020
- SIFT-small, INRIA TEXMEX corpus: http://corpus-texmex.irisa.fr/

## License

MIT. See [LICENSE](LICENSE).
