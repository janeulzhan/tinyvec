"""CLI tests, including the index save and load round trip.

The round trip matters more than it looks: ``build`` and ``query`` are separate
processes, so if serialization drops or reorders anything the recall silently
changes and nothing else catches it. The tests therefore compare a loaded
index's results against the in-memory original rather than just checking that
the file parses.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.cli import build_parser, load_index, main, save_index
from tinyvec.datasets import gaussian_mixture
from tinyvec.index import IVFPQIndex


@pytest.fixture(scope="module")
def dataset():
    return gaussian_mixture(n=800, dim=32, n_queries=20, n_train=800, n_clusters=8, seed=0)


@pytest.fixture(scope="module")
def index(dataset):
    return IVFPQIndex(nlist=8, m=8, nprobe=4, seed=0, kmeans_iter=8).fit(
        dataset.base, dataset.train
    )


def test_version(capsys):
    assert main(["--version"]) == 0
    assert "tinyvec" in capsys.readouterr().out


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_parser_exposes_the_documented_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    assert set(actions[0].choices) == {"info", "build", "query", "bench"}


def test_info_runs(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "sift-small" in out
    assert "synthetic: False" in out


def test_info_on_synthetic_says_so(capsys):
    assert main(["info", "--dataset", "synthetic"]) == 0
    assert "synthetic: True" in capsys.readouterr().out


# ----------------------------------------------------------- save / load


def test_save_writes_to_the_exact_path(tmp_path, index):
    """np.savez_compressed silently appends .npz to a path without it, so this
    pins that the file lands where the caller asked."""
    target = tmp_path / "index.tinyvec"
    save_index(index, target)
    assert target.exists()
    assert not (tmp_path / "index.tinyvec.npz").exists()


def test_round_trip_preserves_the_index(tmp_path, index):
    target = tmp_path / "idx.tinyvec"
    save_index(index, target)
    loaded = load_index(target)

    assert loaded.ntotal == index.ntotal
    assert loaded.dim == index.dim
    assert loaded.nlist == index.nlist
    assert loaded.pq.m == index.pq.m
    assert loaded.precompute_tables == index.precompute_tables
    assert np.array_equal(loaded.centroids, index.centroids)
    assert np.array_equal(loaded.pq.codebooks, index.pq.codebooks)
    assert len(loaded.lists) == len(index.lists)
    for a, b in zip(loaded.lists, index.lists):
        assert np.array_equal(a.ids, b.ids)
        assert np.array_equal(a.codes, b.codes)


def test_round_trip_preserves_search_results(tmp_path, index, dataset):
    target = tmp_path / "idx.tinyvec"
    save_index(index, target)
    loaded = load_index(target)
    for nprobe in (1, 4, 8):
        a, da = index.search(dataset.queries, 10, nprobe=nprobe)
        b, db = loaded.search(dataset.queries, 10, nprobe=nprobe)
        assert np.array_equal(a, b), nprobe
        assert np.allclose(da, db, equal_nan=True), nprobe


def test_round_trip_preserves_rerank(tmp_path, index, dataset):
    target = tmp_path / "idx.tinyvec"
    save_index(index, target)
    loaded = load_index(target)
    a, _ = index.search(dataset.queries, 10, nprobe=8, rerank=50)
    b, _ = loaded.search(dataset.queries, 10, nprobe=8, rerank=50)
    assert np.array_equal(a, b)


def test_round_trip_without_stored_vectors(tmp_path, dataset):
    """store_vectors=False is the configuration that realizes the compression,
    so the archive must stay smaller and reranking must stay unavailable."""
    lean = IVFPQIndex(
        nlist=8, m=8, nprobe=4, store_vectors=False, seed=0, kmeans_iter=8
    ).fit(dataset.base, dataset.train)
    target = tmp_path / "lean.tinyvec"
    save_index(lean, target)
    loaded = load_index(target)
    assert loaded.vectors is None
    assert loaded.store_vectors is False
    with pytest.raises(RuntimeError, match="store_vectors=True"):
        loaded.search(dataset.queries, 10, rerank=10)

    fat = tmp_path / "fat.tinyvec"
    save_index(
        IVFPQIndex(nlist=8, m=8, nprobe=4, seed=0, kmeans_iter=8).fit(
            dataset.base, dataset.train
        ),
        fat,
    )
    assert target.stat().st_size < fat.stat().st_size


def test_round_trip_without_precompute(tmp_path, dataset):
    plain = IVFPQIndex(
        nlist=8, m=8, nprobe=4, precompute_tables=False, seed=0, kmeans_iter=8
    ).fit(dataset.base, dataset.train)
    target = tmp_path / "plain.tinyvec"
    save_index(plain, target)
    loaded = load_index(target)
    assert loaded.precompute_tables is False
    assert loaded._precomputed is None
    a, _ = plain.search(dataset.queries, 10, nprobe=4)
    b, _ = loaded.search(dataset.queries, 10, nprobe=4)
    assert np.array_equal(a, b)


# -------------------------------------------------------------- end to end


def test_build_then_query_via_main(tmp_path, capsys):
    target = tmp_path / "cli.tinyvec"
    assert main([
        "build", "--dataset", "synthetic", "--nlist", "8", "--m", "8",
        "--kmeans-iter", "5", "--out", str(target),
    ]) == 0
    out = capsys.readouterr().out
    assert "compression" in out
    assert target.exists()

    assert main([
        "query", "--dataset", "synthetic", "--index", str(target),
        "--nprobe", "4", "--recall", "--n-queries", "20",
    ]) == 0
    out = capsys.readouterr().out
    assert "recall@10" in out
    assert "candidates per query" in out


def test_build_reports_compression_without_saving(capsys):
    """The synthetic set is 128-dimensional, so m=8 gives 4 * 128 / 8 = 64x."""
    assert main([
        "build", "--dataset", "synthetic", "--nlist", "8", "--m", "8", "--kmeans-iter", "5",
    ]) == 0
    assert "64.0x codes only" in capsys.readouterr().out


def test_query_without_an_index_builds_one(capsys):
    assert main([
        "query", "--dataset", "synthetic", "--nlist", "8", "--m", "8",
        "--kmeans-iter", "5", "--n-queries", "10", "--nprobe", "4",
    ]) == 0
    assert "building a fresh one" in capsys.readouterr().out


def test_bench_subcommand(capsys):
    assert main([
        "bench", "--dataset", "synthetic", "--nlist", "8", "--m", "8",
        "--kmeans-iter", "5", "--nprobe", "1,4", "--runs", "2",
    ]) == 0
    out = capsys.readouterr().out
    assert "recall@10" in out
    assert "QPS" in out


def test_missing_index_file_exits_cleanly(capsys):
    code = main(["query", "--dataset", "synthetic", "--index", "/nonexistent/x.tinyvec"])
    assert code == 2
    assert "tinyvec:" in capsys.readouterr().err
