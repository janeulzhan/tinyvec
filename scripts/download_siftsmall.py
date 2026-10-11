"""Regenerate data/siftsmall.npz from the upstream TEXMEX tarball.

The repository already ships the resulting archive, so you never need to run
this to use tinyvec. It exists so the vendored copy is reproducible: anyone can
re-derive it from the original INRIA distribution and confirm it matches.

The download goes through ``curl`` rather than ``urllib`` because the upstream
host serves over FTP and the ``urllib`` path additionally needs a CA bundle that
a bare Python install may not have.

Usage:
    python3 scripts/download_siftsmall.py [--keep-raw]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
EXPECTED_SHA256 = "b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "siftsmall.npz"


def read_fvecs(path: Path) -> np.ndarray:
    """Parse the TEXMEX .fvecs format: int32 dim, then dim float32, repeated."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    """Parse the TEXMEX .ivecs format: int32 dim, then dim int32, repeated."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return raw.reshape(-1, dim + 1)[:, 1:].copy()


def to_uint8(name: str, array: np.ndarray) -> np.ndarray:
    """Narrow SIFT descriptors to uint8, refusing to do so lossily."""
    if not np.all(array == np.round(array)):
        raise ValueError(f"{name}: values are not integral, refusing to narrow to uint8")
    if array.min() < 0 or array.max() > 255:
        raise ValueError(f"{name}: range [{array.min()}, {array.max()}] does not fit in uint8")
    return array.astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-raw", action="store_true", help="leave the extracted files in data/raw/")
    args = parser.parse_args()

    if shutil.which("curl") is None:
        print("curl not found on PATH", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "siftsmall.tar.gz"
        print(f"fetching {URL}")
        result = subprocess.run(
            ["curl", "-sSL", "--max-time", "300", "-o", str(tarball), URL],
            check=False,
        )
        if result.returncode != 0 or not tarball.exists():
            print("download failed", file=sys.stderr)
            return 1

        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            print(f"sha256 mismatch\n  expected {EXPECTED_SHA256}\n  got      {digest}", file=sys.stderr)
            return 1
        print(f"sha256 ok: {digest}")

        extract_to = (REPO_ROOT / "data" / "raw") if args.keep_raw else Path(tmp)
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as archive:
            archive.extractall(extract_to, filter="data")
        source = extract_to / "siftsmall"

        base = read_fvecs(source / "siftsmall_base.fvecs")
        query = read_fvecs(source / "siftsmall_query.fvecs")
        learn = read_fvecs(source / "siftsmall_learn.fvecs")
        groundtruth = read_ivecs(source / "siftsmall_groundtruth.ivecs")

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT_PATH,
            base=to_uint8("base", base),
            query=to_uint8("query", query),
            learn=to_uint8("learn", learn),
            groundtruth=groundtruth.astype(np.int32),
        )

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(
        f"wrote {OUT_PATH} ({size_mb:.2f} MB)\n"
        f"  base={base.shape} query={query.shape} "
        f"learn={learn.shape} groundtruth={groundtruth.shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
