"""benchlock: keep the numbers in the README honest, mechanically.

A README full of hand-typed measurements rots the moment the code changes, and
nobody notices because prose does not fail a build. benchlock makes the numbers
into something CI can check.

Every claimed number in the README lives inside a marker span:

    recall@10 at nprobe=8 is <!--bench:recall.nprobe8-->0.7000<!--/bench-->

``--update`` runs the benchmark, flattens its JSON output to dotted keys, and
rewrites the value inside each span. ``--check`` does the same but writes
nothing and exits non-zero if any committed value has drifted past the tolerance
declared for its key. So the README cannot silently disagree with the code.

What may be locked
------------------
Only deterministic quantities: recall at k, reconstruction error, candidate
counts, memory sizes, compression ratios. These are functions of the code, the
data and the seed, so a tolerance can be set tight enough to catch a real
regression.

Wall-clock numbers are deliberately **not** lockable. ``TOLERANCES`` has no
entry for anything under ``timing_not_gated``, and :func:`check` refuses to gate
a key that falls under it. The reasoning is in the README: a timing tolerance
tight enough to detect a real slowdown is also tight enough to fire on a noisy
shared runner, so gating it would train everyone to ignore the failure. Timing
claims live in their own table, labelled with the hardware they came from.

Usage:
    python3 tools/benchlock.py --update      # rerun bench, rewrite README spans
    python3 tools/benchlock.py --check       # CI gate, no writes
    python3 tools/benchlock.py --check --metrics bench/metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
METRICS = REPO_ROOT / "bench" / "metrics.json"
BENCH = REPO_ROOT / "bench" / "bench.py"

MARKER = re.compile(
    r"<!--\s*bench:(?P<key>[A-Za-z0-9_.]+)\s*-->(?P<value>.*?)<!--\s*/bench\s*-->",
    re.DOTALL,
)

# Any key under this prefix is a wall-clock measurement and is never gated.
UNGATED_PREFIX = "timing_not_gated"

# Relative tolerance per key prefix, longest prefix wins. Recall is allowed a
# little room because k-means centroids depend on float32 GEMM accumulation
# order, which differs between BLAS builds. Byte counts and ratios are exact
# integer arithmetic, so they get no slack at all.
TOLERANCES: dict[str, float] = {
    "recall": 0.05,
    "recall_rerank": 0.05,
    "candidates": 0.05,
    "quality.residual_reconstruction_error": 0.10,
    # Float32 rounding residuals. These vary with BLAS accumulation order more
    # than recall does, so they get room; they are quoted in the README as
    # orders of magnitude, not as precise values.
    "quality.adc_max_relative_error": 0.50,
    "quality.precompute_max_scaled_distance_diff": 0.50,
    # Counts of nprobe values returning byte-identical ids. Exact integers.
    "quality.precompute_identical_id_nprobe_values": 0.0,
    "quality.precompute_nprobe_values_compared": 0.0,
    "memory.compression_ratio": 0.0,
    "memory.compression_ratio_with_overhead": 0.0,
    "memory.original_mb": 0.0,
    "memory.codes_mb": 0.0,
    "dataset": 0.0,
}


class BenchlockError(RuntimeError):
    """Raised for malformed markers and other unusable input."""


@dataclass(frozen=True)
class Drift:
    """One README value that disagrees with the measured metrics."""

    key: str
    committed: float
    measured: float
    tolerance: float

    @property
    def relative(self) -> float:
        scale = abs(self.measured) if self.measured else 1.0
        return abs(self.committed - self.measured) / scale

    def describe(self) -> str:
        return (
            f"{self.key}: README says {self.committed:g}, benchmark says "
            f"{self.measured:g} (relative drift {self.relative:.4f}, "
            f"tolerance {self.tolerance:.4f})"
        )


def flatten(obj: object, prefix: str = "") -> dict[str, float]:
    """Flatten nested metrics JSON to dotted keys, keeping only numbers."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def tolerance_for(key: str) -> float:
    """Longest matching prefix wins, so a specific key can override its group."""
    best: tuple[int, float] | None = None
    for prefix, tol in TOLERANCES.items():
        if key == prefix or key.startswith(prefix + "."):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), tol)
    if best is None:
        raise BenchlockError(
            f"no tolerance declared for {key!r}. Add one to TOLERANCES, and only "
            f"if the quantity is deterministic."
        )
    return best[1]


def find_markers(text: str) -> list[tuple[str, str]]:
    """Return ``(key, raw_value)`` for every marker span, in document order.

    Raises:
        BenchlockError: If a span is opened and never closed, which would
            otherwise be silently skipped and leave a stale number in place.
    """
    spans = [(m.group("key"), m.group("value")) for m in MARKER.finditer(text)]
    opens = len(re.findall(r"<!--\s*bench:", text))
    closes = len(re.findall(r"<!--\s*/bench\s*-->", text))
    if opens != closes or opens != len(spans):
        raise BenchlockError(
            f"malformed bench markers: {opens} opening tags, {closes} closing tags, "
            f"{len(spans)} well-formed spans. Every <!--bench:key--> needs a "
            f"matching <!--/bench-->."
        )
    return spans


def displayed_decimals(raw: str) -> int | None:
    """Decimal places the README shows for a value, or None if it shows none."""
    stripped = raw.strip().replace(",", "").rstrip("x%")
    if "." not in stripped:
        return None
    return len(stripped.split(".", 1)[1])


def at_displayed_precision(measured: float, raw: str) -> float:
    """Round ``measured`` to the precision the README actually prints.

    A README showing "2.1" cannot be expected to equal 2.0531 exactly, and
    demanding that it does would force either an ugly full-precision document
    or a tolerance that also hides real drift. The question a document lock
    should ask is whether the *displayed* number still round-trips, so the
    measured value is rounded to the committed value's own precision before any
    comparison. Zero-tolerance keys then stay genuinely exact at the precision
    they claim.
    """
    decimals = displayed_decimals(raw)
    if decimals is None:
        return float(round(measured))
    return round(float(measured), decimals)


def parse_committed(key: str, raw: str) -> float:
    cleaned = raw.strip().replace(",", "").replace("x", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise BenchlockError(f"{key}: value {raw.strip()!r} is not a number") from exc


def format_like(previous: str, value: float) -> str:
    """Rewrite ``value`` preserving the committed formatting.

    Keeps decimal places, any trailing ``x`` or ``%``, and surrounding
    whitespace, so an update is a pure numeric diff and does not churn the
    README's layout.
    """
    stripped = previous.strip()
    suffix = ""
    for candidate in ("x", "%"):
        if stripped.endswith(candidate):
            suffix = candidate
            stripped = stripped[: -len(candidate)]
            break
    if "." in stripped:
        decimals = len(stripped.split(".", 1)[1])
        body = f"{value:.{decimals}f}"
    elif float(value).is_integer():
        body = str(int(value))
    else:
        body = f"{value:g}"
    leading = previous[: len(previous) - len(previous.lstrip())]
    trailing = previous[len(previous.rstrip()) :]
    return f"{leading}{body}{suffix}{trailing}"


def load_metrics(path: Path, *, rerun: bool) -> dict[str, float]:
    if rerun:
        result = subprocess.run(
            [sys.executable, str(BENCH), "--json", str(path)],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise BenchlockError(f"benchmark failed with exit code {result.returncode}")
    if not path.exists():
        raise BenchlockError(
            f"metrics file {path} not found. Run bench/bench.py --json {path} first, "
            f"or pass --run."
        )
    return flatten(json.loads(path.read_text()))


def check(text: str, metrics: dict[str, float]) -> list[Drift]:
    """Compare every marker span against the metrics.

    Returns:
        A list of drifts past tolerance. Empty means the README is consistent.

    Raises:
        BenchlockError: On malformed markers, unknown keys, non-numeric values,
            or an attempt to gate a wall-clock quantity.
    """
    drifts: list[Drift] = []
    for key, raw in find_markers(text):
        if key == UNGATED_PREFIX or key.startswith(UNGATED_PREFIX + "."):
            raise BenchlockError(
                f"{key!r} is a wall-clock measurement and must not be locked. "
                f"Timing claims belong in the unverified-timings table, not in a "
                f"bench marker: a tolerance tight enough to be meaningful on a "
                f"shared runner is tight enough to flake."
            )
        if key not in metrics:
            raise BenchlockError(
                f"{key!r} is not present in the metrics. Available keys include: "
                f"{', '.join(sorted(metrics)[:8])}..."
            )
        committed = parse_committed(key, raw)
        tol = tolerance_for(key)
        drift = Drift(key, committed, at_displayed_precision(metrics[key], raw), tol)
        if drift.relative > tol:
            drifts.append(drift)
    return drifts


def update(text: str, metrics: dict[str, float]) -> tuple[str, int]:
    """Rewrite every marker span to the measured value.

    Returns:
        ``(new_text, n_changed)``.
    """
    changed = 0
    find_markers(text)  # validate before touching anything

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        key = match.group("key")
        raw = match.group("value")
        if key == UNGATED_PREFIX or key.startswith(UNGATED_PREFIX + "."):
            raise BenchlockError(f"{key!r} is a wall-clock quantity and must not be locked")
        if key not in metrics:
            raise BenchlockError(f"{key!r} is not present in the metrics")
        tolerance_for(key)
        rendered = format_like(raw, metrics[key])
        if rendered != raw:
            changed += 1
        return f"<!--bench:{key}-->{rendered}<!--/bench-->"

    return MARKER.sub(replace, text), changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if the README has drifted")
    mode.add_argument("--update", action="store_true", help="rewrite README values")
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--run", action="store_true", help="rerun the benchmark first")
    args = parser.parse_args()

    try:
        metrics = load_metrics(args.metrics, rerun=args.run or args.update)
        text = args.readme.read_text()

        if args.update:
            new_text, changed = update(text, metrics)
            args.readme.write_text(new_text)
            print(f"benchlock: updated {changed} value(s) in {args.readme}")
            return 0

        drifts = check(text, metrics)
        locked = len(find_markers(text))
        if drifts:
            print(f"benchlock: {len(drifts)} of {locked} locked value(s) drifted", file=sys.stderr)
            for drift in drifts:
                print(f"  {drift.describe()}", file=sys.stderr)
            print("\nRerun tools/benchlock.py --update if the change is intended.", file=sys.stderr)
            return 1
        print(f"benchlock: all {locked} locked value(s) match the benchmark")
        return 0
    except BenchlockError as exc:
        print(f"benchlock: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
