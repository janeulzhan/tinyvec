"""Tests for the benchlock README consistency gate.

benchlock is the thing that stops the README's numbers from drifting away from
the code, so it needs to be trustworthy itself. Its two interesting failure
modes are a false pass (drift that slips through, which silently makes the
README a lie) and a false alarm (tolerance too tight, which trains people to
ignore the gate). Both are tested here, along with the refusal to lock anything
measured in wall-clock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.benchlock import (  # noqa: E402
    BenchlockError,
    at_displayed_precision,
    check,
    displayed_decimals,
    find_markers,
    flatten,
    format_like,
    load_metrics,
    tolerance_for,
    update,
)

METRICS = {
    "recall": {"nprobe1": 0.403, "nprobe8": 0.700},
    "recall_rerank": {"nprobe8": 0.925},
    "candidates": {"nprobe8": 665.55},
    "memory": {"compression_ratio": 32.0, "original_mb": 5.12},
    "quality": {"residual_reconstruction_error": 12071.06},
    "dataset": {"n": 10000, "dim": 128},
    "timing_not_gated": {"build_seconds": 6.63, "qps": {"adc.nprobe8": 13960.1}},
}


@pytest.fixture
def flat():
    return flatten(METRICS)


# ------------------------------------------------------------------ flatten


def test_flatten_produces_dotted_keys(flat):
    assert flat["recall.nprobe8"] == 0.700
    assert flat["memory.compression_ratio"] == 32.0
    assert flat["timing_not_gated.qps.adc.nprobe8"] == 13960.1


def test_flatten_coerces_ints_to_float(flat):
    assert isinstance(flat["dataset.n"], float)
    assert flat["dataset.n"] == 10000.0


def test_flatten_drops_booleans_and_strings():
    out = flatten({"a": True, "b": "text", "c": 1.5, "d": None})
    assert out == {"c": 1.5}


# ------------------------------------------------------------------- markers


def test_find_markers_in_document_order():
    text = "a <!--bench:recall.nprobe1-->0.403<!--/bench--> b <!--bench:dataset.n-->10000<!--/bench-->"
    assert find_markers(text) == [("recall.nprobe1", "0.403"), ("dataset.n", "10000")]


def test_find_markers_tolerates_whitespace_in_the_tag():
    text = "<!-- bench:recall.nprobe8 -->0.700<!-- /bench -->"
    assert find_markers(text) == [("recall.nprobe8", "0.700")]


def test_unclosed_marker_is_an_error():
    """A dangling open tag must not be silently skipped, because the stale value
    beside it would keep passing review forever."""
    with pytest.raises(BenchlockError, match="malformed bench markers"):
        find_markers("value is <!--bench:recall.nprobe8-->0.700 and then nothing")


def test_stray_closing_marker_is_an_error():
    with pytest.raises(BenchlockError, match="malformed bench markers"):
        find_markers("orphan <!--/bench--> here")


def test_no_markers_is_not_an_error():
    assert find_markers("a README with no locked numbers at all") == []


# ----------------------------------------------------------------- tolerance


def test_tolerance_longest_prefix_wins():
    assert tolerance_for("memory.compression_ratio") == 0.0
    assert tolerance_for("recall.nprobe8") == 0.05
    assert tolerance_for("quality.residual_reconstruction_error") == 0.10


def test_unknown_key_has_no_tolerance():
    with pytest.raises(BenchlockError, match="no tolerance declared"):
        tolerance_for("something.invented")


# ------------------------------------------------------- displayed precision


def test_displayed_decimals():
    assert displayed_decimals("0.4030") == 4
    assert displayed_decimals("2.1") == 1
    assert displayed_decimals("10000") is None
    assert displayed_decimals("32.0x") == 1
    assert displayed_decimals("48.4%") == 1
    assert displayed_decimals(" 10,000 ") is None


def test_at_displayed_precision_rounds_to_what_the_readme_prints():
    assert at_displayed_precision(2.0531, "2.1") == 2.1
    assert at_displayed_precision(2.0531, "2.05") == 2.05
    assert at_displayed_precision(2.0531, "2") == 2.0
    assert at_displayed_precision(0.7, "0.7000") == 0.7


def test_zero_tolerance_survives_display_rounding(flat):
    """A README showing one decimal cannot be expected to equal 2.0531 exactly.
    Rounding the measured value to the committed precision is what lets byte
    ratios keep a genuinely zero tolerance without forcing an ugly document."""
    metrics = flatten({"memory": {"compression_ratio": 2.0531}})
    assert check("<!--bench:memory.compression_ratio-->2.1<!--/bench-->", metrics) == []
    assert check("<!--bench:memory.compression_ratio-->2.05<!--/bench-->", metrics) == []
    # Rounding is not a licence to be wrong: a genuinely different value at the
    # same precision still fails.
    assert len(check("<!--bench:memory.compression_ratio-->2.6<!--/bench-->", metrics)) == 1


def test_display_rounding_does_not_mask_real_drift(flat):
    """Two decimal places shown means two decimal places enforced."""
    metrics = flatten({"recall": {"nprobe8": 0.70}})
    assert check("<!--bench:recall.nprobe8-->0.70<!--/bench-->", metrics) == []
    assert len(check("<!--bench:recall.nprobe8-->0.40<!--/bench-->", metrics)) == 1


# --------------------------------------------------------------------- check


def test_exact_match_passes(flat):
    text = "<!--bench:recall.nprobe8-->0.700<!--/bench-->"
    assert check(text, flat) == []


def test_drift_within_tolerance_passes(flat):
    """0.690 against a measured 0.700 is 1.4 percent, inside the 5 percent
    allowance that exists to absorb BLAS variation."""
    text = "<!--bench:recall.nprobe8-->0.690<!--/bench-->"
    assert check(text, flat) == []


def test_drift_past_tolerance_fails(flat):
    text = "<!--bench:recall.nprobe8-->0.500<!--/bench-->"
    drifts = check(text, flat)
    assert len(drifts) == 1
    assert drifts[0].key == "recall.nprobe8"
    assert drifts[0].committed == 0.500
    assert drifts[0].measured == 0.700
    assert drifts[0].relative == pytest.approx(0.2857, abs=1e-3)
    assert "README says 0.5" in drifts[0].describe()


def test_zero_tolerance_key_rejects_any_drift(flat):
    """Byte counts and ratios are exact integer arithmetic, so even a small
    discrepancy is a real bug rather than noise."""
    assert check("<!--bench:memory.compression_ratio-->32.0<!--/bench-->", flat) == []
    assert len(check("<!--bench:memory.compression_ratio-->32.1<!--/bench-->", flat)) == 1


def test_multiple_drifts_are_all_reported(flat):
    text = (
        "<!--bench:recall.nprobe8-->0.100<!--/bench--> "
        "<!--bench:recall.nprobe1-->0.900<!--/bench--> "
        "<!--bench:dataset.n-->10000<!--/bench-->"
    )
    drifts = check(text, flat)
    assert {d.key for d in drifts} == {"recall.nprobe8", "recall.nprobe1"}


def test_unknown_key_in_readme_is_an_error(flat):
    with pytest.raises(BenchlockError, match="not present in the metrics"):
        check("<!--bench:recall.nprobe999-->0.5<!--/bench-->", flat)


def test_non_numeric_value_is_an_error(flat):
    with pytest.raises(BenchlockError, match="is not a number"):
        check("<!--bench:recall.nprobe8-->roughly seventy percent<!--/bench-->", flat)


def test_suffixed_values_are_parsed(flat):
    assert check("<!--bench:memory.compression_ratio-->32.0x<!--/bench-->", flat) == []


def test_thousands_separators_are_parsed(flat):
    assert check("<!--bench:dataset.n-->10,000<!--/bench-->", flat) == []


def test_wall_clock_keys_are_refused(flat):
    """The core policy: timings may be reported but never gated."""
    with pytest.raises(BenchlockError, match="must not be locked"):
        check("<!--bench:timing_not_gated.build_seconds-->6.63<!--/bench-->", flat)
    with pytest.raises(BenchlockError, match="must not be locked"):
        check("<!--bench:timing_not_gated.qps.adc.nprobe8-->13960.1<!--/bench-->", flat)


# -------------------------------------------------------------------- update


def test_update_rewrites_a_drifted_value(flat):
    text = "recall is <!--bench:recall.nprobe8-->0.500<!--/bench--> today"
    new_text, changed = update(text, flat)
    assert changed == 1
    assert "<!--bench:recall.nprobe8-->0.500<!--/bench-->" not in new_text
    assert check(new_text, flat) == []


def test_update_is_idempotent(flat):
    text = "<!--bench:recall.nprobe8-->0.123<!--/bench-->"
    once, first = update(text, flat)
    twice, second = update(once, flat)
    assert first == 1
    assert second == 0
    assert once == twice


def test_update_preserves_decimal_places(flat):
    text = "<!--bench:recall.nprobe8-->0.1<!--/bench-->"
    new_text, _ = update(text, flat)
    assert ">0.7<" in new_text
    text4 = "<!--bench:recall.nprobe8-->0.1000<!--/bench-->"
    new_text4, _ = update(text4, flat)
    assert ">0.7000<" in new_text4


def test_update_preserves_suffixes(flat):
    text = "<!--bench:memory.compression_ratio-->1.0x<!--/bench-->"
    new_text, _ = update(text, flat)
    assert ">32.0x<" in new_text


def test_update_preserves_integer_formatting(flat):
    text = "<!--bench:dataset.n-->1<!--/bench-->"
    new_text, _ = update(text, flat)
    assert ">10000<" in new_text


def test_update_rejects_malformed_markers_before_writing(flat):
    with pytest.raises(BenchlockError, match="malformed"):
        update("<!--bench:recall.nprobe8-->0.5", flat)


def test_update_refuses_wall_clock_keys(flat):
    with pytest.raises(BenchlockError, match="must not be locked"):
        update("<!--bench:timing_not_gated.build_seconds-->1.0<!--/bench-->", flat)


def test_format_like_cases():
    assert format_like("0.500", 0.7) == "0.700"
    assert format_like(" 0.50 ", 0.7) == " 0.70 "
    assert format_like("12x", 32.0) == "32x"
    assert format_like("5", 10000.0) == "10000"
    assert format_like("1.5%", 48.4) == "48.4%"


# ---------------------------------------------------------------- end to end


def test_round_trip_on_a_fixture_readme(tmp_path, flat):
    """A miniature README exercising every path at once: a correct value, one
    drifted past tolerance, one drifted but inside it, and suffixed formatting."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(METRICS))

    readme = tmp_path / "README.md"
    readme.write_text(
        "# fixture\n\n"
        "| nprobe | recall@10 |\n"
        "| --- | --- |\n"
        "| 1 | <!--bench:recall.nprobe1-->0.403<!--/bench--> |\n"
        "| 8 | <!--bench:recall.nprobe8-->0.250<!--/bench--> |\n"
        "| 8 reranked | <!--bench:recall_rerank.nprobe8-->0.920<!--/bench--> |\n\n"
        "Compression is <!--bench:memory.compression_ratio-->32.0x<!--/bench-->.\n"
    )

    loaded = load_metrics(metrics_path, rerun=False)
    assert loaded == flat

    drifts = check(readme.read_text(), loaded)
    assert [d.key for d in drifts] == ["recall.nprobe8"]

    # update normalizes every span to the measured value, while check only
    # flags the ones past tolerance. So the in-tolerance 0.920 -> 0.925 row is
    # rewritten too, and two values change even though only one had drifted.
    new_text, changed = update(readme.read_text(), loaded)
    readme.write_text(new_text)
    assert changed == 2
    assert check(readme.read_text(), loaded) == []
    assert "<!--bench:recall_rerank.nprobe8-->0.925<!--/bench-->" in new_text
    # Formatting of the untouched rows is preserved exactly.
    assert "| 1 | <!--bench:recall.nprobe1-->0.403<!--/bench--> |" in new_text
    assert "<!--bench:memory.compression_ratio-->32.0x<!--/bench-->" in new_text


def test_load_metrics_reports_a_missing_file(tmp_path):
    with pytest.raises(BenchlockError, match="not found"):
        load_metrics(tmp_path / "absent.json", rerun=False)


def test_the_real_readme_is_locked_and_consistent():
    """Guards the committed README against the committed metrics. This is the
    assertion that CI actually runs."""
    repo = Path(__file__).resolve().parent.parent
    readme = repo / "README.md"
    metrics = repo / "bench" / "metrics.json"
    if not readme.exists() or not metrics.exists():
        pytest.skip("README or metrics not present")
    loaded = load_metrics(metrics, rerun=False)
    markers = find_markers(readme.read_text())
    assert markers, "README has no locked numbers; benchlock would be vacuous"
    assert check(readme.read_text(), loaded) == []
