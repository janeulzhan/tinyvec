"""Asymmetric distance computation correctness.

This is the most important test file in the repository. ADC replaces a real
distance computation with a sum of table lookups, and the whole index is built
on the claim that those two things agree. If they silently disagree, every
recall number downstream is measuring the wrong quantity. So the claim is tested
directly: the table-gather result must equal a distance recomputed from scratch
on the decoded vectors, to float32 tolerance and no looser.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.pq import ProductQuantizer

# float32 has about 7 decimal digits. Distances here are sums of d squared
# terms, so a few ulps of accumulated rounding is expected and anything larger
# indicates a real bug rather than arithmetic noise.
REL_TOL = 1e-5


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=5.0, size=(20, 48)).astype(np.float32)
    which = rng.integers(20, size=2000)
    x = (centers[which] + rng.normal(size=(2000, 48))).astype(np.float32)
    queries = (centers[rng.integers(20, size=25)] + rng.normal(size=(25, 48))).astype(np.float32)
    pq = ProductQuantizer(12, seed=0, n_iter=15).fit(x)
    return pq, x, queries


def test_distance_table_shape_and_contents(fitted):
    """Each table entry must be the squared distance from the query subvector to
    that subspace centroid. Checked entry by entry against a direct computation."""
    pq, _, queries = fitted
    tables = pq.distance_tables(queries[:3])
    assert tables.shape == (3, 12, 256)
    for i in range(3):
        for j in range(12):
            sub = queries[i, j * 4 : (j + 1) * 4].astype(np.float64)
            want = ((pq.codebooks[j].astype(np.float64) - sub) ** 2).sum(axis=1)
            assert np.allclose(tables[i, j], want, rtol=REL_TOL, atol=1e-3)


def test_adc_equals_direct_distance_on_decoded_vectors(fitted):
    """The headline claim. ADC is exact with respect to the reconstruction: it
    is not an approximation of the decoded distance, it is that distance
    computed by a different route."""
    pq, x, queries = fitted
    codes = pq.encode(x)
    decoded = pq.decode(codes)
    tables = pq.distance_tables(queries)
    for i in range(queries.shape[0]):
        got = pq.adc(codes, tables[i])
        want = ((decoded.astype(np.float64) - queries[i].astype(np.float64)) ** 2).sum(axis=1)
        rel = np.abs(got - want) / np.maximum(want, 1e-9)
        assert rel.max() < REL_TOL, f"query {i}: max relative error {rel.max():.2e}"


def test_adc_equals_naive_table_summation(fitted):
    """Independent reimplementation of the gather, written the obvious slow way
    with an explicit Python loop over subspaces."""
    pq, x, queries = fitted
    codes = pq.encode(x[:200])
    table = pq.distance_tables(queries[:1])[0]
    got = pq.adc(codes, table)
    want = np.zeros(200, dtype=np.float64)
    for n in range(200):
        for j in range(12):
            want[n] += table[j, codes[n, j]]
    assert np.allclose(got, want, rtol=REL_TOL, atol=1e-3)


def test_adc_is_a_lower_variance_estimate_than_symmetric(fitted):
    """Keeping the query unquantized is the 'asymmetric' in ADC and is the whole
    reason not to quantize both sides. Verify it actually estimates the true
    distance better than the symmetric alternative, which quantizes the query
    too and therefore carries two sources of error instead of one."""
    pq, x, queries = fitted
    codes = pq.encode(x)
    decoded = pq.decode(codes)
    q_codes = pq.encode(queries)
    q_decoded = pq.decode(q_codes)
    tables = pq.distance_tables(queries)

    adc_err = []
    sdc_err = []
    for i in range(queries.shape[0]):
        true_d = ((x - queries[i]) ** 2).sum(axis=1)
        adc_err.append(np.abs(pq.adc(codes, tables[i]) - true_d).mean())
        sdc_err.append(np.abs(((decoded - q_decoded[i]) ** 2).sum(axis=1) - true_d).mean())
    assert np.mean(adc_err) < np.mean(sdc_err)


def test_adc_preserves_ranking_of_well_separated_candidates(fitted):
    """ADC may misrank near-ties, which is expected and is what reranking fixes.
    It must not misrank candidates that are far apart."""
    pq, x, queries = fitted
    codes = pq.encode(x)
    table = pq.distance_tables(queries[:1])[0]
    approx = pq.adc(codes, table)
    true_d = ((x - queries[0]) ** 2).sum(axis=1)
    order = np.argsort(true_d)
    near, far = order[:20], order[-20:]
    assert approx[near].max() < approx[far].min()


def test_adc_is_non_negative(fitted):
    pq, x, queries = fitted
    codes = pq.encode(x)
    tables = pq.distance_tables(queries)
    for i in range(queries.shape[0]):
        assert np.all(pq.adc(codes, tables[i]) >= 0.0)


def test_adc_of_a_vector_against_itself_is_its_reconstruction_error(fitted):
    """Querying with a stored vector should give back exactly that vector's
    squared reconstruction error, since the query side contributes none."""
    pq, x, _ = fitted
    codes = pq.encode(x[:50])
    decoded = pq.decode(codes)
    tables = pq.distance_tables(x[:50])
    for i in range(50):
        got = pq.adc(codes[i : i + 1], tables[i])[0]
        want = float(((decoded[i].astype(np.float64) - x[i].astype(np.float64)) ** 2).sum())
        assert got == pytest.approx(want, rel=REL_TOL, abs=1e-3)


def test_adc_on_empty_code_array(fitted):
    pq, _, queries = fitted
    table = pq.distance_tables(queries[:1])[0]
    out = pq.adc(np.empty((0, 12), np.uint8), table)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_adc_accepts_a_single_code(fitted):
    pq, x, queries = fitted
    codes = pq.encode(x[:1])
    table = pq.distance_tables(queries[:1])[0]
    assert pq.adc(codes, table).shape == (1,)


def test_adc_rejects_mismatched_shapes(fitted):
    pq, x, queries = fitted
    codes = pq.encode(x[:10])
    table = pq.distance_tables(queries[:1])[0]
    with pytest.raises(ValueError, match="expected"):
        pq.adc(codes[:, :5], table)
    with pytest.raises(ValueError, match="expected"):
        pq.adc(codes, table[:, :128])


def test_distance_tables_accepts_a_1d_query(fitted):
    pq, _, queries = fitted
    one = pq.distance_tables(queries[0])
    batch = pq.distance_tables(queries[:1])
    assert one.shape == (1, 12, 256)
    assert np.array_equal(one, batch)


@pytest.mark.parametrize("m", [2, 4, 8, 16])
def test_adc_exactness_holds_across_m(m):
    """The identity is structural, not tuned for one configuration."""
    rng = np.random.default_rng(m)
    x = rng.normal(size=(500, 32)).astype(np.float32)
    q = rng.normal(size=(5, 32)).astype(np.float32)
    pq = ProductQuantizer(m, seed=0, n_iter=10).fit(x)
    codes = pq.encode(x)
    decoded = pq.decode(codes).astype(np.float64)
    tables = pq.distance_tables(q)
    for i in range(5):
        got = pq.adc(codes, tables[i])
        want = ((decoded - q[i].astype(np.float64)) ** 2).sum(axis=1)
        assert np.abs(got - want).max() / max(want.max(), 1e-9) < REL_TOL
