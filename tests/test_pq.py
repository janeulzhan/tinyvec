"""Tests for product quantization: codebooks, round-trip, compression."""

from __future__ import annotations

import numpy as np
import pytest

from tinyvec.pq import ProductQuantizer


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=6.0, size=(24, 64)).astype(np.float32)
    which = rng.integers(24, size=3000)
    return (centers[which] + rng.normal(size=(3000, 64))).astype(np.float32)


def test_codebook_shapes(data):
    pq = ProductQuantizer(8, seed=0, n_iter=10).fit(data)
    assert pq.codebooks.shape == (8, 256, 8)
    assert pq.dim == 64
    assert pq.dsub == 8
    assert pq.is_fitted


def test_codes_are_uint8_and_in_range(data):
    pq = ProductQuantizer(8, seed=0, n_iter=10).fit(data)
    codes = pq.encode(data)
    assert codes.dtype == np.uint8
    assert codes.shape == (3000, 8)
    assert codes.min() >= 0 and codes.max() <= 255


def test_round_trip_error_is_bounded(data):
    """Reconstruction must be much better than collapsing to the global mean,
    which is the trivial baseline any quantizer has to beat."""
    pq = ProductQuantizer(16, seed=0, n_iter=15).fit(data)
    mse = pq.reconstruction_error(data)
    baseline = float(np.mean(np.sum((data - data.mean(axis=0)) ** 2, axis=1)))
    assert mse < 0.25 * baseline
    assert mse > 0.0


def test_round_trip_is_exact_when_codebook_covers_every_point():
    """With fewer distinct vectors than codebook slots, PQ is lossless."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    pq = ProductQuantizer(4, seed=0, n_iter=30).fit(x)
    assert pq.reconstruction_error(x) == pytest.approx(0.0, abs=1e-4)


def test_error_decreases_as_m_increases(data):
    """More subspaces means more bytes per vector and strictly finer
    quantization, so the error must fall monotonically."""
    errors = []
    for m in (2, 4, 8, 16, 32):
        pq = ProductQuantizer(m, seed=0, n_iter=15).fit(data)
        errors.append(pq.reconstruction_error(data))
    assert errors == sorted(errors, reverse=True), errors


def test_error_decreases_as_codebook_size_increases(data):
    """Second axis of the same trade-off: hold m fixed, grow the codebook.
    nbits is pinned at 8 by the uint8 code layout, so this exercises the
    equivalent knob through the k-means codebook directly."""
    sub = data[:, :8]
    from tinyvec.kmeans import kmeans

    errors = []
    for k in (4, 16, 64, 256):
        result = kmeans(sub, k, n_iter=20, seed=0)
        errors.append(result.inertia)
    assert errors == sorted(errors, reverse=True), errors


def test_compression_ratio_is_four_d_over_m(data):
    for m in (4, 8, 16, 32):
        pq = ProductQuantizer(m, seed=0, n_iter=5).fit(data)
        assert pq.compression_ratio() == pytest.approx(4.0 * 64 / m)
        assert pq.code_size == m


def test_actual_byte_sizes_match_the_claimed_ratio(data):
    """The compression claim is checked against real nbytes, not arithmetic."""
    pq = ProductQuantizer(16, seed=0, n_iter=5).fit(data)
    codes = pq.encode(data)
    assert data.nbytes / codes.nbytes == pytest.approx(pq.compression_ratio())
    assert data.nbytes / codes.nbytes == pytest.approx(16.0)


def test_decode_uses_the_codebook_entries(data):
    pq = ProductQuantizer(8, seed=0, n_iter=10).fit(data)
    codes = pq.encode(data[:5])
    decoded = pq.decode(codes)
    for i in range(5):
        for j in range(8):
            want = pq.codebooks[j, codes[i, j]]
            assert np.array_equal(decoded[i, j * 8 : (j + 1) * 8], want)


def test_encode_picks_the_nearest_subcentroid(data):
    """Encoding is argmin per subspace; verify against an independent search."""
    pq = ProductQuantizer(4, seed=0, n_iter=10).fit(data)
    sample = data[:30]
    codes = pq.encode(sample)
    for j in range(4):
        sub = sample[:, j * 16 : (j + 1) * 16]
        book = pq.codebooks[j]
        brute = np.array(
            [int(np.argmin(((book - row) ** 2).sum(axis=1))) for row in sub], dtype=np.uint8
        )
        assert np.array_equal(codes[:, j], brute)


def test_same_seed_is_deterministic(data):
    a = ProductQuantizer(8, seed=11, n_iter=10).fit(data)
    b = ProductQuantizer(8, seed=11, n_iter=10).fit(data)
    assert np.array_equal(a.codebooks, b.codebooks)
    assert np.array_equal(a.encode(data), b.encode(data))


def test_rejects_m_that_does_not_divide_dim(data):
    with pytest.raises(ValueError, match="does not divide"):
        ProductQuantizer(7, seed=0).fit(data)


def test_rejects_unsupported_nbits():
    with pytest.raises(ValueError, match="only nbits=8"):
        ProductQuantizer(4, nbits=4)


def test_rejects_non_positive_m():
    with pytest.raises(ValueError, match="m must be at least 1"):
        ProductQuantizer(0)


def test_methods_raise_before_fit(data):
    pq = ProductQuantizer(8)
    assert not pq.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        pq.encode(data)
    with pytest.raises(RuntimeError, match="not fitted"):
        pq.decode(np.zeros((2, 8), np.uint8))
    with pytest.raises(RuntimeError, match="not fitted"):
        pq.distance_tables(data[:1])
    with pytest.raises(RuntimeError, match="needs a fitted quantizer"):
        pq.compression_ratio()


def test_rejects_wrong_width_input(data):
    pq = ProductQuantizer(8, seed=0, n_iter=5).fit(data)
    with pytest.raises(ValueError, match="expected"):
        pq.encode(np.zeros((3, 63), np.float32))
    with pytest.raises(ValueError, match="expected"):
        pq.decode(np.zeros((3, 9), np.uint8))


def test_rejects_empty_training_set():
    with pytest.raises(ValueError, match="zero vectors"):
        ProductQuantizer(4).fit(np.zeros((0, 16), np.float32))


def test_m_equal_to_dim_is_scalar_quantization(data):
    """m == d degenerates to one codebook per coordinate. It should still work
    and should be the most accurate setting available."""
    pq = ProductQuantizer(64, seed=0, n_iter=10).fit(data)
    assert pq.dsub == 1
    assert pq.compression_ratio() == pytest.approx(4.0)
    coarse = ProductQuantizer(32, seed=0, n_iter=10).fit(data)
    assert pq.reconstruction_error(data) < coarse.reconstruction_error(data)
