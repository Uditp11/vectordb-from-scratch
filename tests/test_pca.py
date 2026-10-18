"""pca_2d must match a NumPy SVD projection, up to the sign of each axis."""

import numpy as np
import pytest

from pca import pca_2d


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pca_matches_svd_up_to_sign(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((50, 6)) * [5, 3, 1, 1, 0.5, 0.1]
    got = np.array(pca_2d(X.tolist()))

    centered = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    expected = centered @ vt[:2].T

    for axis in range(2):
        sign = np.sign(got[:, axis] @ expected[:, axis])
        np.testing.assert_allclose(got[:, axis] * sign, expected[:, axis], atol=1e-8)


def test_pca_small_inputs():
    assert pca_2d([]) == []
    assert pca_2d([[1.0, 2.0, 3.0]]) == [[0.0, 0.0]]
