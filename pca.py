"""
pca.py — Principal Component Analysis projection to 2D.

Used to visualize high-dimensional vectors on a 2D scatter plot.
Only numpy is required.
"""

import numpy as np


def pca_2d(vectors):
    """
    Project a list of vectors down to 2 dimensions using PCA.

    Steps:
      1. Center the data (subtract the mean).
      2. Compute the covariance matrix.
      3. Find eigenvalues/eigenvectors via numpy.linalg.eigh.
      4. Project onto the top-2 eigenvectors (largest variance).

    Parameters
    ----------
    vectors : list[list[float]]
        N vectors, each of dimension D.

    Returns
    -------
    list[list[float]]
        N points, each [x, y], representing the 2D projection.
    """
    if not vectors or len(vectors) < 2:
        # Not enough data for meaningful PCA
        return [[0.0, 0.0] for _ in vectors] if vectors else []

    # Convert to numpy array: shape (N, D)
    X = np.array(vectors, dtype=np.float64)

    # 1. Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. Compute covariance matrix: shape (D, D)
    #    Using (X^T · X) / (N-1) for the unbiased estimator
    N = X_centered.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (N - 1)

    # 3. Eigendecomposition (eigh returns sorted ascending)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The last two eigenvalues are the largest → take last two eigenvectors
    # eigenvectors[:, -1] is the one with the largest eigenvalue
    top2 = eigenvectors[:, -2:]  # shape (D, 2)

    # Flip so the first column is the principal component
    top2 = top2[:, ::-1]

    # 4. Project: (N, D) @ (D, 2) → (N, 2)
    projected = np.dot(X_centered, top2)

    # Return as plain Python list of [x, y]
    return projected.tolist()
