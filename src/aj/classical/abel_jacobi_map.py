"""
Classical Abel-Jacobi forward pass: divisor (points + weights) -> Jacobian coordinates.

Given precomputed integral table I_plus on a grid, maps a divisor D = Σ ε_k P_k
to a vector in R^{2g} (real/imag of the complex g-vector) by summing
ε_k * A(P_k) after optional standardization.
"""

import numpy as np


def _bilinear_sample_numpy(I_plus, grid_r, grid_i, x, y):
    """Bilinear interpolation of I_plus at (x, y). Returns (g,) complex."""
    g, H, W = I_plus.shape
    r_min, r_max = float(grid_r.min()), float(grid_r.max())
    i_min, i_max = float(grid_i.min()), float(grid_i.max())
    # Normalize to [0, H-1] and [0, W-1]
    nx = (x - r_min) / (r_max - r_min) * (W - 1) if W > 1 else 0.0
    ny = (y - i_min) / (i_max - i_min) * (H - 1) if H > 1 else 0.0
    ix0 = max(0, min(int(np.floor(nx)), W - 2))
    iy0 = max(0, min(int(np.floor(ny)), H - 2))
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    tx = nx - ix0
    ty = ny - iy0
    out = np.empty(g, dtype=I_plus.dtype)
    for k in range(g):
        v00 = I_plus[k, iy0, ix0]
        v10 = I_plus[k, iy0, ix1]
        v01 = I_plus[k, iy1, ix0]
        v11 = I_plus[k, iy1, ix1]
        out[k] = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
    return out


def abel_jacobi_forward(
    points,
    weights,
    I_plus,
    grid_r,
    grid_i,
    mu=None,
    sigma=None,
    gamma=1.0,
):
    """Classical Abel-Jacobi forward: divisor (points, weights) -> R^{2g}.

    Computes V = γ * Σ_k weight_k * (A(P_k) - μ) / σ where A(P_k) is the
    bilinear interpolation of the integral table at point P_k. If mu/sigma
    are None, no standardization is applied.

    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Each row is (x, y) = (real, imag) on the complex plane.
    weights : array-like of shape (n_points,)
        Coefficient (e.g. sheet sign) for each point.
    I_plus : array-like of shape (g, H, W)
        Precomputed Abel-Jacobi integrals (complex) on the grid.
    grid_r : array-like of length W
        Real coordinates of the grid.
    grid_i : array-like of length H
        Imaginary coordinates of the grid.
    mu : array-like of shape (2*g,) or None
        Per-channel mean (Re then Im of I_plus). If None, no subtraction.
    sigma : array-like of shape (2*g,) or None
        Per-channel std. If None, no scaling.
    gamma : float
        Global scale on the output.

    Returns
    -------
    coords : np.ndarray of shape (2*g,)
        Real vector: [Re(A(D)_0), ..., Re(A(D)_{g-1}), Im(A(D)_0), ..., Im(A(D)_{g-1})].
    """
    I_plus = np.asarray(I_plus)
    grid_r = np.asarray(grid_r)
    grid_i = np.asarray(grid_i)
    points = np.asarray(points)
    weights = np.asarray(weights)
    g = I_plus.shape[0]
    n_points = points.shape[0]
    if weights.ndim == 0:
        weights = np.broadcast_to(weights, (n_points,))
    assert points.shape == (n_points, 2) and len(weights) == n_points

    sum_re = np.zeros(g)
    sum_im = np.zeros(g)
    for i in range(n_points):
        x, y = points[i, 0], points[i, 1]
        val = _bilinear_sample_numpy(I_plus, grid_r, grid_i, x, y)  # (g,) complex
        w = weights[i]
        sum_re += w * val.real
        sum_im += w * val.imag
    coords_2g = np.concatenate([sum_re, sum_im])
    if mu is not None:
        mu = np.asarray(mu)
        coords_2g = coords_2g - mu
    if sigma is not None:
        sigma = np.asarray(sigma)
        sigma = np.where(sigma > 1e-10, sigma, 1.0)
        coords_2g = coords_2g / sigma
    coords_2g = gamma * coords_2g
    return coords_2g


def compute_aj_normalization(I_plus):
    """Compute per-channel mean and std of I_plus (Re then Im over H×W).

    Uses nanmean/nanstd so that NaN entries (e.g. from integration near branch
    points) do not propagate; any remaining NaN in mu/sigma is replaced with
    safe defaults so the model never sees NaN normalization.

    Same statistics as the inline ``mu`` / ``sigma`` block in
    ``AJ_training_genus30.ipynb`` (per-channel mean/std of stacked Re/Im integrals).

    Returns
    -------
    mu : np.ndarray of shape (2*g,)
    sigma : np.ndarray of shape (2*g,)
    """
    if hasattr(I_plus, "cpu"):
        I_plus = I_plus.cpu().numpy()
    else:
        I_plus = np.asarray(I_plus)
    g, H, W = I_plus.shape
    I_re = I_plus.real  # (g, H, W)
    I_im = I_plus.imag
    I_ch = np.concatenate([I_re, I_im], axis=0)  # (2g, H, W)
    mu = np.nanmean(I_ch.reshape(2 * g, -1), axis=1)
    sigma = np.nanstd(I_ch.reshape(2 * g, -1), axis=1)
    sigma = np.maximum(sigma, 1e-6)
    # If a channel was all NaN, nanmean/nanstd return NaN; use safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma
