"""
Period tables: evaluate ω_k and Abel-Jacobi integrals on a grid.

Produces omega_plus (g, H, W) and I_plus (g, H, W) for use in the classical
AJ forward pass. Uses mpmath for integration.
"""

import numpy as np
from tqdm import tqdm

from .differentials import make_omega, integrate_omega


def build_omega_table(
    genus,
    branch_points,
    grid_r,
    grid_i,
    dtype=np.complex128,
):
    """Evaluate ω_k(z) on a 2D grid for k = 0, ..., genus-1.

    Parameters
    ----------
    genus : int
        Number of differentials (g).
    branch_points : sequence of complex
        All 2g+2 branch points.
    grid_r : array-like
        Real part coordinates (length W).
    grid_i : array-like
        Imaginary part coordinates (length H).
    dtype : np.dtype
        Output dtype (e.g. np.complex128).

    Returns
    -------
    omega_plus : np.ndarray of shape (g, H, W)
        omega_plus[k, iy, ix] = ω_k(z_iy,ix).
    """
    grid_r = np.asarray(grid_r)
    grid_i = np.asarray(grid_i)
    H, W = len(grid_i), len(grid_r)
    omega_fn = make_omega(branch_points)
    out = np.zeros((genus, H, W), dtype=dtype)
    for k in range(genus):
        for iy in range(H):
            for ix in range(W):
                z = complex(grid_r[ix], grid_i[iy])
                out[k, iy, ix] = omega_fn(k, z)
    return out


def build_omega_table_tqdm(
    genus,
    branch_points,
    grid_r,
    grid_i,
    dtype=np.complex128,
):
    """Same as build_omega_table with a progress bar."""
    grid_r = np.asarray(grid_r)
    grid_i = np.asarray(grid_i)
    H, W = len(grid_i), len(grid_r)
    omega_fn = make_omega(branch_points)
    out = np.zeros((genus, H, W), dtype=dtype)
    for k in tqdm(range(genus), desc="omega"):
        for iy in range(H):
            for ix in range(W):
                z = complex(grid_r[ix], grid_i[iy])
                out[k, iy, ix] = omega_fn(k, z)
    return out


def build_integral_table(
    genus,
    branch_points,
    grid_r,
    grid_i,
    base_point,
    dtype=np.complex128,
):
    """Compute I_k(z) = ∫_{base_point}^z ω_k on a 2D grid.

    Parameters
    ----------
    genus : int
        Number of integrals (g).
    branch_points : sequence of complex
        All 2g+2 branch points.
    grid_r, grid_i : array-like
        Grid coordinates.
    base_point : complex
        Base point for the Abel map.
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    I_plus : np.ndarray of shape (g, H, W)
        I_plus[k, iy, ix] = ∫_{base_point}^{z} ω_k.
    """
    grid_r = np.asarray(grid_r)
    grid_i = np.asarray(grid_i)
    H, W = len(grid_i), len(grid_r)
    omega_fn = make_omega(branch_points)
    out = np.zeros((genus, H, W), dtype=dtype)
    for k in range(genus):
        for iy in range(H):
            for ix in range(W):
                z = complex(grid_r[ix], grid_i[iy])
                out[k, iy, ix] = integrate_omega(omega_fn, k, z, base_point)
    return out


def build_integral_table_tqdm(
    genus,
    branch_points,
    grid_r,
    grid_i,
    base_point,
    dtype=np.complex128,
):
    """Same as build_integral_table with a progress bar (slow for large grids)."""
    grid_r = np.asarray(grid_r)
    grid_i = np.asarray(grid_i)
    H, W = len(grid_i), len(grid_r)
    omega_fn = make_omega(branch_points)
    out = np.zeros((genus, H, W), dtype=dtype)
    for k in tqdm(range(genus), desc="I"):
        for iy in tqdm(range(H), leave=False, desc=f"I[{k}]"):
            for ix in range(W):
                z = complex(grid_r[ix], grid_i[iy])
                out[k, iy, ix] = integrate_omega(omega_fn, k, z, base_point)
    return out
