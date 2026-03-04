"""
Inverse Abel-Jacobi map via Newton on θ(A(z) − u) = 0.

Uses theta evaluation from theta_functions (Riemann theta, log θ, gradients).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .theta_functions import (
    _ensure_complex_array,
    grad_log_theta,
    grad_riemann_theta,
    kleinian_p_matrix,
    log_theta,
    riemann_theta,
)


def inverse_abel_jacobi_newton(
    u: Union[np.ndarray, list],
    Omega: np.ndarray,
    A_fun: Callable[[complex], np.ndarray],
    omega_fun: Callable[[complex], np.ndarray],
    z0: complex,
    tol: float = 1e-10,
    max_iter: int = 50,
    use_log_theta: bool = True,
    eps_log: float = 1e-15,
    N_max: Optional[int] = None,
) -> Tuple[complex, bool, int]:
    """
    Find z on the curve such that A(z) = u by Newton on θ(A(z) − u) = 0
    (or on log θ for stability).

    Parameters
    ----------
    u : array-like, shape (g,), complex
        Target in the Jacobian.
    Omega : array-like, shape (g, g)
        Period matrix for theta.
    A_fun : callable z -> (g,) array
        Abel map from base point to z.
    omega_fun : callable z -> (g,) array
        Holomorphic differentials at z.
    z0 : complex
        Initial guess.
    tol, max_iter : convergence controls
    use_log_theta : use log θ and ∇log θ in the step (more stable).
    eps_log : regularization when |θ| is small.
    N_max : truncation for theta sum (None = auto).

    Returns
    -------
    z_sol : complex
    converged : bool
    num_iter : int
    """
    u = _ensure_complex_array(u, Omega.shape[0])
    g = len(u)

    for it in range(max_iter):
        A_z = A_fun(z0)
        w = A_z - u
        if use_log_theta:
            log_th = log_theta(w, Omega, N_max=N_max, eps=eps_log)
            grad_log = grad_log_theta(w, Omega, N_max=N_max, eps=eps_log)
            omega_z = omega_fun(z0)
            F_val = log_th
            F_prime = np.dot(grad_log, omega_z)
        else:
            th = riemann_theta(w, Omega, N_max=N_max)
            grad_th = grad_riemann_theta(w, Omega, N_max=N_max)
            omega_z = omega_fun(z0)
            F_val = th
            F_prime = np.dot(grad_th, omega_z)
        if np.abs(F_prime) < 1e-20:
            return z0, False, it + 1
        z_new = z0 - F_val / F_prime
        if np.abs(z_new - z0) < tol and np.abs(F_val) < tol:
            return z_new, True, it + 1
        z0 = z_new
    return z0, False, max_iter


def abel_map_vector(
    z: complex,
    integrate_omega: Callable[[int, complex], complex],
    g: int,
) -> np.ndarray:
    """
    Abel map A(z) = (∫_base^z ω_0, ..., ∫_base^z ω_{g-1}).

    Parameters
    ----------
    z : complex
    integrate_omega : (k, z) -> complex
    g : genus

    Returns
    -------
    ndarray, shape (g,), complex
    """
    return np.array([integrate_omega(k, z) for k in range(g)], dtype=np.complex128)


def omega_vector(
    z: complex,
    omega_k: Callable[[int, complex], complex],
    g: int,
) -> np.ndarray:
    """
    Vector of holomorphic differentials at z: (ω_0(z), ..., ω_{g-1}(z)).
    """
    return np.array([omega_k(k, z) for k in range(g)], dtype=np.complex128)


def inverse_abel_jacobi_via_kleinian_p(
    u: Union[np.ndarray, list],
    column_to_monic_coeffs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    omega: Optional[np.ndarray] = None,
    eta: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
    delta: Optional[Union[np.ndarray, list]] = None,
    column_index: int = 0,
    discriminant: Union[float, complex] = 1.0,
    n_max: Optional[int] = None,
    h: float = 1e-5,
    tau_star: Optional[np.ndarray] = None,
    log_sigma_fun: Optional[Callable[[np.ndarray], complex]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Invert AJ coordinates by:
      1) computing a column of Kleinian P_{i,j}(u),
      2) mapping that column to monic polynomial coefficients,
      3) root-finding to recover divisor x-coordinates.

    Parameters
    ----------
    u : array-like, shape (g,)
    column_to_monic_coeffs : callable (p_col, p_mat) -> coeffs
        Must return a 1D monic polynomial coefficient array for np.roots.
    omega, eta, tau, delta, discriminant, n_max, h, tau_star :
        Parameters for sigma/P evaluation (required unless log_sigma_fun provided).
    column_index : int
        Which P-matrix column to use.
    log_sigma_fun : callable or None
        Optional override for log sigma; useful for custom models/tests.

    Returns
    -------
    roots : np.ndarray
        Recovered divisor x-coordinates (polynomial roots).
    coeffs : np.ndarray
        Monic polynomial coefficients used for root-finding.
    p_mat : np.ndarray
        Full Kleinian P-matrix at u.
    """
    if log_sigma_fun is None:
        if omega is None or eta is None or tau is None or delta is None:
            raise ValueError("omega, eta, tau, delta are required unless log_sigma_fun is provided")
        g = np.asarray(omega).shape[0]
    else:
        g = np.atleast_1d(np.asarray(u)).size
    u = _ensure_complex_array(u, g)

    p_mat = kleinian_p_matrix(
        u=u,
        omega=omega,
        eta=eta,
        tau=tau,
        delta=delta,
        discriminant=discriminant,
        n_max=n_max,
        h=h,
        tau_star=tau_star,
        log_sigma_fun=log_sigma_fun,
    )
    j = int(column_index)
    if j < 0 or j >= p_mat.shape[1]:
        raise ValueError(f"column_index must be in [0, {p_mat.shape[1] - 1}]")
    p_col = p_mat[:, j]

    coeffs = np.asarray(column_to_monic_coeffs(p_col, p_mat), dtype=np.complex128).ravel()
    if coeffs.size < 2:
        raise ValueError("column_to_monic_coeffs must return at least degree-1 polynomial")
    if np.abs(coeffs[0]) == 0:
        raise ValueError("polynomial must be monic/nonzero leading coefficient")
    if not np.isclose(coeffs[0], 1.0 + 0.0j):
        coeffs = coeffs / coeffs[0]
    roots = np.roots(coeffs)
    return roots, coeffs, p_mat
