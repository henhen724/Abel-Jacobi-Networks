"""
Theta function evaluation and Abel-Jacobi inverse via Newton's method on log Klein (Riemann) theta.

For a period matrix Ω (g×g, Im Ω > 0), the Riemann theta function is
  θ(z, Ω) = Σ_{n ∈ ℤ^g} exp(πi nᵀΩn + 2πi nᵀz).
Klein theta in this context refers to the same Riemann theta (or with a fixed characteristic).
Used to invert the Abel-Jacobi map: find z on the curve such that A(z) = u by solving
θ(A(z) - u) = 0 via Newton, optionally in log form for stability.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple, Union


def _ensure_complex_array(z: np.ndarray, g: int) -> np.ndarray:
    """Ensure z is a 1d complex array of length g."""
    z = np.atleast_1d(np.asarray(z, dtype=np.complex128)).ravel()
    if z.size != g:
        raise ValueError(f"z must have length g={g}, got {z.size}")
    return z


def _default_N_max(Omega: np.ndarray, safety: float = 2.0) -> int:
    """Default truncation radius for the lattice sum from Im(Ω)."""
    ImO = np.imag(Omega)
    lam_min = np.min(np.linalg.eigvalsh(ImO))
    if lam_min <= 0:
        raise ValueError("Im(Ω) must be positive definite")
    # Bound n^T Im(Ω) n >= lam_min |n|^2; we want exp(-π lam_min |n|^2) small.
    # Use |n|_inf <= N => |n|^2 <= g*N^2; take N so that π*lam_min*g*N^2 >= safety*10 => N ~ sqrt(10*safety/(π*lam_min*g))
    from math import sqrt, pi
    N = max(3, int(np.ceil(sqrt(10 * safety / (np.pi * lam_min * ImO.shape[0])))))
    return N


def riemann_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
) -> np.complex128:
    """
    Evaluate the Riemann theta function θ(z, Ω).

    θ(z, Ω) = Σ_{n ∈ ℤ^g} exp(πi nᵀΩn + 2πi nᵀz).

    Parameters
    ----------
    z : array-like, shape (g,), complex
        Argument.
    Omega : array-like, shape (g, g), complex
        Period matrix (symmetric, Im(Ω) positive definite).
    N_max : int or None
        Truncation: |n_j| ≤ N_max. If None, chosen from Im(Ω).

    Returns
    -------
    complex
        θ(z, Ω).
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    g = Omega.shape[0]
    if Omega.shape != (g, g):
        raise ValueError("Omega must be g×g")
    z = _ensure_complex_array(z, g)
    if N_max is None:
        N_max = _default_N_max(Omega)

    # Lattice sum over n in [-N_max, N_max]^g
    grids = [np.arange(-N_max, N_max + 1, dtype=np.float64) for _ in range(g)]
    nn = np.meshgrid(*grids, indexing="ij")
    n = np.stack([nn[j].ravel() for j in range(g)], axis=1)  # (M, g)
    # n^T Omega n: (M, g) @ (g, g) @ (g, M) -> (M,)
    nO = n @ Omega
    nOn = np.sum(nO * n, axis=1)
    nz = n @ z
    terms = np.exp(np.pi * 1j * nOn + 2 * np.pi * 1j * nz)
    return np.sum(terms)


def grad_riemann_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
) -> np.ndarray:
    """
    Gradient of Riemann theta w.r.t. z: ∇θ = (∂θ/∂z_1, ..., ∂θ/∂z_g).

    ∂θ/∂z_j = 2πi Σ_n n_j exp(πi nᵀΩn + 2πi nᵀz).

    Returns
    -------
    ndarray, shape (g,), complex
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    g = Omega.shape[0]
    z = _ensure_complex_array(z, g)
    if N_max is None:
        N_max = _default_N_max(Omega)

    grids = [np.arange(-N_max, N_max + 1, dtype=np.float64) for _ in range(g)]
    nn = np.meshgrid(*grids, indexing="ij")
    n = np.stack([nn[j].ravel() for j in range(g)], axis=1)
    nO = n @ Omega
    nOn = np.sum(nO * n, axis=1)
    nz = n @ z
    base = np.exp(np.pi * 1j * nOn + 2 * np.pi * 1j * nz)
    grad = np.zeros(g, dtype=np.complex128)
    for j in range(g):
        grad[j] = 2 * np.pi * 1j * np.sum(n[:, j] * base)
    return grad


def log_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
    eps: float = 1e-15,
) -> np.complex128:
    """
    Log of Riemann theta: log θ(z, Ω). Uses principal branch; near zeros of θ
    the value is regularized so that |θ| is clipped to eps (log is then log(eps)).

    Parameters
    ----------
    z, Omega, N_max : as in riemann_theta
    eps : float
        When |θ(z)| < eps, return log(eps) (or log(eps) + i*arg(θ)) to avoid singularity.

    Returns
    -------
    complex
        log θ(z, Ω), with regularization when |θ| < eps.
    """
    th = riemann_theta(z, Omega, N_max)
    if np.abs(th) < eps:
        return np.log(eps) + 1j * np.angle(th)
    return np.log(th)


def grad_log_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Gradient of log θ: ∇log θ = (1/θ) ∇θ. Regularized when |θ| < eps.

    Returns
    -------
    ndarray, shape (g,), complex
    """
    th = riemann_theta(z, Omega, N_max)
    g_th = grad_riemann_theta(z, Omega, N_max)
    if np.abs(th) < eps:
        return g_th / (eps + 1j * np.imag(th))
    return g_th / th


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
    Find z on the curve such that the Abel-Jacobi image A(z) equals u, by Newton's
    method on the equation θ(A(z) - u) = 0 (or equivalently on log θ for stability).

    The Abel map A(z) is the vector of integrals from a fixed base point to z of
    the holomorphic differentials; omega_fun(z) returns the vector of those differentials
    evaluated at z. So dA/dz = omega_fun(z).

    Newton update: we solve F(z) = 0 with F(z) = θ(A(z) - u). Then
    F'(z) = ∇θ(A(z)-u) · (dA/dz) = ∇θ(A(z)-u) · omega_fun(z)  (scalar in 1-complex variable).
    So z_new = z - F(z) / F'(z).

    When use_log_theta is True, we use G(z) = log θ(A(z)-u) and update
    z_new = z - G(z) / G'(z) = z - (log θ) / (∇log θ · omega), which is equivalent
    to the same Newton step for θ = 0 (since d/dz log θ = (1/θ)(dθ/dz)).

    Parameters
    ----------
    u : array-like, shape (g,), complex
        Target point in the Jacobian (in coordinates given by the period lattice).
    Omega : array-like, shape (g, g)
        Period matrix for theta.
    A_fun : callable
        A_fun(z) -> array of shape (g,) = Abel map from base point to z.
    omega_fun : callable
        omega_fun(z) -> array of shape (g,) = holomorphic differentials at z.
    z0 : complex
        Initial guess (on or near the curve).
    tol, max_iter : convergence controls
    use_log_theta : if True, use log θ and its gradient in the step (more stable).
    eps_log : regularization for log theta when |θ| is small.
    N_max : truncation for theta sum (None = auto).

    Returns
    -------
    z_sol : complex
        Approximate point on the curve with A(z_sol) ≈ u.
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
    Build the Abel map vector A(z) = (∫_base^z ω_0, ..., ∫_base^z ω_{g-1}) using
    a callback that computes the k-th integral.

    Parameters
    ----------
    z : complex
        Point on the curve (in the complex x-plane).
    integrate_omega : callable (k, z) -> complex
        Returns the integral of ω_k from base to z.
    g : int
        Genus (number of components).

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
    Build the vector of holomorphic differentials at z: (ω_0(z), ..., ω_{g-1}(z)).

    Parameters
    ----------
    z : complex
        Point on the curve.
    omega_k : callable (k, t) -> complex
        Returns ω_k(t) for k = 0..g-1.
    g : int
        Genus.

    Returns
    -------
    ndarray, shape (g,), complex
    """
    return np.array([omega_k(k, z) for k in range(g)], dtype=np.complex128)
