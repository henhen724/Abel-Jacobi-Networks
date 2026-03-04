"""
Riemann theta function and related theta utilities.

Uses the lattice-sum implementation with characteristic; the zero-characteristic
case gives the standard θ(z, Ω) = Σ_n exp(πi nᵀΩn + 2πi nᵀz). Supports both
numpy and (optionally) PyTorch for autograd in the argument z/u.
"""

from __future__ import annotations

from itertools import product
from math import sqrt
from typing import Callable, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _is_torch_tensor(x):
    return HAS_TORCH and isinstance(x, torch.Tensor)


def _stack(rows, use_torch):
    if use_torch:
        return torch.stack(rows, dim=0)
    return np.stack(rows, axis=0)


def _sum(x, axis=None):
    if _is_torch_tensor(x):
        return torch.sum(x, dim=axis)
    return np.sum(x, axis=axis)


def _ensure_complex_array(z: np.ndarray, g: int) -> np.ndarray:
    """Ensure z is a 1d complex array of length g."""
    z = np.atleast_1d(np.asarray(z, dtype=np.complex128)).ravel()
    if z.size != g:
        raise ValueError(f"z must have length g={g}, got {z.size}")
    return z


def _default_N_max(Omega: np.ndarray, safety: float = 2.0) -> int:
    """Default truncation radius for the lattice sum from Im(Ω).
    Uses a Gershgorin lower bound for the minimum eigenvalue to avoid np.linalg.eigvalsh,
    which can abort when PyTorch has already initialized MKL/BLAS.
    """
    ImO = np.asarray(np.imag(Omega), dtype=np.float64)
    g = ImO.shape[0]
    # Gershgorin: for symmetric A, λ_min >= min_i ( A[i,i] - sum_{j!=i} |A[i,j]| )
    radii = np.array([np.sum(np.abs(ImO[i, :])) - np.abs(ImO[i, i]) for i in range(g)])
    lam_min_bound = float(np.min(ImO.diagonal() - radii))
    if lam_min_bound <= 0:
        raise ValueError("Im(Ω) must be positive definite")
    N = max(3, int(np.ceil(sqrt(10 * safety / (np.pi * lam_min_bound * g)))))
    return N


def riemann_theta_function(
    delta,
    u,
    tau,
    tau_star=None,
    n_max=8,
    return_derivatives=False,
):
    """
    Truncated Riemann theta with characteristic.

    θ[δ](u | τ) = Σ_{m ∈ ℤ^g} exp(2πi (m+ε)ᵀ(u+ε') + πi (m+ε)ᵀ τ (m+ε))
    where δ = [ε, ε'] ∈ ℚ^{2g}.

    Parameters
    ----------
    delta : array-like, shape (2g,)
        Characteristic [epsilon, epsilon_prime].
    u : array-like, shape (g,)
        Argument in ℂ^g.
    tau : array-like, shape (g, g)
        Period matrix.
    tau_star : array-like, optional
        Assumed tau_star = (τ^{-1})^T. If provided, checked for consistency.
    n_max : int, default=8
        Truncation: m_j ∈ [-n_max, n_max].
    return_derivatives : bool, default=False
        If True, also return grad (dθ/du) and hess (d²θ/du_j du_k).

    Returns
    -------
    theta : complex scalar
    (optional) grad : shape (g,), hess : shape (g, g)
    """
    use_torch = _is_torch_tensor(u) or _is_torch_tensor(tau) or _is_torch_tensor(delta)

    if use_torch:
        if not _is_torch_tensor(u):
            u = torch.as_tensor(u, dtype=torch.complex128)
        if not _is_torch_tensor(tau):
            tau = torch.as_tensor(tau, dtype=torch.complex128, device=u.device)
        if not _is_torch_tensor(delta):
            delta = torch.as_tensor(delta, dtype=torch.complex128, device=u.device)
        if tau_star is not None and not _is_torch_tensor(tau_star):
            tau_star = torch.as_tensor(tau_star, dtype=torch.complex128, device=u.device)
    else:
        u = np.asarray(u, dtype=np.complex128)
        tau = np.asarray(tau, dtype=np.complex128)
        delta = np.asarray(delta, dtype=np.complex128)
        if tau_star is not None:
            tau_star = np.asarray(tau_star, dtype=np.complex128)

    if u.ndim != 1:
        raise ValueError("u must be a 1D vector of shape (g,)")
    g = u.shape[0]
    if tau.shape != (g, g):
        raise ValueError("tau must have shape (g, g) matching u")
    if delta.shape != (2 * g,):
        raise ValueError("delta must have shape (2g,)")
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    if tau_star is not None and tau_star.shape != (g, g):
        raise ValueError("tau_star must have shape (g, g)")

    eps = delta[:g]
    eps_prime = delta[g:]
    u_shifted = u + eps_prime

    rows = []
    for m in product(range(-n_max, n_max + 1), repeat=g):
        if use_torch:
            rows.append(torch.tensor(m, dtype=tau.real.dtype, device=u.device))
        else:
            rows.append(np.asarray(m, dtype=np.float64))
    m_grid = _stack(rows, use_torch)
    mp = m_grid + eps
    # Per-row matmul to avoid BLAS abort when torch has already initialized MKL (numpy single large matmul can crash)
    if use_torch:
        tau_mp = mp @ tau
    else:
        tau_mp = np.empty((mp.shape[0], tau.shape[1]), dtype=np.complex128)
        for i in range(mp.shape[0]):
            tau_mp[i] = mp[i] @ tau
    quad = _sum(tau_mp * mp, axis=1)
    linear = _sum(mp * u_shifted, axis=1)
    exponent = (1j * np.pi) * quad + (2j * np.pi) * linear
    terms = torch.exp(exponent) if use_torch else np.exp(exponent)
    theta = _sum(terms)

    if tau_star is not None:
        if use_torch:
            ident = torch.eye(g, dtype=tau.dtype, device=tau.device)
            approx = tau @ tau_star.transpose(-1, -2)
            if torch.max(torch.abs(approx - ident)).item() > 1e-5:
                raise ValueError("tau_star is not numerically close to (tau^{-1})^T")
        else:
            ident = np.eye(g, dtype=np.complex128)
            approx = tau @ tau_star.T
            if np.max(np.abs(approx - ident)) > 1e-8:
                raise ValueError("tau_star is not numerically close to (tau^{-1})^T")

    if not return_derivatives:
        return theta

    pref_1 = 2j * np.pi
    pref_2 = (2j * np.pi) ** 2
    weighted = terms[:, None] * mp
    grad = pref_1 * _sum(weighted, axis=0)
    hess_terms = terms[:, None, None] * (mp[:, :, None] * mp[:, None, :])
    hess = pref_2 * _sum(hess_terms, axis=0)
    return theta, grad, hess


def riemann_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
) -> np.complex128:
    """
    Riemann theta θ(z, Ω) with zero characteristic.

    θ(z, Ω) = Σ_{n ∈ ℤ^g} exp(πi nᵀΩn + 2πi nᵀz).

    Parameters
    ----------
    z : array-like, shape (g,), complex
    Omega : array-like, shape (g, g), complex
        Period matrix (symmetric, Im(Ω) > 0).
    N_max : int or None
        Truncation |n_j| ≤ N_max. If None, chosen from Im(Ω).

    Returns
    -------
    complex
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    g = Omega.shape[0]
    if Omega.shape != (g, g):
        raise ValueError("Omega must be g×g")
    z = _ensure_complex_array(z, g)
    if N_max is None:
        N_max = _default_N_max(Omega)
    delta = np.zeros(2 * g, dtype=np.complex128)
    out = riemann_theta_function(delta, z, Omega, n_max=N_max)
    return np.complex128(out) if isinstance(out, (np.floating, np.complexfloating)) else out


def grad_riemann_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
) -> np.ndarray:
    """
    Gradient of Riemann theta w.r.t. z: ∇θ = (∂θ/∂z_1, ..., ∂θ/∂z_g).
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    g = Omega.shape[0]
    z = _ensure_complex_array(z, g)
    if N_max is None:
        N_max = _default_N_max(Omega)
    delta = np.zeros(2 * g, dtype=np.complex128)
    _, grad, _ = riemann_theta_function(delta, z, Omega, n_max=N_max, return_derivatives=True)
    return np.asarray(grad, dtype=np.complex128)


def log_theta(
    z: Union[np.ndarray, list],
    Omega: np.ndarray,
    N_max: Optional[int] = None,
    eps: float = 1e-15,
) -> np.complex128:
    """
    Log of Riemann theta; regularized when |θ| < eps.
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
    """
    th = riemann_theta(z, Omega, N_max)
    g_th = grad_riemann_theta(z, Omega, N_max)
    if np.abs(th) < eps:
        return g_th / (eps + 1j * np.imag(th))
    return g_th / th


def kleinian_sigma(
    u: Union[np.ndarray, list],
    omega: np.ndarray,
    eta: np.ndarray,
    tau: np.ndarray,
    delta: Union[np.ndarray, list],
    discriminant: Union[float, complex] = 1.0,
    n_max: Optional[int] = None,
    tau_star: Optional[np.ndarray] = None,
) -> np.complex128:
    """
    Kleinian sigma via theta with characteristic:

      sigma(u) = Delta^{-1/8} * exp(1/2 u^T eta omega^{-1} u) *
                 theta[delta](omega^{-1} u | tau)
    """
    omega = np.asarray(omega, dtype=np.complex128)
    eta = np.asarray(eta, dtype=np.complex128)
    tau = np.asarray(tau, dtype=np.complex128)
    g = omega.shape[0]
    if omega.shape != (g, g) or eta.shape != (g, g) or tau.shape != (g, g):
        raise ValueError("omega, eta, tau must all be shape (g, g)")
    u = _ensure_complex_array(u, g)
    delta = np.asarray(delta, dtype=np.complex128).ravel()
    if delta.shape != (2 * g,):
        raise ValueError("delta must have shape (2g,)")
    if n_max is None:
        n_max = _default_N_max(tau)

    omega_inv = np.linalg.inv(omega)
    v = omega_inv @ u
    quad = 0.5 * np.dot(u, eta @ (omega_inv @ u))
    theta_val = riemann_theta_function(
        delta=delta,
        u=v,
        tau=tau,
        tau_star=tau_star,
        n_max=n_max,
        return_derivatives=False,
    )
    pref = np.power(np.complex128(discriminant), -1.0 / 8.0)
    return np.complex128(pref * np.exp(quad) * theta_val)


def log_kleinian_sigma(
    u: Union[np.ndarray, list],
    omega: np.ndarray,
    eta: np.ndarray,
    tau: np.ndarray,
    delta: Union[np.ndarray, list],
    discriminant: Union[float, complex] = 1.0,
    n_max: Optional[int] = None,
    tau_star: Optional[np.ndarray] = None,
    eps: float = 1e-30,
) -> np.complex128:
    """Principal-branch log of Kleinian sigma with small-magnitude regularization."""
    s = kleinian_sigma(
        u=u,
        omega=omega,
        eta=eta,
        tau=tau,
        delta=delta,
        discriminant=discriminant,
        n_max=n_max,
        tau_star=tau_star,
    )
    if np.abs(s) < eps:
        return np.log(eps) + 1j * np.angle(s)
    return np.log(s)


def kleinian_p_matrix(
    u: Union[np.ndarray, list],
    omega: Optional[np.ndarray] = None,
    eta: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
    delta: Optional[Union[np.ndarray, list]] = None,
    discriminant: Union[float, complex] = 1.0,
    n_max: Optional[int] = None,
    h: float = 1e-5,
    tau_star: Optional[np.ndarray] = None,
    log_sigma_fun: Optional[Callable[[np.ndarray], complex]] = None,
) -> np.ndarray:
    """
    Kleinian P-matrix from second derivatives:
      P_{i,j}(u) = - d_i d_j log sigma(u)

    Uses central finite differences in u. If log_sigma_fun is provided, it is
    used directly; otherwise log_kleinian_sigma(...) is used.
    """
    if log_sigma_fun is None:
        if omega is None or eta is None or tau is None or delta is None:
            raise ValueError("omega, eta, tau, delta are required unless log_sigma_fun is provided")
        g = np.asarray(omega).shape[0]
    else:
        g = np.atleast_1d(np.asarray(u)).size
    u = _ensure_complex_array(u, g)
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive")

    if log_sigma_fun is None:
        def f(vec):
            return log_kleinian_sigma(
                u=vec,
                omega=omega,
                eta=eta,
                tau=tau,
                delta=delta,
                discriminant=discriminant,
                n_max=n_max,
                tau_star=tau_star,
            )
    else:
        f = log_sigma_fun

    p = np.zeros((g, g), dtype=np.complex128)
    f0 = f(u)
    eye = np.eye(g, dtype=np.complex128)
    for i in range(g):
        ei = eye[i]
        f_p = f(u + h * ei)
        f_m = f(u - h * ei)
        p[i, i] = -((f_p - 2.0 * f0 + f_m) / (h * h))
        for j in range(i + 1, g):
            ej = eye[j]
            f_pp = f(u + h * ei + h * ej)
            f_pm = f(u + h * ei - h * ej)
            f_mp = f(u - h * ei + h * ej)
            f_mm = f(u - h * ei - h * ej)
            hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            p[i, j] = -hij
            p[j, i] = -hij
    return p


def kleinian_p_column(
    u: Union[np.ndarray, list],
    column_index: int,
    **kwargs,
) -> np.ndarray:
    """Return one column of the Kleinian P-matrix."""
    p = kleinian_p_matrix(u=u, **kwargs)
    g = p.shape[0]
    j = int(column_index)
    if j < 0 or j >= g:
        raise ValueError(f"column_index must be in [0, {g - 1}]")
    return p[:, j]
