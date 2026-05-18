"""
Inverse Abel-Jacobi network: uses inverse Abel-Jacobi map as forward pass.

The network takes Jacobian coordinates u ∈ C^g and outputs coefficients of a
symmetric polynomial in the x-coordinates of points on the curve whose Abel-Jacobi
image is u (or close to u).

Parameters:
- Branch points (zeros of the hyperelliptic polynomial)
- Base point for Abel map
- Coefficients for symmetric polynomial (learnable)

Precomputes:
- Period matrix Ω (g×g)
- Riemann constant K (g-vector)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import mpmath as mp
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

from .theta_functions import (
    riemann_theta,
    grad_riemann_theta,
    log_theta,
    grad_log_theta,
)
from .inverse_abel_jacobi_map import (
    inverse_abel_jacobi_newton,
    inverse_abel_jacobi_via_kleinian_p,
)


def _to_mpc(z):
    """Convert numpy/Python complex to mpmath mpc. Avoids TypeError: cannot create mpf from complex."""
    if hasattr(mp, "mpc") and isinstance(z, (mp.mpc, mp.mpf)):
        return z if isinstance(z, mp.mpc) else mp.mpc(z)
    c = complex(z)
    return mp.mpc(c.real, c.imag)


def _to_mpf(r):
    """Convert numpy/Python real to mpmath mpf."""
    if hasattr(mp, "mpf") and isinstance(r, (mp.mpf, mp.mpc)):
        return mp.mpf(r) if isinstance(r, mp.mpf) else mp.mpf(r.real)
    return mp.mpf(float(r))


def make_omega_xn_dx(branch_points):
    """
    Holomorphic differentials ω_k = x^k dx / y where y^2 = prod(x - a_i).

    For hyperelliptic curve y^2 = prod_{i=1}^{2g+2} (x - a_i), the canonical
    basis is ω_k = x^k dx / y for k = 0, ..., g-1.

    Parameters
    ----------
    branch_points : array-like, shape (2g+2,), complex
        All branch points a_i.

    Returns
    -------
    omega_k : callable
        (k, x) -> ω_k(x) = x^k / sqrt(prod(x - a_i))
    """
    if not HAS_MPMATH:
        raise ImportError("mpmath required for differentials")
    pts = [_to_mpc(a) for a in branch_points]

    def omega_k(k, x):
        x = _to_mpc(x)
        prod = mp.mpf(1)
        for a in pts:
            prod *= (x - a)
        return x**k / mp.sqrt(prod)

    return omega_k


def integrate_omega_xn_dx(omega_fn, k, z, base_point):
    """Integrate ω_k = x^k dx / y from base_point to z."""
    if not HAS_MPMATH:
        raise ImportError("mpmath required")
    base_mp = _to_mpc(base_point)
    z_mp = _to_mpc(z)
    try:
        result = mp.quad(lambda t: omega_fn(k, t), [base_mp, z_mp])
        return complex(result)
    except Exception:
        eps = _to_mpc(1e-12 + 1e-12j)
        return complex(mp.quad(lambda t: omega_fn(k, t), [base_mp, z_mp + eps]))


def compute_period_matrix_hyperelliptic(
    branch_points: np.ndarray,
    base_point: complex,
    genus: int,
    cycles_a: Optional[list] = None,
    cycles_b: Optional[list] = None,
    mp_dps: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the period matrix Ω (g×g) and Riemann constant K (g-vector) for a
    hyperelliptic curve.

    For a hyperelliptic curve y^2 = prod(x - a_i), we use a standard symplectic
    basis: a-cycles around each cut [a_{2i}, a_{2i+1}], and b-cycles connecting
    cuts. The period matrix is Ω = (Ω_1)^{-1} Ω_2 where Ω_1 (resp. Ω_2) are
    the a-periods (resp. b-periods).

    Parameters
    ----------
    branch_points : array-like, shape (2g+2,), complex
        All branch points.
    base_point : complex
        Base point for Abel map (not used for periods, but needed for consistency).
    genus : int
        Genus of the curve.
    cycles_a, cycles_b : list of paths, optional
        If None, uses default hyperelliptic cycles (contours around cuts).
    mp_dps : int
        mpmath precision.

    Returns
    -------
    Omega : np.ndarray, shape (g, g), complex
        Period matrix (symmetric, Im(Ω) > 0).
    K : np.ndarray, shape (g,), complex
        Riemann constant vector.
    """
    if not HAS_MPMATH:
        raise ImportError("mpmath required for period matrix computation")
    mp.mp.dps = mp_dps

    omega_fn = make_omega_xn_dx(branch_points)
    branch_pts = np.asarray(branch_points)

    # Default cycles: a-cycles around cuts, b-cycles connecting them
    if cycles_a is None or cycles_b is None:
        # For hyperelliptic curves, standard basis:
        # a_i: small loop around cut [a_{2i}, a_{2i+1}]
        # b_i: path from cut i to cut i+1 (or similar)
        cycles_a = []
        cycles_b = []
        for i in range(genus):
            a_start, a_end = branch_pts[2*i], branch_pts[2*i+1]
            center_a = (a_start + a_end) / 2
            radius_a = 0.6 * abs(a_start - a_end) / 2
            cycles_a.append(("circle", _to_mpc(center_a), _to_mpf(radius_a)))

            if i < genus - 1:
                b_start = (branch_pts[2*i] + branch_pts[2*i+1]) / 2
                b_end = (branch_pts[2*(i+1)] + branch_pts[2*(i+1)+1]) / 2
                cycles_b.append(("line", _to_mpc(b_start), _to_mpc(b_end)))
            else:
                b_start = (branch_pts[2*i] + branch_pts[2*i+1]) / 2
                b_end = base_point + 10.0
                cycles_b.append(("line", _to_mpc(b_start), _to_mpc(b_end)))

    # Compute a-periods: Ω_1 (g×g)
    Omega_1 = np.zeros((genus, genus), dtype=np.complex128)
    for i, cycle_a in enumerate(cycles_a):
        if cycle_a[0] == "circle":
            _, center, radius = cycle_a
            for k in range(genus):
                def integrand(phi):
                    # Use only mpmath types so quad does not try mpf(complex)
                    t = center + radius * mp.exp(mp.mpc(0, phi))
                    j_mp = mp.mpc(0, 1)
                    return radius * j_mp * mp.exp(mp.mpc(0, phi)) * omega_fn(k, t)
                Omega_1[k, i] = complex(mp.quad(integrand, [0, 2*mp.pi], maxdegree=20))
        else:
            raise NotImplementedError("Only circular a-cycles implemented")

    # Compute b-periods: Ω_2 (g×g)
    Omega_2 = np.zeros((genus, genus), dtype=np.complex128)
    for i, cycle_b in enumerate(cycles_b):
        if cycle_b[0] == "line":
            _, start, end = cycle_b
            for k in range(genus):
                Omega_2[k, i] = integrate_omega_xn_dx(omega_fn, k, end, start)
        else:
            raise NotImplementedError("Only linear b-cycles implemented")

    # Period matrix: Ω = (Ω_1)^{-1} Ω_2
    Omega = np.linalg.solve(Omega_1, Omega_2)

    # Riemann constant: K = (1/2) diag(Ω) + sum of half-periods
    # For hyperelliptic curves, K = (1/2) * sum of odd-indexed branch points
    # (in Abel-Jacobi coordinates). More precisely:
    # K_j = (1/2) * sum_{i odd} ∫_{base}^{a_i} ω_j
    K = np.zeros(genus, dtype=np.complex128)
    for j in range(genus):
        # Sum over odd-indexed branch points (a_1, a_3, ..., a_{2g+1})
        for i in range(1, 2*genus+2, 2):
            K[j] += integrate_omega_xn_dx(omega_fn, j, branch_pts[i], base_point)
        K[j] *= 0.5

    return Omega, K


class InverseAbelJacobiNetwork(nn.Module):
    """
    Neural network that uses inverse Abel-Jacobi map as forward pass.

    Input: u ∈ C^g (Jacobian coordinates)
    Output: coefficients of symmetric polynomial in x-coordinates

    The symmetric polynomial P(x) = x^g - σ_1 x^{g-1} + σ_2 x^{g-2} - ... + (-1)^g σ_g
    represents a divisor of degree g. The network learns the σ_k coefficients.

    Parameters
    ----------
    genus : int
        Genus of the curve.
    branch_points : torch.Tensor, shape (2g+2, 2) or (2g+2,)
        Branch points (zeros). If shape (2g+2, 2), interpreted as (real, imag).
        If shape (2g+2,), interpreted as complex (requires complex dtype).
    base_point : complex or tuple (real, imag)
        Base point for Abel map.
    init_coeffs : torch.Tensor, optional, shape (g,)
        Initial values for symmetric polynomial coefficients σ_k.
        If None, initialized to small random values.
    mp_dps : int
        mpmath precision for period matrix computation (only used in __init__).
    newton_tol : float
        Tolerance for Newton iteration in inverse Abel-Jacobi.
    newton_max_iter : int
        Maximum iterations for Newton.
    """

    def __init__(
        self,
        genus: int,
        branch_points: torch.Tensor,
        base_point: complex | Tuple[float, float],
        init_coeffs: Optional[torch.Tensor] = None,
        init_divisor_points: Optional[torch.Tensor] = None,
        use_kleinian_p: bool = True,
        p_column_index: int = 0,
        p_column_to_monic_coeffs=None,
        log_sigma_fun=None,
        Omega_init: Optional[np.ndarray] = None,
        K_init: Optional[np.ndarray] = None,
        mp_dps: int = 50,
        newton_tol: float = 1e-8,
        newton_max_iter: int = 50,
    ):
        super().__init__()
        self.genus = genus
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.use_kleinian_p = use_kleinian_p
        self.p_column_index = int(p_column_index)
        self._log_sigma_fun = log_sigma_fun

        # Convert branch_points to numpy complex array
        if branch_points.dim() == 2 and branch_points.shape[1] == 2:
            branch_pts_np = branch_points.detach().cpu().numpy()
            branch_pts_np = branch_pts_np[:, 0] + 1j * branch_pts_np[:, 1]
        elif branch_points.dim() == 1:
            branch_pts_np = branch_points.detach().cpu().numpy()
            if not np.iscomplexobj(branch_pts_np):
                raise ValueError("1D branch_points must be complex dtype")
        else:
            raise ValueError(f"branch_points shape {branch_points.shape} not supported")
        self.register_buffer("_branch_points_np", torch.from_numpy(branch_pts_np.real))
        self.register_buffer("_branch_points_np_imag", torch.from_numpy(branch_pts_np.imag))
        self._branch_points_complex = branch_pts_np

        # Base point
        if isinstance(base_point, tuple):
            base_point = complex(base_point[0], base_point[1])
        self.base_point = base_point

        # Precompute period matrix and Riemann constant.
        # Default cycle assumption follows standard cut ordering:
        # cuts are [a_{2j}, a_{2j+1}] in the order provided by branch_points.
        if Omega_init is None or K_init is None:
            Omega, K = compute_period_matrix_hyperelliptic(
                branch_pts_np, base_point, genus, mp_dps=mp_dps
            )
        else:
            Omega = np.asarray(Omega_init, dtype=np.complex128)
            K = np.asarray(K_init, dtype=np.complex128)
        self.register_buffer("Omega", torch.from_numpy(Omega.real))
        self.register_buffer("Omega_imag", torch.from_numpy(Omega.imag))
        self.register_buffer("K", torch.from_numpy(K.real))
        self.register_buffer("K_imag", torch.from_numpy(K.imag))
        self._Omega_complex = Omega
        self._K_complex = K

        # Learnable symmetric polynomial coefficients σ_1, ..., σ_g
        if init_coeffs is None:
            init_coeffs = torch.randn(genus, dtype=torch.float32) * 0.1
        self.coeffs = nn.Parameter(init_coeffs)

        # Learnable divisor point offsets (gradable for training).
        if init_divisor_points is None:
            init_divisor_points = torch.zeros(genus, 2, dtype=torch.float32)
        else:
            init_divisor_points = init_divisor_points.detach().clone().float()
            if init_divisor_points.shape != (genus, 2):
                raise ValueError("init_divisor_points must have shape (g, 2)")
        self.divisor_points = nn.Parameter(init_divisor_points)

        # Default map from P-column to monic polynomial coefficients.
        # Uses p_col[k-1] as sigma_k in x^g - sigma1 x^{g-1} + ... + (-1)^g sigmag.
        if p_column_to_monic_coeffs is None:
            def _default_p_to_monic(p_col, _p_mat):
                p_col = np.asarray(p_col, dtype=np.complex128).ravel()
                if p_col.size < self.genus:
                    raise ValueError("p_col must have at least g entries")
                coeffs = [1.0 + 0.0j]
                for k in range(1, self.genus + 1):
                    coeffs.append(((-1) ** k) * p_col[k - 1])
                return np.asarray(coeffs, dtype=np.complex128)
            self._p_column_to_monic_coeffs = _default_p_to_monic
        else:
            self._p_column_to_monic_coeffs = p_column_to_monic_coeffs

        # Cache for omega_fn and integrate_omega (computed once per forward)
        self._omega_fn = None
        self._integrate_omega_cache = {}

    def _get_omega_fn(self):
        """Get or create omega function."""
        if self._omega_fn is None:
            self._omega_fn = make_omega_xn_dx(self._branch_points_complex)
        return self._omega_fn

    def _abel_map(self, x: complex) -> np.ndarray:
        """Abel map A(x) = (∫_base^x ω_0, ..., ∫_base^x ω_{g-1})."""
        omega_fn = self._get_omega_fn()
        result = np.zeros(self.genus, dtype=np.complex128)
        for k in range(self.genus):
            result[k] = integrate_omega_xn_dx(omega_fn, k, x, self.base_point)
        return result

    def _omega_at(self, x: complex) -> np.ndarray:
        """Vector of differentials at x: (ω_0(x), ..., ω_{g-1}(x))."""
        omega_fn = self._get_omega_fn()
        result = np.zeros(self.genus, dtype=np.complex128)
        for k in range(self.genus):
            result[k] = complex(omega_fn(k, x))
        return result

    def _elementary_symmetric_polynomials(self, x_coords: np.ndarray) -> np.ndarray:
        """
        Compute elementary symmetric polynomials σ_1, ..., σ_g from x-coordinates.

        For roots x_1, ..., x_g, the polynomial is:
        P(x) = x^g - σ_1 x^{g-1} + σ_2 x^{g-2} - ... + (-1)^g σ_g

        where σ_k is the k-th elementary symmetric polynomial:
        σ_1 = sum_i x_i
        σ_2 = sum_{i<j} x_i x_j
        ...
        σ_g = x_1 x_2 ... x_g
        """
        g = len(x_coords)
        coeffs = np.zeros(g, dtype=np.complex128)
        from itertools import combinations
        for k in range(1, g + 1):
            # Sum of all products of k distinct elements
            coeffs[k-1] = sum(np.prod(x_coords[list(comb)]) for comb in combinations(range(g), k))
        return coeffs

    def _elementary_symmetric_polynomials_torch(self, roots: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma_1..sigma_g from roots via polynomial recurrence, differentiable.
        roots: (g,) complex tensor
        """
        g = roots.numel()
        poly = torch.zeros(g + 1, dtype=roots.dtype, device=roots.device)
        poly[0] = 1.0 + 0.0j
        for r in roots:
            prev = poly.clone()
            poly[1:] = prev[1:] - r * prev[:-1]
        sigma = torch.empty(g, dtype=roots.dtype, device=roots.device)
        for k in range(1, g + 1):
            sigma[k - 1] = ((-1) ** k) * poly[k]
        return sigma

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: inverse Abel-Jacobi map.

        For input u ∈ C^g (Jacobian coordinates), finds a divisor D = P_1 + ... + P_g
        such that AJ(D) ≈ u, then outputs coefficients of symmetric polynomial in
        x-coordinates of P_i.

        **Gradient Note**: This forward pass detaches `u` and uses numpy/mpmath
        for the inverse Abel-Jacobi computation, so gradients **cannot** flow back
        to `u`. Only `self.coeffs` receives gradients (via the final addition).
        See `inverse_network_gradients.md` for details.

        Parameters
        ----------
        u : torch.Tensor, shape (..., g, 2) or (..., g)
            Jacobian coordinates. If shape (..., g, 2), interpreted as (real, imag).
            If shape (..., g), must be complex dtype.

        Returns
        -------
        coeffs : torch.Tensor, shape (..., g)
            Coefficients σ_1, ..., σ_g of symmetric polynomial P(x) = x^g - σ_1 x^{g-1} + ...
        """
        # Convert u to numpy complex
        # NOTE: .detach() breaks gradient flow back to u; only self.coeffs gets gradients
        if u.shape[-1] == 2:
            u_np = u.detach().cpu().numpy()
            u_complex = u_np[..., 0] + 1j * u_np[..., 1]
        elif u.dtype.is_complex:
            u_complex = u.detach().cpu().numpy()
        else:
            raise ValueError(f"u shape {u.shape} and dtype {u.dtype} not supported")

        # Flatten batch dimensions
        orig_shape = u_complex.shape[:-1]
        u_flat = u_complex.reshape(-1, self.genus)
        batch_size = u_flat.shape[0]

        # For each u in batch, find g points on curve via inverse Abel-Jacobi.
        coeffs_list = []
        offset_complex = torch.complex(self.divisor_points[:, 0], self.divisor_points[:, 1])
        for i in range(batch_size):
            u_i = u_flat[i]
            # Adjust by Riemann constant: solve for divisor D with AJ(D) = u_i + K
            u_adj = u_i + self._K_complex
            if self.use_kleinian_p:
                roots_np, _, _ = inverse_abel_jacobi_via_kleinian_p(
                    u=u_adj,
                    column_to_monic_coeffs=self._p_column_to_monic_coeffs,
                    column_index=self.p_column_index,
                    log_sigma_fun=self._log_sigma_fun,
                )
            else:
                x_coords = []
                u_remaining = u_adj.copy()
                for j in range(self.genus):
                    if j == 0:
                        x0 = self.base_point + 0.1 * (1 + 1j)
                    else:
                        angle = 2 * np.pi * j / self.genus
                        x0 = self.base_point + 0.5 * np.exp(1j * angle)
                    target = u_remaining / max(1, self.genus - j)
                    x_sol, _, _ = inverse_abel_jacobi_newton(
                        target,
                        self._Omega_complex,
                        self._abel_map,
                        self._omega_at,
                        x0,
                        tol=self.newton_tol,
                        max_iter=self.newton_max_iter,
                        use_log_theta=True,
                    )
                    x_coords.append(x_sol)
                    u_remaining = u_remaining - self._abel_map(x_sol)
                roots_np = np.asarray(x_coords, dtype=np.complex128)

            roots_t = torch.as_tensor(roots_np, dtype=torch.complex64, device=self.coeffs.device)
            roots_t = roots_t + offset_complex.to(roots_t.dtype)
            sigma_t = self._elementary_symmetric_polynomials_torch(roots_t)
            coeffs_list.append(sigma_t.real.float())

        coeffs_out = torch.stack(coeffs_list, dim=0)  # (batch_size, g)

        # Add learned adjustment
        coeffs_out = coeffs_out + self.coeffs.unsqueeze(0)

        # Reshape to original batch shape
        coeffs_out = coeffs_out.reshape(*orig_shape, self.genus)
        return coeffs_out
