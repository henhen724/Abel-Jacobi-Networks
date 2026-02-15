"""
Period matrix for a hyperelliptic curve.

The period matrix Ω is the g×2g matrix (Ω_1 | Ω_2) where columns are the
periods of the holomorphic differentials ω_0, ..., ω_{g-1} along a choice
of a- and b-cycles. The grid-based tables (omega_plus, I_plus) from
build_omega_table and build_integral_table give ω_k(z) and ∫_base^z ω_k
on a grid; they do not by themselves form the period matrix.

To compute Ω you would:
  1. Choose a symplectic basis of cycles {a_i, b_i} on the curve.
  2. For each k and each cycle γ, compute ∫_γ ω_k (e.g. via line integrals).
  3. Assemble into g×2g matrix.

This module provides a placeholder; full implementation would require
cycle paths and numerical integration along them (e.g. using mpmath.quad
over piecewise paths).
"""

def period_matrix_from_cycles(omega_fn, genus, cycles_a, cycles_b):
    """Compute the period matrix Ω = (Ω_1 | Ω_2) from cycle integrals.

    Placeholder: cycles_a and cycles_b should be lists of piecewise paths
    (each path a list of complex segments). Not fully implemented; raises
    if called with non-trivial cycles.

    Parameters
    ----------
    omega_fn : callable
        From make_omega(branch_points); (k, t) -> ω_k(t).
    genus : int
        Number of differentials.
    cycles_a : list of paths
        a-cycles (each path a sequence of (start, end) or similar).
    cycles_b : list of paths
        b-cycles.

    Returns
    -------
    Omega : np.ndarray of shape (g, 2*g)
        Period matrix (Ω_1 | Ω_2).
    """
    raise NotImplementedError(
        "Full period matrix from cycle integrals is not implemented. "
        "Use build_omega_table and build_integral_table for grid-based "
        "ω and Abel map values."
    )
