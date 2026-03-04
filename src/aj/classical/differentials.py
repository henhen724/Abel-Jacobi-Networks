"""
Holomorphic differentials ω_k and their line integrals on a hyperelliptic curve.

Uses mpmath for arbitrary precision. ω_k(t) = t^k / sqrt(prod(t - a_i)).
"""


def make_omega(branch_points):
    """Return a function omega(k, t) for the k-th holomorphic differential.

    ω_k(t) = t^k / sqrt(prod(t - a_i)) where a_i are the branch points.

    Parameters
    ----------
    branch_points : sequence of complex
        All 2g+2 branch points (flattened from g+1 cuts).

    Returns
    -------
    omega_k : callable
        (k, t) -> ω_k(t) with t complex, k int.
    """
    pts = list(branch_points)

    def omega_k(k, t):
        import mpmath as mp
        prod = mp.mpf(1)
        for a in pts:
            prod *= (t - a)
        return t**k / mp.sqrt(prod)

    return omega_k


def integrate_omega(omega_fn, k, z, base_point):
    """Integrate ω_k from base_point to z.

    Parameters
    ----------
    omega_fn : callable
        From make_omega(branch_points).
    k : int
        Index of the differential.
    z : complex
        Upper endpoint.
    base_point : complex
        Lower endpoint (fixed base on the curve).

    Returns
    -------
    complex
        Line integral ∫_{base_point}^z ω_k.
    """
    try:
        import mpmath as mp
        result = mp.quad(lambda t: omega_fn(k, t), [base_point, z])
        return complex(result)
    except Exception:
        eps = 1e-12 + 1e-12j
        import mpmath as mp
        return complex(mp.quad(lambda t: omega_fn(k, t), [base_point, z + eps]))
