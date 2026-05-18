# Classical Abel-Jacobi: hyperelliptic curves, period tables, and forward map.

from .cuts import make_hyperelliptic_cuts
from .differentials import make_omega, integrate_omega
from .tables import build_omega_table, build_integral_table
from .abel_jacobi_map import abel_jacobi_forward, compute_aj_normalization
from .period_matrix import period_matrix_from_cycles
from .theta_functions import (
    riemann_theta,
    grad_riemann_theta,
    log_theta,
    grad_log_theta,
    riemann_theta_function,
    kleinian_sigma,
    log_kleinian_sigma,
    kleinian_p_matrix,
    kleinian_p_column,
)
from .inverse_abel_jacobi_map import (
    inverse_abel_jacobi_newton,
    inverse_abel_jacobi_via_kleinian_p,
    abel_map_vector,
    omega_vector,
)
from .inverse_network import (
    InverseAbelJacobiNetwork,
    compute_period_matrix_hyperelliptic,
    make_omega_xn_dx,
    integrate_omega_xn_dx,
)

__all__ = [
    "make_hyperelliptic_cuts",
    "make_omega",
    "integrate_omega",
    "build_omega_table",
    "build_integral_table",
    "abel_jacobi_forward",
    "compute_aj_normalization",
    "AJGridActivationNorm",
    "pack_complex_table",
    "period_matrix_from_cycles",
    "riemann_theta",
    "grad_riemann_theta",
    "log_theta",
    "grad_log_theta",
    "riemann_theta_function",
    "kleinian_sigma",
    "log_kleinian_sigma",
    "kleinian_p_matrix",
    "kleinian_p_column",
    "inverse_abel_jacobi_newton",
    "inverse_abel_jacobi_via_kleinian_p",
    "abel_map_vector",
    "omega_vector",
    "InverseAbelJacobiNetwork",
    "compute_period_matrix_hyperelliptic",
    "make_omega_xn_dx",
    "integrate_omega_xn_dx",
]

# Torch-only helpers (lazy import so `import aj.classical` works without torch).
_LAZY_TORCH_EXPORTS = {
    "AJGridActivationNorm": ("grid_activation", "AJGridActivationNorm"),
    "pack_complex_table": ("grid_activation", "pack_complex_table"),
}


def __getattr__(name: str):
    if name in _LAZY_TORCH_EXPORTS:
        mod_name, attr = _LAZY_TORCH_EXPORTS[name]
        from importlib import import_module

        mod = import_module(f".{mod_name}", __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
