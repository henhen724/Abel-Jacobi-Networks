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
