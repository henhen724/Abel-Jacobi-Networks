import numpy as np
import pytest

from aj.classical import riemann_theta_function

try:
    import mpmath as mp

    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _theta_g1(delta, z, tau, n_max=10):
    tau_mat = np.array([[tau]], dtype=np.complex128)
    u = np.array([z], dtype=np.complex128)
    d = np.array(delta, dtype=np.complex128)
    return riemann_theta_function(d, u, tau_mat, n_max=n_max)


def test_g1_theta3_zero_matches_known_value():
    # theta_3(0, exp(-pi)) = sum_n exp(-pi n^2)
    val = _theta_g1([0.0, 0.0], 0.0, 1j, n_max=12)
    expected = 1.0864348112133080146
    assert abs(val - expected) < 1e-12


def test_g1_theta2_zero_matches_known_value():
    # theta_2(0, exp(-pi)) = sum_n exp(-pi (n+1/2)^2)
    val = _theta_g1([0.5, 0.0], 0.0, 1j, n_max=12)
    expected = 0.9135791381561168214
    assert abs(val - expected) < 1e-12


def test_g1_theta4_zero_matches_known_value():
    # theta_4(0, exp(-pi)) = sum_n (-1)^n exp(-pi n^2)
    val = _theta_g1([0.0, 0.5], 0.0, 1j, n_max=12)
    expected = 0.9135791381561168214
    assert abs(val - expected) < 1e-12


@pytest.mark.parametrize(
    "delta, z, tau, m_int, n_int",
    [
        ([0.0, 0.0], 0.21 + 0.13j, 0.4 + 1.2j, 1, -1),
        ([0.5, 0.0], -0.3 + 0.2j, 0.2 + 1.4j, -2, 1),
        ([0.0, 0.5], 0.1 - 0.15j, 0.1 + 1.0j, 2, 2),
    ],
)
def test_g1_functional_equation(delta, z, tau, m_int, n_int):
    # theta[eps,eps'](z + tau*n + m) =
    # exp(-pi i n^2 tau - 2pi i n (z + eps') + 2pi i eps m) theta[eps,eps'](z)
    eps, eps_p = delta
    lhs = _theta_g1(delta, z + tau * n_int + m_int, tau, n_max=12)
    rhs = np.exp(
        -1j * np.pi * (n_int ** 2) * tau
        - 2j * np.pi * n_int * (z + eps_p)
        + 2j * np.pi * eps * m_int
    ) * _theta_g1(delta, z, tau, n_max=12)
    assert abs(lhs - rhs) < 1e-10


def test_torch_autograd_u_and_explicit_derivatives():
    if not HAS_TORCH:
        pytest.skip("torch not installed")

    tau = torch.tensor([[0.3 + 1.1j]], dtype=torch.complex128)
    delta = torch.tensor([0.5, 0.0], dtype=torch.complex128)
    u = torch.tensor([0.2 + 0.1j], dtype=torch.complex128, requires_grad=True)

    theta = riemann_theta_function(delta, u, tau, n_max=10)
    loss = torch.real(theta * torch.conj(theta))
    loss.backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()

    theta2, grad, hess = riemann_theta_function(
        delta, u.detach(), tau, n_max=10, return_derivatives=True
    )
    assert torch.isfinite(theta2).all()
    assert torch.isfinite(grad).all()
    assert torch.isfinite(hess).all()
