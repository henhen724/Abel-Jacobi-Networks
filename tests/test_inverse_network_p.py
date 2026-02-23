import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aj.classical.inverse_network import InverseAbelJacobiNetwork


def _toy_log_sigma(v: np.ndarray) -> complex:
    v = np.asarray(v, dtype=np.complex128)
    # Gives P[:,0] = [v0, v1] approximately under finite differences:
    # logσ = -(v0^3)/6 - (v0*v1^2)/2
    return -(v[0] ** 3) / 6.0 - 0.5 * v[0] * (v[1] ** 2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_inverse_network_uses_p_and_divisor_points_are_trainable():
    g = 2
    branch_points = torch.tensor(
        [
            [-1.5, -0.5],
            [-1.5, 0.5],
            [0.0, -0.8],
            [0.0, 0.8],
            [1.5, -0.5],
            [1.5, 0.5],
        ],
        dtype=torch.float32,
    )
    init_divisor_points = torch.zeros(g, 2, dtype=torch.float32)
    Omega_init = np.array(
        [[1.2 + 2.0j, 0.3 + 0.5j], [0.3 + 0.5j, 1.0 + 1.8j]],
        dtype=np.complex128,
    )
    K_init = np.zeros(g, dtype=np.complex128)

    net = InverseAbelJacobiNetwork(
        genus=g,
        branch_points=branch_points,
        base_point=(-3.0, -3.0),
        init_divisor_points=init_divisor_points,
        use_kleinian_p=True,
        log_sigma_fun=_toy_log_sigma,
        Omega_init=Omega_init,
        K_init=K_init,
    )

    # Build u from known symmetric values: u=(s1,s2)
    roots_true = np.array([0.8 + 0.1j, -0.35 + 0.2j], dtype=np.complex128)
    s1 = roots_true[0] + roots_true[1]
    s2 = roots_true[0] * roots_true[1]
    u = torch.tensor(
        [[[float(np.real(s1)), float(np.imag(s1))], [float(np.real(s2)), float(np.imag(s2))]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    out = net(u)  # (B,g) real coefficients
    assert out.shape == (1, g)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert net.divisor_points.grad is not None
    assert torch.isfinite(net.divisor_points.grad).all()
