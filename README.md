# Abel–Jacobi Networks

Research code for **tropical** and **classical** Abel–Jacobi maps on hyperelliptic curves, with applications to metric graphs, period integrals, and neural network training.

## Overview

This repository provides:

- **`aj` library** — A coherent Python package to compute period tables (ω and Abel–Jacobi integrals on a grid), run the **classical Abel–Jacobi forward pass** (divisor → Jacobian coordinates via lookup), and run the **tropical Abel–Jacobi forward pass** (metric graph + divisor → Jacobian coordinates mod cycle lengths).
- **Notebooks** — Original notebooks for exploration and training; they can import from `aj` or use inlined code.

## Library usage

Install in development mode (from repo root):

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate aj
pip install -e .

# Or pip + venv
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
pip install -e .
```

Verify the environment:

```bash
python scripts/check_aj_env.py
```

### Tropical Abel–Jacobi

```python
import numpy as np
from aj import build_chain_of_loops, cycle_data, tropical_abel_jacobi_forward

graph, positions = build_chain_of_loops(genus=2, loop_length=1.6, bridge_length=0.9)
divisor = [("v0", 1), ("v2", 2), ("l0_b", 1)]
coords, cycle_lengths = tropical_abel_jacobi_forward(graph, divisor)
# coords: shape (g,) in [0, cycle_len) per cycle
```

### Classical Abel–Jacobi (period tables + forward pass)

```python
from aj.classical import (
    make_hyperelliptic_cuts,
    build_omega_table,
    build_integral_table,
    abel_jacobi_forward,
    compute_aj_normalization,
)
import numpy as np

g = 2
grid_r = np.linspace(-3, 3, 32)
grid_i = np.linspace(-3, 3, 32)
base_point = complex(-5, -5)
cuts = make_hyperelliptic_cuts(g, r_max=3, r_min=-3, i_max=3, i_min=-3)
branch_pts = [z for a, b in cuts for z in (a, b)]

# Build period tables (ω and integrals on grid)
omega_plus = build_omega_table(g, branch_pts, grid_r, grid_i)
I_plus = build_integral_table(g, branch_pts, grid_r, grid_i, base_point)

# Forward: divisor (points, weights) -> R^{2g}
points = np.array([[0.5, 0.5], [-0.5, 0.5]])  # (x, y) = (Re, Im)
weights = np.array([1.0, -1.0])
mu, sigma = compute_aj_normalization(I_plus)
coords_2g = abel_jacobi_forward(points, weights, I_plus, grid_r, grid_i, mu=mu, sigma=sigma)
```

## Contents

| Item | Description |
|------|-------------|
| **`aj`** | Package: `aj.tropical` (graph, tropical AJ map), `aj.classical` (cuts, differentials, period tables, AJ forward). |
| `tropical_abel_jacobi.ipynb` | Tropical chain-of-loops, divisor → Jacobian, plots. |
| `higher_genus_lookup_tables.ipynb` | Precompute ω and I on a grid (e.g. genus 30); saves to Drive/local. |
| `AJ_training_genus30.ipynb` | Training with precomputed tables; 2D and 2g×2g baselines. |
| `planewave_analysis.ipynb` | Plane-wave / periodic approximations in the AJ setting. |
| **`aj.classical.theta_functions`** | Riemann theta (with characteristic), log θ, ∇θ, ∇log θ. |
| **`aj.classical.inverse_abel_jacobi_map`** | Inverse Abel–Jacobi via Newton on θ(A(z)−u)=0; `abel_map_vector`, `omega_vector`. |
| `theta_abel_jacobi.ipynb` | Evaluates theta functions and demonstrates inverse Abel–Jacobi with Newton’s method on log Klein theta (genus 2 example). |
| **`aj.classical.inverse_network`** | `InverseAbelJacobiNetwork`: PyTorch module using inverse Abel–Jacobi as forward pass. Parameters: branch points, base point, symmetric polynomial coefficients. Precomputes period matrix Ω and Riemann constant K. Uses x^n dx / y as basis of differentials. |

## Requirements

- **Core (library + tropical):** Python ≥3.9, NumPy, Matplotlib, NetworkX, mpmath, tqdm.
- **Classical period tables:** mpmath (required for integration).
- **Notebooks (higher-genus / training):** PyTorch, torchvision; Colab/Drive paths configurable.

## Creating the `aj` environment

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate aj
pip install -e .
jupyter notebook tropical_abel_jacobi.ipynb
```

**pip + venv:**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
pip install -e .
jupyter notebook tropical_abel_jacobi.ipynb
```

- **environment.yml** — Conda env named `aj` with Python, PyTorch, and notebook deps.  
- **requirements.txt** — Pip dependencies.  
- **pyproject.toml** — Package metadata and `pip install -e .` install.

## Quick start (tropical)

Run the tropical notebook or use the library as in the examples above.

## Running tests

Unit tests for theta and inverse Abel–Jacobi live in `tests/test_abel_jacobi_theta.py` and `tests/test_klienian_functions.py` (they import from `aj.classical`). From the repo root:

```bash
pip install pytest numpy mpmath   # if not already in the env
python -m pytest tests/test_abel_jacobi_theta.py -v
```

To skip slow tests (inverse AJ with mpmath integration): `pytest tests/ -v -m "not slow"`.

## License

Private research repository. All rights reserved.
