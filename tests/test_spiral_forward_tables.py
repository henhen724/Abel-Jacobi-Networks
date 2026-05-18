"""
Test that spiral forward table loading works (get_or_build_forward_tables +
tables_for_spiral_forward). Run with: pytest tests/test_spiral_forward_tables.py -v
Requires aj_env (or torch, numpy, src.util).
"""
import pytest
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def repo_path():
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def test_spiral_forward_tables_load_and_forward(repo_path):
    from src import util as aj_util
    import torch
    import numpy as np

    device = torch.device("cpu")
    genus = 2
    grid_bound = 4.0
    # Load (or build if missing) with same args as spiral notebook
    tables, tables_found = aj_util.get_or_build_forward_tables(
        device,
        data_root=repo_path / "data",
        genus=genus,
        grid_size=32,
        r_min=-grid_bound,
        r_max=grid_bound,
        i_min=-grid_bound,
        i_max=grid_bound,
        base_point=complex(-5.0, -5.0),
        seed=123,
        radius=4.0,
        jitter=0.25,
        auto_build=True,
    )
    assert tables is not None
    assert Path(tables_found).is_dir()
    tables_2d = aj_util.tables_for_spiral_forward(tables)
    assert tables_2d["genus"] == genus
    assert "I_plus" in tables_2d and "omega_plus" in tables_2d
    assert "mu" in tables_2d and "sigma" in tables_2d
    # Minimal forward: build a 2-layer like ForwardAJ2D and run one batch
    g = tables_2d["genus"]
    I_plus = tables_2d["I_plus"]
    mu = np.asarray(tables_2d["mu"], dtype=np.float32)
    sigma = np.asarray(tables_2d["sigma"], dtype=np.float32)
    assert I_plus.shape[0] == g
    assert mu.shape[0] == 2 * g
    assert sigma.shape[0] == 2 * g
