from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


def get_repo_root(start: Path | str | None = None) -> Path:
    """
    Return the repository root, assuming this file lives under src/ in the repo.

    This is handy for notebooks and scripts that may be run from either the
    repo root or from a subdirectory like notebooks/.
    """
    if start is None:
        start_path = Path(__file__).resolve()
    else:
        start_path = Path(start).resolve()

    for p in [start_path, *start_path.parents]:
        if (p / "pyproject.toml").is_file() or (p / ".git").is_dir():
            return p
    # Fallback: current working directory
    return Path.cwd()


def ensure_mnist_available(root: str | Path = "./data") -> None:
    """
    Ensure MNIST files exist under ``root`` without relying on the default
    Yann LeCun HTTP endpoint (which can be flaky on some clusters).

    Strategy (mirrors the old aj_mnist_test_accuracy helper):
      - If processed ``training.pt`` exists, do nothing.
      - Otherwise, download the raw .gz files from the Google-hosted mirror:
            https://storage.googleapis.com/cvdf-datasets/mnist/
        into ``root/MNIST/raw``, matching torchvision's expected filenames.
      - Then let ``torchvision.datasets.MNIST(download=True)`` do its normal processing.
    """
    import urllib.request
    import torchvision  # type: ignore[import]

    root = os.path.abspath(os.fspath(root))
    mnist_root = os.path.join(root, "MNIST")
    processed_dir = os.path.join(mnist_root, "processed")
    raw_dir = os.path.join(mnist_root, "raw")

    training_pt = os.path.join(processed_dir, "training.pt")
    if os.path.isfile(training_pt):
        return

    os.makedirs(raw_dir, exist_ok=True)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for fname in files:
        url = f"{base_url}/{fname}"
        dest = os.path.join(raw_dir, fname)
        print(f"Downloading {fname} from {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # pragma: no cover - network / filesystem dependent
            raise RuntimeError(f"Failed to download {fname} from {url}: {e}") from e

    # Let torchvision handle integrity checks and processing
    torchvision.datasets.MNIST(root=root, train=True, download=True)


def get_mnist_loaders(
    root: str | Path = "./data",
    test_batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """
    Return MNIST (train, test) DataLoaders with the standard normalization used
    across the AJ MNIST notebooks and scripts.
    """
    import torch  # type: ignore[import]
    import torchvision  # type: ignore[import]
    import torchvision.transforms as T  # type: ignore[import]

    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(root=os.fspath(root), train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root=os.fspath(root), train=False, download=True, transform=tfm)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def safe_torch_load(path: str | Path, map_location=None):
    """Load checkpoint/state; use weights_only=False when supported."""
    import torch  # type: ignore[import]

    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        kwargs["weights_only"] = False
        return torch.load(os.fspath(path), **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(os.fspath(path), **kwargs)


def _pack_complex_table(table_gHW):
    re, im = table_gHW.real, table_gHW.imag
    import torch  # type: ignore[import]

    return torch.cat([re, im], dim=0).unsqueeze(0).contiguous()


def load_forward_tables(tables_dir: str | Path, device, genus: int = 30):
    """Load forward AJ tables and precompute anchors/normalization."""
    import numpy as np
    import torch  # type: ignore[import]

    ints_path, omeg_path = _forward_table_paths(tables_dir, genus)
    if not ints_path.is_file() or not omeg_path.is_file():
        return None

    ints = safe_torch_load(ints_path, map_location="cpu")
    omeg = safe_torch_load(omeg_path, map_location="cpu")

    genus = int(ints["genus"])
    grid_r_np = np.array(ints["grid_r"])
    grid_i_np = np.array(ints["grid_i"])
    branch_pts = np.array(ints["branch_pts"])
    I_plus = ints["I_plus"]
    Om_plus = omeg["omega_plus"]
    # Avoid NaN/Inf from integration near branch points so loss stays finite
    if torch.is_tensor(I_plus) and (torch.isnan(I_plus).any() or torch.isinf(I_plus).any()):
        I_plus = torch.nan_to_num(I_plus, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.is_tensor(Om_plus) and (torch.isnan(Om_plus).any() or torch.isinf(Om_plus).any()):
        Om_plus = torch.nan_to_num(Om_plus, nan=0.0, posinf=0.0, neginf=0.0)
    H, W = I_plus.shape[-2:]
    grid_r = torch.tensor(grid_r_np, dtype=torch.float32)
    grid_i = torch.tensor(grid_i_np, dtype=torch.float32)
    branch_pts_t = torch.tensor(branch_pts)

    with torch.no_grad():
        Wmap = Om_plus.abs().sum(dim=0)
        Wmap_np = Wmap.cpu().numpy()
    H, W = Wmap_np.shape
    gy = np.linspace(-1, 1, H)[:, None]
    gx = np.linspace(-1, 1, W)[None, :]
    edge_mask = (np.abs(gx) < 0.92) & (np.abs(gy) < 0.92)
    grid_x = grid_r_np[None, :].repeat(H, axis=0)
    grid_y = grid_i_np[:, None].repeat(W, axis=1)
    bp_real = np.real(branch_pts).reshape(-1, 1, 1)
    bp_imag = np.imag(branch_pts).reshape(-1, 1, 1)
    d2 = (grid_x - bp_real) ** 2 + (grid_y - bp_imag) ** 2
    bp_mask = d2.min(axis=0) > 0.25
    mask = edge_mask & bp_mask
    score = np.where(mask, Wmap_np, -np.inf)
    th = np.quantile(score[score > -np.inf], 0.85)
    cand = np.argwhere(score >= th)

    def farthest_k(points_hw, k):
        pts = points_hw.copy()
        start = pts[np.argmax(score[tuple(pts.T)])]
        chosen = [start]
        if k == 1:
            return np.array(chosen)
        coords = np.stack([grid_x[tuple(pts.T)], grid_y[tuple(pts.T)]], axis=1)
        c0 = np.array([grid_x[start[0], start[1]], grid_y[start[0], start[1]]])[None, :]
        mind = np.sum((coords - c0) ** 2, axis=1)
        for _ in range(1, k):
            j = np.argmax(mind)
            chosen.append(pts[j])
            cj = coords[j][None, :]
            mind = np.minimum(mind, np.sum((coords - cj) ** 2, axis=1))
        return np.array(chosen)

    anchors_hw = farthest_k(cand, genus)
    x0 = grid_r_np[anchors_hw[:, 1]]
    y0 = grid_i_np[anchors_hw[:, 0]]
    anchors_xy = np.stack([x0, y0], axis=1)
    I_re = I_plus.real
    I_im = I_plus.imag
    I_ch = torch.cat([I_re, I_im], dim=0)
    mu = I_ch.mean(dim=(1, 2))
    sigma = I_ch.std(dim=(1, 2)).clamp_min(1e-6)
    anchors_xy_t = torch.tensor(anchors_xy, dtype=torch.float32)
    mu_t = mu.float()
    sigma_t = sigma.float()

    return {
        "genus": genus,
        "I_plus": I_plus.to(device),
        "Om_plus": Om_plus.to(device),
        "grid_r": grid_r.to(device),
        "grid_i": grid_i.to(device),
        "branch_pts_t": branch_pts_t.to(device),
        "anchors_xy_t": anchors_xy_t.to(device),
        "mu_t": mu_t.to(device),
        "sigma_t": sigma_t.to(device),
    }


def refresh_forward_table_normalization(tables: dict, device=None, nan_safe: bool = True) -> dict:
    """
    Set ``tables[\"mu_t\"]`` and ``tables[\"sigma_t\"]`` for AJGridActivationNorm.

    By default uses ``aj.classical.compute_aj_normalization`` (nan-safe over the grid).
    If ``nan_safe=False``, keeps mean/std already computed in ``load_forward_tables``.
    """
    import torch  # type: ignore[import]

    if not nan_safe:
        return tables
    from aj.classical import compute_aj_normalization

    mu, sigma = compute_aj_normalization(tables["I_plus"])
    mu_t = torch.as_tensor(mu, dtype=torch.float32)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32)
    if device is not None:
        mu_t = mu_t.to(device)
        sigma_t = sigma_t.to(device)
    else:
        ref = tables.get("mu_t")
        if torch.is_tensor(ref):
            mu_t = mu_t.to(ref.device)
            sigma_t = sigma_t.to(ref.device)
    tables["mu_t"] = mu_t
    tables["sigma_t"] = sigma_t
    return tables


def _forward_table_paths(base_dir: str | Path, genus: int) -> Tuple[Path, Path]:
    base = Path(base_dir)
    return (
        base / f"aj_integrals_genus{genus}.pt",
        base / f"aj_omegas_genus{genus}.pt",
    )


def find_forward_tables_dir(
    tables_dir: Optional[str | Path] = None,
    data_root: str | Path = "./data",
    genus: int = 30,
    subdir: Optional[str] = None,
) -> Optional[Path]:
    """
    Find a directory containing both forward AJ table files for the given genus.

    Search order:
      1) explicit `tables_dir` (if provided)
      2) data_root / subdir (if subdir is provided)
      3) data_root
      4) data_root/tables
      5) data_root/AJ_Tables_g{genus}
    """
    candidates = []
    if tables_dir:
        candidates.append(Path(tables_dir))
    droot = Path(data_root).expanduser().resolve()
    if subdir:
        candidates.append(droot / subdir)
    candidates.extend([droot, droot / "tables", droot / f"AJ_Tables_g{genus}"])

    seen = set()
    for c in candidates:
        c = c.expanduser().resolve()
        if str(c) in seen:
            continue
        seen.add(str(c))
        ints_path, omeg_path = _forward_table_paths(c, genus)
        if ints_path.is_file() and omeg_path.is_file():
            return c
    return None


def build_and_save_forward_tables(
    out_dir: str | Path,
    genus: int = 30,
    grid_size: int = 96,
    r_min: float = -6.0,
    r_max: float = 6.0,
    i_min: float = -6.0,
    i_max: float = 6.0,
    base_point: complex = complex(-8.0, -8.0),
    seed: int = 123,
    radius: float = 4.0,
    jitter: float = 0.25,
) -> Path:
    """
    Build forward AJ lookup tables with `aj.classical` and save them to disk.

    This can be expensive for large genus/grid_size.
    """
    import numpy as np
    import torch  # type: ignore[import]
    from aj.classical import (  # type: ignore[import]
        build_integral_table,
        build_omega_table,
        make_hyperelliptic_cuts,
    )

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    grid_r = np.linspace(r_min, r_max, grid_size)
    grid_i = np.linspace(i_min, i_max, grid_size)
    cuts = make_hyperelliptic_cuts(
        genus,
        seed=seed,
        radius=radius,
        jitter=jitter,
        r_max=r_max,
        r_min=r_min,
        i_max=i_max,
        i_min=i_min,
    )
    branch_pts = np.array([z for a, b in cuts for z in (a, b)], dtype=np.complex128)

    print(
        f"Building forward AJ tables: genus={genus}, grid={grid_size}x{grid_size} "
        f"(this may take a while)"
    )
    omega_plus = build_omega_table(genus, branch_pts, grid_r, grid_i)
    I_plus = build_integral_table(genus, branch_pts, grid_r, grid_i, base_point)

    ints_path, omeg_path = _forward_table_paths(out, genus)
    torch.save(
        {
            "genus": int(genus),
            "grid_r": grid_r,
            "grid_i": grid_i,
            "branch_pts": branch_pts,
            "I_plus": torch.as_tensor(I_plus),
        },
        ints_path,
    )
    torch.save(
        {
            "genus": int(genus),
            "grid_r": grid_r,
            "grid_i": grid_i,
            "branch_pts": branch_pts,
            "omega_plus": torch.as_tensor(omega_plus),
        },
        omeg_path,
    )
    print(f"Saved forward tables to {out}")
    return out


def _atomic_torch_save(payload, path: str | Path) -> None:
    import torch  # type: ignore[import]

    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)


def _omega_eval_torch(t, k: int, branch_pts_t):
    """
    Evaluate omega_k(t) = t^k / sqrt(prod_j (t - a_j)) using torch complex tensors.
    Supports vectorized `t` with arbitrary leading shape.
    """
    import torch  # type: ignore[import]

    a = branch_pts_t.view(*([1] * t.ndim), -1)
    prod = torch.prod(t.unsqueeze(-1) - a, dim=-1)
    return torch.pow(t, k) / torch.sqrt(prod)


def build_and_save_forward_tables_gpu_checkpointed(
    out_dir: str | Path,
    genus: int = 30,
    grid_size: int = 96,
    r_min: float = -6.0,
    r_max: float = 6.0,
    i_min: float = -6.0,
    i_max: float = 6.0,
    base_point: complex = complex(-8.0, -8.0),
    seed: int = 123,
    integration_steps: int = 128,
    chunk_points: int = 2048,
    save_every_chunks: int = 8,
    resume: bool = True,
    device=None,
) -> Path:
    """
    Build forward AJ lookup tables with torch vectorization (GPU if available) and
    save partial progress checkpoints to disk.

    Notes
    -----
    - This is a fast approximate integrator (trapezoidal rule on straight-line paths),
      unlike the mpmath high-precision `build_integral_table`.
    - It is designed for practical throughput on large table builds.
    """
    import numpy as np
    import torch  # type: ignore[import]
    from aj.classical import make_hyperelliptic_cuts  # type: ignore[import]

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ints_path, omeg_path = _forward_table_paths(out, genus)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Building forward tables on device: {device}")

    grid_r = np.linspace(r_min, r_max, grid_size).astype(np.float64)
    grid_i = np.linspace(i_min, i_max, grid_size).astype(np.float64)
    H = W = grid_size
    N = H * W

    cuts = make_hyperelliptic_cuts(
        genus, seed=seed, r_max=r_max, r_min=r_min, i_max=i_max, i_min=i_min
    )
    branch_pts = np.array([z for a, b in cuts for z in (a, b)], dtype=np.complex128)
    branch_pts_t = torch.as_tensor(branch_pts, dtype=torch.complex64, device=device)

    rr, ii = np.meshgrid(grid_r, grid_i)
    z_flat = torch.as_tensor((rr + 1j * ii).reshape(-1), dtype=torch.complex64, device=device)
    base = torch.tensor(base_point, dtype=torch.complex64, device=device)
    s = torch.linspace(0.0, 1.0, integration_steps, dtype=torch.float32, device=device)

    # Resume / initialize omega table
    if resume and omeg_path.is_file():
        pkg = safe_torch_load(omeg_path, map_location="cpu")
        Om_plus = pkg.get("omega_plus")
        prog_o = pkg.get("progress", {"k_done": 0, "point_done": [0] * genus})
        if Om_plus is None or int(pkg.get("genus", -1)) != genus:
            Om_plus = torch.zeros((genus, H, W), dtype=torch.complex64)
            prog_o = {"k_done": 0, "point_done": [0] * genus}
        else:
            Om_plus = Om_plus.to(torch.complex64)
    else:
        Om_plus = torch.zeros((genus, H, W), dtype=torch.complex64)
        prog_o = {"k_done": 0, "point_done": [0] * genus}

    Om_flat = Om_plus.view(genus, -1)
    print(f"Omega progress: k_done={prog_o['k_done']}/{genus}")

    for k in range(int(prog_o["k_done"]), genus):
        start = int(prog_o["point_done"][k])
        n_chunks = 0
        for idx in range(start, N, chunk_points):
            j = min(idx + chunk_points, N)
            z = z_flat[idx:j]
            vals = _omega_eval_torch(z, k, branch_pts_t)
            Om_flat[k, idx:j] = vals.detach().cpu()
            prog_o["point_done"][k] = j
            n_chunks += 1
            if n_chunks % max(1, save_every_chunks) == 0:
                _atomic_torch_save(
                    {
                        "genus": int(genus),
                        "grid_r": grid_r,
                        "grid_i": grid_i,
                        "branch_pts": branch_pts,
                        "omega_plus": Om_plus,
                        "progress": prog_o,
                    },
                    omeg_path,
                )
                print(f"Saved omega partial: k={k}, points={j}/{N}")
        prog_o["k_done"] = k + 1
        _atomic_torch_save(
            {
                "genus": int(genus),
                "grid_r": grid_r,
                "grid_i": grid_i,
                "branch_pts": branch_pts,
                "omega_plus": Om_plus,
                "progress": prog_o,
            },
            omeg_path,
        )
        print(f"Omega done for k={k} ({k+1}/{genus})")

    # Resume / initialize integral table
    if resume and ints_path.is_file():
        pkg = safe_torch_load(ints_path, map_location="cpu")
        I_plus = pkg.get("I_plus")
        prog_i = pkg.get("progress", {"k_done": 0, "point_done": [0] * genus})
        if I_plus is None or int(pkg.get("genus", -1)) != genus:
            I_plus = torch.zeros((genus, H, W), dtype=torch.complex64)
            prog_i = {"k_done": 0, "point_done": [0] * genus}
        else:
            I_plus = I_plus.to(torch.complex64)
    else:
        I_plus = torch.zeros((genus, H, W), dtype=torch.complex64)
        prog_i = {"k_done": 0, "point_done": [0] * genus}

    I_flat = I_plus.view(genus, -1)
    print(f"Integral progress: k_done={prog_i['k_done']}/{genus}")

    for k in range(int(prog_i["k_done"]), genus):
        start = int(prog_i["point_done"][k])
        n_chunks = 0
        for idx in range(start, N, chunk_points):
            j = min(idx + chunk_points, N)
            z = z_flat[idx:j]  # (B,)
            dz = (z - base).unsqueeze(1)  # (B,1)
            t = base + dz * s.unsqueeze(0)  # (B,Q)
            omega_t = _omega_eval_torch(t, k, branch_pts_t)  # (B,Q)
            integrand = omega_t * dz  # (B,Q)
            vals = torch.trapz(integrand, s, dim=1)  # (B,)
            I_flat[k, idx:j] = vals.detach().cpu()
            prog_i["point_done"][k] = j
            n_chunks += 1
            if n_chunks % max(1, save_every_chunks) == 0:
                _atomic_torch_save(
                    {
                        "genus": int(genus),
                        "grid_r": grid_r,
                        "grid_i": grid_i,
                        "branch_pts": branch_pts,
                        "I_plus": I_plus,
                        "progress": prog_i,
                    },
                    ints_path,
                )
                print(f"Saved integral partial: k={k}, points={j}/{N}")
        prog_i["k_done"] = k + 1
        _atomic_torch_save(
            {
                "genus": int(genus),
                "grid_r": grid_r,
                "grid_i": grid_i,
                "branch_pts": branch_pts,
                "I_plus": I_plus,
                "progress": prog_i,
            },
            ints_path,
        )
        print(f"Integral done for k={k} ({k+1}/{genus})")

    print(f"Saved forward tables to {out}")
    return out


def get_or_build_forward_tables(
    device,
    tables_dir: Optional[str | Path] = None,
    data_root: str | Path = "./data",
    genus: int = 30,
    auto_build: bool = True,
    grid_size: int = 96,
    r_min: float = -6.0,
    r_max: float = 6.0,
    i_min: float = -6.0,
    i_max: float = 6.0,
    base_point: complex = complex(-8.0, -8.0),
    seed: int = 123,
    radius: float = 4.0,
    jitter: float = 0.25,
    tables_subdir: Optional[str] = None,
):
    """
    Dataset-style forward-table loader:
      - If `tables_dir` is valid, load from there.
      - Else search common paths under `data_root` (and `data_root/tables_subdir` if set).
      - If still missing and `auto_build=True`, compute + save tables, then load.

    Use `tables_subdir` (e.g. "AJ_Tables_g2_spiral_s123_r4_j0.25_b4") to cache
    spiral or other config-specific tables separately from the default genus cache.
    """
    droot = Path(data_root).expanduser().resolve()
    found = find_forward_tables_dir(
        tables_dir=tables_dir, data_root=data_root, genus=genus, subdir=tables_subdir
    )
    if found is None and auto_build:
        target = (
            Path(tables_dir).expanduser().resolve()
            if tables_dir
            else (droot / (tables_subdir or f"AJ_Tables_g{genus}"))
        )
        found = build_and_save_forward_tables(
            target,
            genus=genus,
            grid_size=grid_size,
            r_min=r_min,
            r_max=r_max,
            i_min=i_min,
            i_max=i_max,
            base_point=base_point,
            seed=seed,
            radius=radius,
            jitter=jitter,
        )
    if found is None:
        raise FileNotFoundError(
            "Forward AJ tables not found. Provide tables_dir or enable auto_build."
        )
    return load_forward_tables(found, device, genus=genus), found


def tables_for_spiral_forward(tables: dict) -> dict:
    """
    Convert loaded forward tables (from load_forward_tables) to the dict format
    expected by ForwardAJ2D in the spiral notebook (numpy arrays, keys I_plus,
    omega_plus, grid_r, grid_i, branch_pts, mu, sigma).
    """
    import numpy as np
    import torch  # type: ignore[import]

    def to_np(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return np.asarray(x)

    branch_pts = tables["branch_pts_t"]
    if torch.is_tensor(branch_pts):
        bp = branch_pts.cpu().numpy()
        if bp.dtype.kind != "c":
            bp = bp[:, 0] + 1j * bp[:, 1]
    else:
        bp = np.asarray(branch_pts)
        if bp.dtype.kind != "c" and bp.ndim == 2 and bp.shape[1] == 2:
            bp = bp[:, 0] + 1j * bp[:, 1]

    return {
        "genus": int(tables["genus"]),
        "I_plus": to_np(tables["I_plus"]),
        "omega_plus": to_np(tables["Om_plus"]),
        "grid_r": to_np(tables["grid_r"]),
        "grid_i": to_np(tables["grid_i"]),
        "branch_pts": bp,
        "mu": to_np(tables["mu_t"]).astype(np.float32),
        "sigma": to_np(tables["sigma_t"]).astype(np.float32),
    }


def eval_epoch(model, loader, device) -> Tuple[float, float]:
    """Evaluation helper: CE loss and accuracy."""
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]

    ce = nn.CrossEntropyLoss()
    model.eval()
    tot, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot += ce(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return tot / n, 100.0 * correct / n

