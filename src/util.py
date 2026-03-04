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


def load_forward_tables(tables_dir: str | Path, device):
    """Load genus-30 forward AJ tables and precompute anchors/normalization."""
    import numpy as np
    import torch  # type: ignore[import]

    ints_path = Path(tables_dir) / "aj_integrals_genus30.pt"
    omeg_path = Path(tables_dir) / "aj_omegas_genus30.pt"
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
) -> Optional[Path]:
    """
    Find a directory containing both forward AJ table files for the given genus.

    Search order:
      1) explicit `tables_dir` (if provided)
      2) data_root
      3) data_root/tables
      4) data_root/AJ_Tables_g{genus}
    """
    candidates = []
    if tables_dir:
        candidates.append(Path(tables_dir))
    droot = Path(data_root)
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
        genus, seed=seed, r_max=r_max, r_min=r_min, i_max=i_max, i_min=i_min
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
):
    """
    Dataset-style forward-table loader:
      - If `tables_dir` is valid, load from there.
      - Else search common paths under `data_root`.
      - If still missing and `auto_build=True`, compute + save tables, then load.
    """
    found = find_forward_tables_dir(tables_dir=tables_dir, data_root=data_root, genus=genus)
    if found is None and auto_build:
        target = (
            Path(tables_dir).expanduser().resolve()
            if tables_dir
            else (Path(data_root).expanduser().resolve() / f"AJ_Tables_g{genus}")
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
        )
    if found is None:
        raise FileNotFoundError(
            "Forward AJ tables not found. Provide tables_dir or enable auto_build."
        )
    return load_forward_tables(found, device), found


def build_forward_model(tables_data: dict, device, embed_dim: int = 4, K: int = 2):
    """Build AJMNIST_AxisPeriodic on device."""
    import numpy as np
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]

    class AJGridActivationNorm(nn.Module):
        def __init__(self, I_plus, Om_plus, grid_r, grid_i, branch_pts, mu, sigma):
            super().__init__()
            self.g = I_plus.shape[0]
            self.register_buffer("I_plus", _pack_complex_table(I_plus))
            self.register_buffer("Om_plus", _pack_complex_table(Om_plus))
            self.register_buffer("mu", mu.view(1, 1, -1))
            self.register_buffer("sigma", sigma.view(1, 1, -1))
            self.gamma = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("r_min", torch.tensor(float(grid_r.min())))
            self.register_buffer("r_max", torch.tensor(float(grid_r.max())))
            self.register_buffer("i_min", torch.tensor(float(grid_i.min())))
            self.register_buffer("i_max", torch.tensor(float(grid_i.max())))
            self.register_buffer("bp_real", branch_pts.real.float())
            self.register_buffer("bp_imag", branch_pts.imag.float())

        def _map_raw_to_bounds(self, raw_xy):
            xr = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw_xy[..., 0])
            yi = self.i_min + (self.i_max - self.i_min) * torch.sigmoid(raw_xy[..., 1])
            return xr, yi

        def _norm_to_grid(self, xr, yi):
            gx = 2.0 * (xr - self.r_min) / (self.r_max - self.r_min) - 1.0
            gy = 2.0 * (yi - self.i_min) / (self.i_max - self.i_min) - 1.0
            return gx, gy

        def forward(self, raw_xy: torch.Tensor, sheet_logits: torch.Tensor, return_aux=True):
            B, g, _ = raw_xy.shape
            assert g == self.g
            xr, yi = self._map_raw_to_bounds(raw_xy)
            gx, gy = self._norm_to_grid(xr, yi)
            grid = torch.stack([gx, gy], dim=-1).view(B * g, 1, 1, 2)
            I = F.grid_sample(
                self.I_plus.expand(B * g, -1, -1, -1),
                grid, mode="bilinear", align_corners=True
            ).view(B, g, -1)
            I_std = (I - self.mu) / self.sigma
            sign = torch.tanh(sheet_logits).unsqueeze(-1)
            contrib = sign * I_std
            coords = self.gamma * contrib.sum(dim=1)
            aux = None
            if return_aux:
                margin = 0.95
                bpen = ((gx.abs() - margin).clamp_min(0) ** 2 + (gy.abs() - margin).clamp_min(0) ** 2).mean()
                dx = xr.unsqueeze(-1) - self.bp_real
                dy = yi.unsqueeze(-1) - self.bp_imag
                d2 = dx * dx + dy * dy
                tau = 0.07
                rpen = torch.exp(-d2 / (2 * tau * tau)).mean()
                aux = {"bound_penalty": bpen, "branch_penalty": rpen}
            return coords, aux

    class AJMNIST_Anchored(nn.Module):
        def __init__(self, genus, I_plus, Om_plus, grid_r, grid_i, branch_pts,
                     anchors_xy, mu, sigma, embed_dim=8):
            super().__init__()
            self.genus = genus
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.embed = nn.Parameter(torch.empty(genus, embed_dim))
            nn.init.uniform_(self.embed, -2.0, 2.0)
            self.point_head = nn.Linear(64 + embed_dim, 3, bias=False)
            nn.init.xavier_uniform_(self.point_head.weight)
            self.point_bias = nn.Parameter(torch.zeros(genus, 3))
            self.aj = AJGridActivationNorm(I_plus, Om_plus, grid_r, grid_i, branch_pts, mu, sigma)
            self.classifier = nn.Linear(2 * genus, 10)
            rmin, rmax = float(grid_r.min()), float(grid_r.max())
            imin, imax = float(grid_i.min()), float(grid_i.max())

            def logit(p):
                p = np.clip(p, 1e-6, 1 - 1e-6)
                return float(np.log(p / (1 - p)))

            with torch.no_grad():
                for i in range(genus):
                    x0, y0 = float(anchors_xy[i, 0]), float(anchors_xy[i, 1])
                    px = (x0 - rmin) / (rmax - rmin)
                    py = (y0 - imin) / (imax - imin)
                    self.point_bias[i, 0] = logit(np.array(px))
                    self.point_bias[i, 1] = logit(np.array(py))
                    target_sign = 0.8 if (i % 2 == 0) else -0.8
                    self.point_bias[i, 2] = float(torch.atanh(torch.tensor(target_sign)))

        def forward(self, x, return_aux=False):
            B = x.size(0)
            h = self.conv(x).view(B, -1)
            h_exp = h.unsqueeze(1).expand(-1, self.genus, -1)
            emb = self.embed.unsqueeze(0).expand(B, -1, -1)
            out = self.point_head(torch.cat([h_exp, emb], dim=2)) + self.point_bias.unsqueeze(0)
            raw_xy, sheet_logits = out[..., :2], out[..., 2]
            coords, aux = self.aj(raw_xy, sheet_logits, return_aux=True)
            logits = self.classifier(coords)
            if return_aux:
                return logits, aux
            return logits

    class TorusFeatures(nn.Module):
        def __init__(self, dim: int, K: int = 2, freqs=None, learnable: bool = False):
            super().__init__()
            if freqs is None:
                freqs = torch.tensor([0.5, 1.0], dtype=torch.float32)[:K]
            else:
                freqs = torch.as_tensor(freqs, dtype=torch.float32)[:K]
            if learnable:
                self.freqs = nn.Parameter(freqs)
            else:
                self.register_buffer("freqs", freqs)
            self.dim, self.K = dim, len(freqs)

        def forward(self, u):
            B, D = u.shape
            f = self.freqs.view(1, 1, -1).to(u.device)
            ang = u.unsqueeze(-1) * f
            return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).view(B, D * 2 * self.K)

    class AJMNIST_AxisPeriodic(nn.Module):
        def __init__(self, genus, I_plus, Om_plus, grid_r, grid_i, branch_pts,
                     anchors_xy, mu, sigma, embed_dim=8, K=2, learnable_freqs=False):
            super().__init__()
            self.base = AJMNIST_Anchored(
                genus, I_plus, Om_plus, grid_r, grid_i, branch_pts,
                anchors_xy, mu, sigma, embed_dim=embed_dim,
            )
            D = 2 * genus
            self.torus = TorusFeatures(D, K=K, learnable=learnable_freqs)
            self.classifier = nn.Linear(D * 2 * K, 10)

        def forward(self, x, return_aux=False):
            B = x.size(0)
            h = self.base.conv(x).view(B, -1)
            h_exp = h.unsqueeze(1).expand(-1, self.base.genus, -1)
            emb = self.base.embed.unsqueeze(0).expand(B, -1, -1)
            out = self.base.point_head(torch.cat([h_exp, emb], dim=2)) + self.base.point_bias.unsqueeze(0)
            raw_xy, sheet_logits = out[..., :2], out[..., 2]
            coords, aux = self.base.aj(raw_xy, sheet_logits, return_aux=True)
            feats = self.torus(coords)
            logits = self.classifier(feats)
            if return_aux:
                return logits, aux
            return logits

    return AJMNIST_AxisPeriodic(
        tables_data["genus"],
        tables_data["I_plus"], tables_data["Om_plus"],
        tables_data["grid_r"], tables_data["grid_i"],
        tables_data["branch_pts_t"],
        tables_data["anchors_xy_t"],
        tables_data["mu_t"], tables_data["sigma_t"],
        embed_dim=embed_dim, K=K, learnable_freqs=False,
    ).to(device)


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

