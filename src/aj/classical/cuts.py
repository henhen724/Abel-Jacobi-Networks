"""
Hyperelliptic branch cuts: 2g+2 branch points arranged as g+1 cuts.

Used to define the curve y^2 = prod(t - a_i) and the holomorphic differentials.
"""

import numpy as np


def make_hyperelliptic_cuts(
    g,
    radius=4.0,
    jitter=0.25,
    seed=123,
    r_max=6.0,
    r_min=-6.0,
    i_max=6.0,
    i_min=-6.0,
):
    """Build g+1 branch cuts for a hyperelliptic curve of genus g.

    Places 2g+2 points on a perturbed circle and pairs neighbors to form
    short segments, avoiding overlaps. Returns g+1 pairs (a, b) with
    a, b complex.

    Parameters
    ----------
    g : int
        Genus (number of cuts is g+1).
    radius : float
        Circle radius for base placement.
    jitter : float
        Random radial jitter (e.g. 0.25).
    seed : int
        Random seed for reproducibility.
    r_max, r_min, i_max, i_min : float
        Grid bounds; used to scale points so they stay within the box.

    Returns
    -------
    cuts : list of (complex, complex)
        g+1 pairs of endpoints for branch cuts.
    """
    rng = np.random.RandomState(seed)
    m = 2 * g + 2
    thetas = np.linspace(0, 2 * np.pi, m, endpoint=False)
    rng.shuffle(thetas)
    radii = radius * (1.0 + jitter * (rng.rand(m) - 0.5))
    pts = radii * np.exp(1j * thetas)

    scale = max(
        (pts.real.max() - pts.real.min()) / (r_max - r_min + 1e-6),
        (pts.imag.max() - pts.imag.min()) / (i_max - i_min + 1e-6),
    )
    if scale > 0.85:
        pts = pts / (scale / 0.85)

    remaining = list(range(m))
    cuts = []
    while remaining:
        i = remaining.pop(0)
        pi = pts[i]
        dists = [(j, abs(pi - pts[j])) for j in remaining]
        j = min(dists, key=lambda t: t[1])[0]
        remaining.remove(j)
        a, b = pi, pts[j]
        mid = 0.5 * (a + b)
        a = a + 0.05 * (a - mid)
        b = b + 0.05 * (b - mid)
        cuts.append((complex(a), complex(b)))

    if len(cuts) > g + 1:
        cuts.sort(key=lambda ab: -abs(ab[0] - ab[1]))
        cuts = cuts[: g + 1]
    assert len(cuts) == g + 1, f"Expected {g+1} cuts, got {len(cuts)}."
    return cuts
