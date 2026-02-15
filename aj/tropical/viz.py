"""
Visualization helpers for the tropical metric graph and Jacobian projection.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx


def plot_metric_graph(graph, positions, highlight=None, ax=None):
    """Draw the metric graph with edge lengths; optionally highlight divisor points.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with edge attribute `length`.
    positions : dict
        Node -> (x, y) for layout.
    highlight : list of (node, weight) or None
        Divisor points to highlight with larger markers and legend.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; uses plt.gca() if None.
    """
    ax = ax or plt.gca()
    for u, v, data in graph.edges(data=True):
        xs = [positions[u][0], positions[v][0]]
        ys = [positions[u][1], positions[v][1]]
        ax.plot(xs, ys, color="black")
        mid = ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2)
        ax.text(
            *mid, f"{data['length']:.2f}", ha="center", va="center", fontsize=8, color="gray"
        )
    node_positions = list(positions.values())
    ax.scatter(*zip(*node_positions), color="black", s=30, zorder=3)
    if highlight is not None:
        for node, weight in highlight:
            if node in positions:
                ax.scatter(
                    *positions[node],
                    s=80 + weight * 20,
                    label=f"{node} (weight={weight})",
                    zorder=4,
                )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, max(x for x, _ in positions.values()) + 0.5)
    ax.set_ylim(-0.5, 1.5)
    if highlight is not None:
        ax.legend(loc="upper right", fontsize=8)


def plot_jacobian_projection(coords, cycle_lengths, ax=None):
    """Draw the Jacobian torus projection (first two cycle coordinates).

    Parameters
    ----------
    coords : array-like of shape (g,)
        Jacobian coordinates (one per cycle).
    cycle_lengths : list of float
        Periods (cycle lengths).
    ax : matplotlib.axes.Axes or None
        Axes to draw on; uses plt.gca() if None.
    """
    ax = ax or plt.gca()
    g = len(coords)
    if g == 0:
        ax.text(0.5, 0.5, "no cycles", ha="center", va="center")
        return
    if g == 1:
        lim = cycle_lengths[0] if cycle_lengths else 1.0
        ax.plot([0, coords[0]], [0, 0], marker="o")
        ax.set_xlim(-0.1 * lim, 1.1 * lim)
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel("dummy coordinate")
        ax.set_xlabel("cycle 1 coordinate")
        ax.add_patch(
            Rectangle((-0.1 * lim, -0.4), lim * 1.1, 0.8, fill=False, linestyle="--", edgecolor="gray")
        )
        ax.set_title("Jacobian circle projection")
        return
    xs, ys = coords[0], coords[1] if g >= 2 else (coords[0], 0)
    rect_width = cycle_lengths[0] if cycle_lengths else 1.0
    rect_height = cycle_lengths[1] if len(cycle_lengths) > 1 else 1.0
    ax.add_patch(
        Rectangle((0, 0), rect_width, rect_height, fill=False, linestyle="--", edgecolor="gray")
    )
    ax.scatter(xs, ys, s=80, color="tab:red")
    ax.set_xlim(-0.1 * rect_width, rect_width * 1.1)
    ax.set_ylim(-0.1 * rect_height, rect_height * 1.1)
    ax.set_xlabel("cycle 1")
    ax.set_ylabel("cycle 2")
    ax.set_title("Jacobian slice (projection)")
