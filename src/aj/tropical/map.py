"""
Tropical Abel-Jacobi map: divisor -> Jacobian coordinates (mod cycle lengths).

Given a metric graph and a divisor (list of (node, weight)), computes the
tropical Abel-Jacobi image as coordinates on the Jacobian torus.
"""

import numpy as np
import networkx as nx

from .graph import cycle_data


def tropical_abel_jacobi_forward(graph, divisor, base="v0"):
    """Compute the tropical Abel-Jacobi image of a divisor.

    The divisor D = sum_i n_i P_i is mapped to a point in the Jacobian
    (torus) with one coordinate per cycle, each reduced mod the cycle length.

    Parameters
    ----------
    graph : networkx.Graph
        Metric graph with edge attribute `length`.
    divisor : list of (str, float)
        List of (node_name, weight); node names must be in the graph.
    base : str
        Base vertex for shortest-path distances and cycle basis.

    Returns
    -------
    coords : np.ndarray of shape (g,)
        Jacobian coordinates, one per cycle; each is in [0, cycle_len) when
        cycle_len > 0.
    cycle_lengths : list of float
        Length of each cycle (periods of the torus).
    """
    _, cycle_lengths, _ = cycle_data(graph, base=base)
    distances = nx.single_source_dijkstra_path_length(graph, base, weight="length")
    total_weight = sum(weight for _, weight in divisor)
    total_weight = total_weight if total_weight > 0 else 1

    coords = []
    for cycle_len in cycle_lengths:
        contribution = (
            sum(weight * distances.get(node, 0.0) for node, weight in divisor)
            / total_weight
        )
        coords.append(contribution % cycle_len if cycle_len > 0 else contribution)

    return np.array(coords), cycle_lengths


# Notebook compatibility: same function, alternate name.
tropical_abel_jacobi_divisor = tropical_abel_jacobi_forward
