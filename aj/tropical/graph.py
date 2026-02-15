"""
Tropical hyperelliptic metric graph: chain of loops.

Graph construction uses `length` edge attribute and node names like
`v0`, `l0_a`, `l0_b`. Genus g gives g loops between g+1 backbone vertices.
"""

import networkx as nx


def build_chain_of_loops(genus, loop_length=1.0, bridge_length=1.0):
    """Build a chain-of-loops metric graph of given genus.

    Each loop spans between consecutive backbone vertices so that the final
    graph has genus equal to `genus`. Edge lengths are stored in the
    `length` attribute.

    Parameters
    ----------
    genus : int
        Number of loops (genus of the tropical curve).
    loop_length : float
        Total length of each loop (split into three edges per loop).
    bridge_length : float
        Length of each backbone edge between consecutive vertices.

    Returns
    -------
    graph : networkx.Graph
        Graph with nodes v0, v1, ..., v{genus} and l{idx}_a, l{idx}_b per loop.
    positions : dict
        Mapping node -> (x, y) for layout.
    """
    graph = nx.Graph()
    positions = {}
    spacing = bridge_length + 0.6

    for idx in range(genus + 1):
        node = f"v{idx}"
        graph.add_node(node)
        positions[node] = (idx * spacing, 0.0)

    for idx in range(genus):
        v0, v1 = f"v{idx}", f"v{idx + 1}"
        graph.add_edge(v0, v1, length=bridge_length)

        loop_a = f"l{idx}_a"
        loop_b = f"l{idx}_b"
        graph.add_node(loop_a)
        graph.add_node(loop_b)
        positions[loop_a] = (positions[v0][0], 0.8)
        positions[loop_b] = (positions[v1][0], 0.8)

        third = loop_length / 3
        graph.add_edge(v0, loop_a, length=third)
        graph.add_edge(loop_a, loop_b, length=third)
        graph.add_edge(loop_b, v1, length=third)

    return graph, positions


def cycle_data(graph, base="v0"):
    """Compute cycle basis and per-cycle lengths for a metric graph.

    Uses the graph's `length` edge attribute. Cycles are computed from
    the cycle basis rooted at `base`.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with edge attribute `length`.
    base : str
        Root node for cycle basis.

    Returns
    -------
    cycles : list of list
        Each element is a list of nodes forming a cycle.
    cycle_lengths : list of float
        Total length of each cycle.
    cycle_edges : list of list of (node, node)
        Edges in each cycle.
    """
    cycles = nx.cycle_basis(graph, root=base)
    cycle_lengths = []
    cycle_edges = []

    for cycle in cycles:
        total_length = 0.0
        edges = []
        for start, end in zip(cycle, cycle[1:] + cycle[:1]):
            if graph.has_edge(start, end):
                length = graph[start][end]["length"]
            else:
                length = graph[end][start]["length"]
            edges.append((start, end))
            total_length += length
        cycle_lengths.append(total_length)
        cycle_edges.append(edges)

    return cycles, cycle_lengths, cycle_edges
