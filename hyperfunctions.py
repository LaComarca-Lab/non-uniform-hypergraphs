"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Functions relevant for the non-uniform hypergraph centrality
"""

import copy
import numpy as np
import xgi

def uniformize(H, m=None):
    '''
    From  a Hypergraph H, make it uniform adding an artificial node connected in such a way
    that all hyperedges are of the same (maximum) dimension m.
    '''

    Hextra = copy.copy(H)

    # Find hyperedge dimensions dict
    h_dict = Hextra.edges.members(dtype=dict)
    hyperdim = {edge: len(edgenodes) for edge, edgenodes in h_dict.items()}

    # Find maximum hyperedge dimension
    if not m:
        m = max(hyperdim.values())
    else:
        assert isinstance(m, int) and m >= max(hyperdim.values())

    # Add a node to each edge which needs it
    for edge in Hextra.edges:

        # Count how many nodes the edge will need
        needed = m - hyperdim[edge]

        # Add the number needed as an attribute, and then add it.
        Hextra.edges[edge]['extra'] = needed
        if needed > 0:
            Hextra.add_node_to_edge(edge, '*')

    return Hextra