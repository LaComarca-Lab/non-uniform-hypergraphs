"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Other hypergraph centrality measures defined in the literature
"""

import copy
import numpy as np
import xgi
from itertools import permutations
from itertools import combinations
from collections import Counter

######################### Vector centrality #########################
# From "Vector Centrality in Networks with Higher-Order Interactions"

def line_graph(H):
    '''Constructs the line graph of a hypergraph
    :param H :: xgi.Hypergraph:
    :return LG :: nx.Graph:
    ''' 
    
    # Initialize the line graph and its nodes (associated to hyperedges)
    LG = nx.Graph()

    edge_label_dict = {tuple(edge):index for index, edge in enumerate(H.edges.members())}
    LG.add_nodes_from(edge_label_dict.values())
    
    # Connect two nodes if the intersection of their hyperedges is not empty
    for edge1, edge2 in combinations(H.edges.members(), 2):
        if edge1.intersection(edge2):
            LG.add_edge(edge_label_dict[tuple(edge1)], edge_label_dict[tuple(edge2)])

    return LG


######## Benson ###########

def increase_edge(edgeset):
    '''
    Given some edges, returns all edges whose length is one higher with those same nodes
    :param edgeset :: set of edges:
    :return new_edgeset :: modified set of edges:
    '''
    
    new_edgeset = set()
    counterlist = [] # all counters must be different, otherwise we don't add the node
    for edge in edgeset:
        
        edge = list(edge)
        
        # Add each node in the edge to itself, increasing in 1 the length
        for node in edge:
            
            copyedge = copy.copy(edge)
            copyedge.append(node)
            
            # Check for duplicates (permuted edges already with the same node count within)
            if Counter(copyedge) not in counterlist: 
                new_edgeset.add(tuple(copyedge))
                
                counterlist.append(Counter(copyedge))
            
    return new_edgeset


def alternative_uniformization(H, m=None, math_notation=True):
    '''
    Given a Hypergraph H, returns its adjacency tensor.
    If the hypergraph is not uniform or m != the dimension of the hyperedges, we duplicate the existing indices of smaller hyperedges, 
    with suitable weight, and we project down higher hyperedges.
    :param h :: Hypergraph:
    :param math_notation :: Boolean (wether the first node starst at 0 or 1):
    :return t :: (python dictionary, shape):
    '''
    N = len(H.nodes)
    # Find maximum hyperedge dimension
    if not m:
        m = H.edges.size.max()
    else:
        assert isinstance(m, int)
    
    # Product of p_i's (occurrences of each node) on an edge
    psprod = lambda edge: np.prod([np.math.factorial(p_i) for p_i in Counter(edge).values()])
    
    shape = tuple(N for _ in range(m))
    # Insert edges in the tensor, multiplying them by their combinatorial factor
    aux_map = dict()
    for hyperedge in H.edges.members():

        if math_notation:
            hyperedge = {i - 1 for i in hyperedge}
        
        initial_len = len(hyperedge)
        edge = tuple(hyperedge) # convert to list to add auxiliary nodes (possibly more than 1)

        edgeset = {edge}
        # Uplift adding an extra node enough times
        
        # Use this uniformization to add existing nodes enough times 
        if len(edge) <= m:
            
            # Increase up to the desired size
            while len(list(edgeset)[0]) < m:
                edgeset = increase_edge(edgeset)
            
            # Calculate the alpha factor for the combinatorial factor
            alpha = np.sum([np.math.factorial(len(edge))/psprod(edge) for edge in edgeset])

            # Combinatorial factor
            weight = len(edge)/alpha
            
            # Get all permutations of all increased hyperedges
            perms = []
            for edge in edgeset:
                perms += list(permutations(edge))
                        
        # Projection if higher dimensional (same as in the UPHEC case)
        else:
            perms = []
            for comb in combinations(edge, m):
                perms += list(permutations(comb))
            weight = 1
            
        # Add the permutation (uplift) / combination (projection) to the tensor
        for indices in perms:

            if indices in aux_map:
                aux_map[indices] += weight
            else:
                aux_map[indices] = weight

    return aux_map, shape