"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Other hypergraph centrality measures defined in the literature
"""

import copy
import numpy as np
import xgi
from itertools import permutations
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


def vector_centrality(H):
    '''Compute the vector centrality of nodes in hypergraphs
    :param H :: xgi.Hypergraph:
    :return vc :: dict:
    '''

    # Construct the line graph and compute its eigenvector centrality 
    LG = line_graph(H)
    LGcent = nx.eigenvector_centrality(LG)

    # Initialize the vector centrality dictionary
    vc = {node: [] for node in H.nodes}

    # Get hyperedge indices and dimensions
    edge_label_dict = {tuple(edge):index for index, edge in enumerate(H.edges.members())}
    hyperedge_dims = {tuple(edge):len(edge) for edge in H.edges.members()} 

    # Maximum dimension of the hyperedges
    D = np.max(list(hyperedge_dims.values()))

    for k in range(2, D+1):
        c_i = np.zeros(len(H.nodes))

        # Hyperedges of dimension K
        for edge, _ in list(filter(lambda x: x[1] == k, hyperedge_dims.items())):
        
            # Sum the centrality contribution to its hyperedges
            for node in edge:
                try:
                    c_i[node] += LGcent[edge_label_dict[edge]]
                except IndexError:
                    raise Exception('Nodes must be written with the Pythonic indexing (0,1,2...)')
        # Weight
        c_i *= 1/k

        # Append to dictionary
        for node in H.nodes:
            vc[node].append(c_i[node])

    return vc


################# Benson's non-uniformity proposal ##################
# From "Three hypergraph eigenvector centralities"

def repeated_perms(li, m):
    '''Given a list with unique elements, return the set of lists obtained from it
    by duplicating any entries to reach length m lists, and their permutations. 
    '''

    unique = len(li)
    to_add = m - unique
    
    assert to_add > 0 and max(Counter(li).values()) == 1 and isinstance(li,list)

    goodperms = set()
    for perm in permutations(li * (to_add+1), m):
        if len(Counter(perm).values()) == unique:
            goodperms.add(perm)
    
    return goodperms


def uniform_adjacency_tensor_Benson(H):
    '''
    Given a non-uniform Hypergraph H, returns its adjacency tensor,
    as defined by Benson in the conclusions of this paper.
    :param h :: Hypergraph:
    :return t :: numpy.ndarray:
    '''
    assert isinstance(H, xgi.Hypergraph)

    if xgi.is_uniform(H):
        raise Exception('Use the uniform_adjacency_tensor() funcion')
    
    # Obtain the list of hyperedge lengths
    ms = [len(he) for he in H.edges.members()]

    # Initialize a tensor with the order of the maximum hyperedge
    shape = [len(H.nodes)] * max(ms)  
    T = np.zeros(shape)
    
    for he in H.edges.members():
        
        repeat = max(ms) - len(he)
        
        if repeat == 0:
            goodperms = permutations(he)
        else:
            goodperms = repeated_perms(list(he), max(ms))
        
        for indices in goodperms:
            T[indices] = 1
    
    return T
    