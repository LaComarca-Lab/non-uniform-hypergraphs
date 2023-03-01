"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Functions relevant for the non-uniform hypergraph centrality
"""

import copy
import numpy as np
import xgi
from itertools import permutations
from collections import Counter

######## Ours ###########

def uniformize(H, m=None):
    '''
    From  a Hypergraph H, make it uniform adding an artificial node connected in such a way
    that all hyperedges are of the same (maximum) dimension m.
    '''

    Hextra = copy.deepcopy(H)

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


def uniform_adjacency_tensor(H):
    '''
    Given a Hypergraph H, returns its adjacency tensor.
    If the hypergraph is non-uniform, 
    we first uniformize (Hu) it adding a artificial node
    :param h :: Hypergraph:
    :return t :: numpy.ndarray:
    '''
    assert isinstance(H, xgi.Hypergraph)

    if not xgi.is_uniform(H):
        Hu = uniformize(H)
    
    dimension = len(Hu.nodes)
    m = len(Hu.edges.members()[0])
    
    shape = [dimension] * m
    T = np.zeros(shape)

    for edge in Hu.edges.members():
        
        if '*' in edge:
            edge.remove('*')
            edge.add(dimension-1)
        
        perms = permutations(edge)
        
        for indices in perms:
            T[indices] = 1
    
    return T



############ BENSON ################

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
    