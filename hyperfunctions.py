"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Functions relevant for the non-uniform hypergraph centrality
"""

import copy
import numpy as np
import xgi
from itertools import permutations

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


def uniform_adjacency_combinatorial_tensor(H, m = None):
    '''
    Given a Hypergraph H, returns its adjacency tensor (with the Permutations with Repetions number corresponding to 
    the number of phantom nodes added). If the hypergraph is not uniform, we uniformize it first by adding an aditional
    dimension and an extra node (the one corresponding to this last dimension) the times we need
    :param h :: Hypergraph:
    :return t :: numpy.ndarray:
    '''
    assert isinstance(H, xgi.Hypergraph)
        
    dimension = len(H.nodes)
    h_dict = H.edges.members(dtype=dict)
    hyperdim = {edge: len(edgenodes) for edge, edgenodes in h_dict.items()}

    # Find maximum hyperedge dimension
    if not m:
        m = max(hyperdim.values())
    else:
        assert isinstance(m, int) and m >= max(hyperdim.values())
    if not is_uniform(h):
        # In case it isn't uniform, we node to add the phantom node, i.e, one more dimension
        dimension += 1
    shape = [dimension] * m
    T = np.zeros(shape)

    for i in H.edges.members():
        initial_len = len(i)
        edge = [k for k in i]
        while len(edge) < m:
            # We get here just in case is not uniform
            edge.append(dimension - 1)
        print(edge)
        perms = permutations(edge)
        entry = math.factorial(initial_len)/math.factorial(len(edge))
        for indices in perms:
            print(indices)
            T[indices] = entry
    return T

def apply_testing(T, x):
    '''
    Given an 3th order tensor T, contract it twice with vector x
    :param T :: Tensor (hypergraph):
    :param x :: vector (centralities):
    :return y :: vector (T*x):
    '''
    assert x.shape[0] == T.shape[0]
    # Initialize and sum accordingly
    y = np.zeros(x.shape[0])
    for i in range(T.shape[0]):
        for j in product(range(T.shape[0]), repeat = len(T.shape) - 1):
            aux = [n for n in j]
            aux.insert(0, i)
            aux = tuple(aux)
            y[i] += T[aux]*np.prod(x[[i for i in j]])
    return y

def HEC_ours(T, m=3, niter=2000, tol=1e-5, verbose=True):
    '''
    Chunk of code translated to Python from Julia (H_evec_NQI function) from
    https://github.com/arbenson/Hyper-Evec-Centrality/blob/master/centrality.jl
    '''
    converged = False
    x = np.array([1, 6, 5, 4, 3, 8])#np.ones(T.shape[0])
    print([1, 6, 5, 4, 3, 8])
    y = apply_testing(T, x)
    for i in range(niter):
        y_scaled = np.power(y, 1/(m - 1))   
        x = y_scaled / np.sum(y_scaled)
        y = apply_testing(T, x)
        s = np.divide(y, np.power(x, m - 1))
        converged = (max(s) - min(s)) / min(s) < tol
        
        if converged and verbose:
            print('Finished in', i, 'iterations.')
            break
            
    return x, converged
