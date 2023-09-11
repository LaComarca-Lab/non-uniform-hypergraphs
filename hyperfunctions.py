"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Functions relevant for the non-uniform hypergraph centrality
"""

import copy
import numpy as np
import xgi
from itertools import permutations
from itertools import combinations

######## Ours ###########


def is_uniform(h):
    '''
    Given a Hypergraph h, returns if it's wether uniform or not.
    :param h :: Hypergraph:
    :return t :: boolean:
    '''
    return len(set([len(i) for i in h.edges.members()])) == 1


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




def uniform_adjacency_combinatorial_tensor(H, m = None, math_notation = True):
    '''
    Given a Hypergraph H, returns its adjacency tensor (with the Permutations with Repetions number corresponding to
    the number of phantom nodes added).
    If the hypergraph is not uniform or m != the dimension of the hyperedges, we uplift the lower-dimensional hyperedges
    by introducing a "phantom node" (indexed as N+1), and we project down the higher-dimensional hyperedges.
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

    if not xgi.is_uniform(H) and H.edges.size.max() <= m:
        # In case it isn't uniform AND we are not projecting, we node to add the phantom node
        N += 1

    shape = tuple(N for _ in range(m))
    # Insert edges in the tensor, multiplying them by their combinatorial factor
    aux_map = dict()
    for hyperedge in H.edges.members():

        if math_notation:
            hyperedge = {i - 1 for i in hyperedge}
            
        initial_len = len(hyperedge)
        edge = list(hyperedge) # convert to list to add phantom nodes (possibly more than 1)

        # Uplift adding an extra node enough times
        if len(edge) <= m:
            while len(edge) < m:
                edge.append(N - 1)
            perms = list(permutations(edge))

            # Combinatorial factor
            entry = np.math.factorial(initial_len)/np.math.factorial(len(edge))

        # Projection if higher dimensional
        else:
            perms = []
            for comb in combinations(edge,2):
                perms += list(permutations(comb))
            entry = 1/len(perms)
            
        # Add the permutation (uplift) / combination (projection) to the tensor
        for indices in perms:

            if indices in aux_map:
                aux_map[indices] += entry
            else:
                aux_map[indices] = entry

    return aux_map, shape


def HEC_ours(T, m=3, niter=100, tol=1e-6, verbose=True):
    '''
    Given a hypergraph (tensor, in the form of a sparse tensor, (python dictionary, shape)) T, we calculate it HEC using the power method.
    :param T :: (python dictionary, shape):
    :param m :: Integer (power):
    :param niter :: Integer (number of iterations):
    '''
    converged = False
    x = np.ones(T[1][0])
    y = apply(T, x)
    for i in range(niter):
        y_scaled = np.power(y, 1 / (m - 1))
        x = y_scaled / np.sum(y_scaled)
        y = apply(T, x)
        s = np.divide(y, np.power(x, m - 1))
        converged = (max(s) - min(s)) / min(s) < tol

        if converged and verbose:
            print('Finished in', i, 'iterations.')
            break

    return x, converged


def apply(T, x):
    '''
    Given an 3th order tensor T, contract it twice with vector x
    :param T :: (python dictionary, shape):
    :param x :: vector (centralities):
    :return y :: vector (T*x):
    '''
    # Product of a tensor and a vector, only taking the non-zero values (using a dictionary)
    y = np.zeros(x.shape[0])
    for edge, weight in T[0].items():
        ls = [node for node in edge]
        y[ls[0]] += weight * np.prod(x[[node for node in ls[1:]]])
        
    return y
