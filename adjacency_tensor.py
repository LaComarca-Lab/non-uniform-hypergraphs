"""
Authors: Cristian Pérez Corral, Gonzalo Contreras Aso
Title: Functions relevant for the non-uniform hypergraph centrality
"""
import itertools
import xgi
import numpy as np



def is_uniform(h):
    '''
    Check if h is uniform. It returns True if all the hyperedges sizes are the same, and False if not.
    :param h :: Hypergraph:
    :return :
    '''
    return len(set([len(i) for i in h.edges.members()])) == 1

def uniform_adjacency_tensor(h):
    '''
    Given a Hypergraph h, returns it's tensor. If the hypergraph is not uniform, then first we uniformize it adding a artificial node
    :param h :: Hypergraph:
    :return t :: numpy.ndarray:
    '''
    assert isinstance(h, xgi.Hypergraph)
    if not is_uniform(h):
        h = uniformize(h)
    dimension = len(h.nodes)
    m = len(h.edges.members()[0])
    shape = [dimension for i in range(m)]
    t = np.ndarray(shape)
    t.fill(0)
    for i in h.edges.members():
        perms = itertools.permutations(i)
        for j in perms:
            t[j] = 1
    return t

if __name__ == '__main__':
    h = xgi.Hypergraph([[0, 1, 2], [0, 2, 3], [0, 3, 4], [1, 2, 4]])
    h1 = xgi.Hypergraph([[0, 1], [0, 2], [0, 3], [1, 2]])
    print(uniform_adjacency_tensor(h))
    print(uniform_adjacency_tensor(h1))