from tqdm import tqdm
from threading import Thread
import numpy as np
cimport numpy as np
cimport cython
import random


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef random_step(dict neighbors, list last_step, list id2node, dict node2id, int num_nodes, float p=.1, float q=.1):
    cdef list nn
    cdef list step = []
    cdef int j
    for j in range(num_nodes):
        node = last_step[j]
        rand = random.random()
        if rand <= p:
            next_node = random.randint(0, num_nodes - 1)
        elif rand <= p + q:
            next_node = j
        else:
            nn = neighbors[id2node[node]]
            next_node = node2id[random.choice(nn)]
        step.append(next_node)
    return step


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_walk(int walk_len, dict neighbors, list id2node, dict node2id, int num_nodes, float p=.1, float q=.1):
    cdef list steps = [list(range(num_nodes))]
    cdef int i
    for i in range(walk_len - 1):
        step = random_step(neighbors,
                           steps[i],
                           id2node,
                           node2id,
                           num_nodes,
                           p, q)
        steps.append(step)
    return steps


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_similarity(np.ndarray walks, np.ndarray Adj, int walk_len):
    cdef int walks_length = len(walks)
    cdef np.ndarray similarity = np.zeros(
        (walks.shape[0], walks.shape[1] - 1),
        dtype=np.float16)

    cdef Py_ssize_t i
    cdef Py_ssize_t j

    cdef list x_tuple = []
    cdef list y_tuple = []
    cdef list a_tuple = []
    cdef list b_tuple = []
    for i in range(walk_len - 1):
        a_tuple += list(walks[:, i])
        b_tuple += list(walks[:, i + 1])
        for j in range(walks_length):
            x_tuple.append(j)
            y_tuple.append(i)

    similarity[x_tuple, y_tuple] = Adj[a_tuple, b_tuple]
    return similarity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_sparse_similarity(np.ndarray walks, Adj, int walk_len):
    cdef int walks_length = len(walks)
    cdef np.ndarray similarity = np.zeros(
        (walks.shape[0], walks.shape[1] - 1),
        dtype=np.float16)

    cdef Py_ssize_t i
    cdef Py_ssize_t j

    cdef list x_tuple = []
    cdef list y_tuple = []
    cdef list a_tuple = []
    cdef list b_tuple = []
    for i in range(walk_len - 1):
        a_tuple += list(walks[:, i])
        b_tuple += list(walks[:, i + 1])
        for j in range(walks_length):
            x_tuple.append(j)
            y_tuple.append(i)
    
    similarity[x_tuple, y_tuple] = Adj[a_tuple, b_tuple]
    return similarity


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef edges_from_neighbors(np.ndarray neighbors, int N):
    cpdef list edges = []
    cpdef int neighbor
    cpdef int i
    for i in range(N):
        for neighbor in neighbors[i]:
            # don't count the edges two times
            if neighbor < i:
                edges.append((i, neighbor))
    return edges


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef edges_from_neighbors_threshold(np.ndarray neighbors, int N, np.ndarray dist, float threshold):
    cpdef list edges = []
    cpdef int neighbor
    cpdef int i
    for i in range(N):
        for neighbor in neighbors[i]:
            # don't count the edges two times
            if neighbor < i and dist[i, neighbor] <= threshold:
                edges.append((i, neighbor))
    return edges


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef weighted_edges_from_neighbors(np.ndarray neighbors, int N, np.ndarray dist, percentile, float threshold):
    cpdef list edges = []
    cpdef int neighbor
    cpdef int i
    cpdef float value
    mininimum = 6000
    for i in range(N):
        for neighbor in neighbors[i]:
            if neighbor < i:
                value = percentile(dist[i, neighbor])
                if value <= threshold:
                    # value = (value - (1 - threshold)) / threshold
                    # edges.append((i, neighbor, 1 - value))
                    value = (threshold - value) / threshold
                    edges.append((i, neighbor, value))
    return edges
