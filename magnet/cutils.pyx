from tqdm import tqdm
from threading import Thread
import numpy as np
cimport numpy as np
cimport cython
import random


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef random_step(dict neighbors, list last_step, int num_nodes, float min_dist, float max_dist, float p=.1, float q=.1, dict weights={}, int is_directed=False):
    cdef list nn
    cdef list step = []
    cdef list sim = []
    cdef float weight
    cdef int j
    for j in range(num_nodes):
        node = last_step[j]
        rand = random.random()
        if rand <= p:
            next_node = random.randint(0, num_nodes - 1)
        elif rand <= p + q:
            next_node = j
        else:
            nn = neighbors[node]
            if len(nn) == 0:
                next_node = random.randint(0, num_nodes - 1)
            else:
                next_node = random.choice(nn)
        step.append(next_node)

        # append weight
        if node == next_node:
            sim.append(0)
        else:
            node_str = str(node)
            next_node = str(next_node)
            if is_directed:
                weight = weights.get(node_str + "_" + next_node, max_dist)
                sim.append(weight)
            else:
                weight = weights.get(
                    node_str + "_" + next_node, 
                    weights.get(next_node + "_" + node_str, max_dist))
                sim.append(weight)
    return step, sim


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_walk(int walk_len, int num_nodes, dict neighbors, float min_dist, float max_dist, float p=.1, float q=.1, dict weights={}, int is_directed=False):
    cdef list steps = [list(range(num_nodes))]
    cdef list similarity = []
    cdef int i
    for i in range(walk_len - 1):
        step, sim = random_step(neighbors,
                                steps[i],
                                num_nodes,
                                min_dist, max_dist,
                                p, q,
                                weights, is_directed)
        steps.append(step)
        similarity.append(sim)
    return steps, similarity
