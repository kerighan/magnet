import time
import itertools
import numpy as np
import networkx as nx
from annoy import AnnoyIndex


def knn_graph(X, k=10, metric='euclidean', n_trees=20, directed=False):
    """
    Creates a k nearest-neighbor graph from cloud points.

    :param X: Points in a numpy array of size (N_samples, N_features)
    :param k: Max number of nearest neighbors include in the graph
    :param metric: Distance function used in the pairwise distance
    """
    from sklearn.metrics import pairwise_distances
    # start timer
    start = time.time()

    # compute pairwise distances and get nearest neighbors
    N = X.shape[0]
    dim = X.shape[1]

    index = AnnoyIndex(dim, metric)
    for i in range(N):
        index.add_item(i, X[i])
    index.build(n_trees)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    # edges = []
    G.add_nodes_from(range(N))
    for i in range(N):
        for neighbor in index.get_nns_by_item(i, k):
            if i != neighbor:
                # edges.append((i, neighbor))
                G.add_edge(i, neighbor)

    # G.add_edges_from(edges)

    elapsed = time.time() - start
    print(f"KNN Graph - T={elapsed:.2f}")
    return G


def radius_graph(X, metric='euclidean', threshold=.1):
    from scipy.spatial.distance import pdist, squareform

    N = X.shape[0]

    # compute pairwise distances and get nearest neighbors
    dist = pdist(X, metric=metric)
    dist_sort = np.sort(dist)
    distance_threshold = dist_sort[round(len(dist_sort) * threshold)]
    dist = squareform(dist)

    # build edges
    edges = []
    for i in range(N):
        for j in range(i):
            if dist[i, j] <= distance_threshold:
                edges.append((i, j))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    print(f"[*] Radius Graph - {len(edges)} edges")
    return G


def percentile_function(dist_flatten):
    """Creates an interpolated percentile function from an array of values."""
    from scipy.interpolate import interp1d
    y = np.arange(dist_flatten.shape[0]) / dist_flatten.shape[0]
    f = interp1d(dist_flatten, y)
    return f


def similarity_graph(X, metric="euclidean"):
    from sklearn.metrics import pairwise_distances

    N = X.shape[0]

    # compute pairwise distances and get nearest neighbors
    dist = pairwise_distances(X, metric=metric)
    min_dist = np.min(dist[np.nonzero(dist)])
    max_dist = np.max(dist[np.nonzero(dist)])

    edges = []
    for i in range(N):
        for j in range(i):
            weight = 1 - (dist[i, j] - min_dist) / max_dist
            edges.append((i, j, weight))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(edges)
    return G
