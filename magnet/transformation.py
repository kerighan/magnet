import numpy as np
import networkx as nx


def knn_graph(X, k=10, metric='euclidean', threshold=None, weighted=True):
    """
    Creates a k nearest-neighbor graph from cloud points.

    :param X: Points in a numpy array of size (N_samples, N_features)
    :param k: Max number of nearest neighbors include in the graph
    :param metric: Distance function used in the pairwise distance
    :param threshold: If not `None`, this parameter between 0 and 1
                      is used to avoid adding edges to distances above
                      the `threshold` percentile (expressed in ratio).
                      e.g.: if threshold = 0.1, it means that no
                      edge will be created if the distance between two
                      nodes is higher than the 10% smallest distinct
                      pairwise distances.
    :param weighted: If `True`, the graph is weighted by the scaled inverse
                     quantile value of the distance between two nodes.
                     Scaled because the value is compared to :param threshold:
                     and its value does not exceed 1
    """
    from sklearn.metrics import pairwise_distances

    N = X.shape[0]

    # compute pairwise distances and get nearest neighbors
    dist = pairwise_distances(X, metric=metric)
    neighbors = np.argsort(dist, axis=1)[:, 1:k + 1]

    if weighted:
        from .cutils import weighted_edges_from_neighbors
        dist_flatten = np.sort(dist.flatten())[N - 1:]
        percentile = percentile_function(dist_flatten)
        threshold = 1 if threshold is None else threshold
        G = nx.Graph()
        G.add_nodes_from(range(N))
        edges = weighted_edges_from_neighbors(
            neighbors, N, dist, percentile, threshold)
        G.add_weighted_edges_from(edges)
        print(f"[*] KNN Graph - {len(edges)} edges")
        return G
    elif threshold is not None:
        from .cutils import edges_from_neighbors_threshold
        dist_flatten = np.sort(dist.flatten())
        index = min(N + int(threshold * (N * N)), N * N - 1)
        percentile = dist_flatten[index]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        edges = edges_from_neighbors_threshold(neighbors, N, dist, percentile)
        G.add_edges_from(edges)
        print(f"[*] KNN Graph - {len(edges)} edges")
        return G
    else:
        from .cutils import edges_from_neighbors
        G = nx.Graph()
        G.add_nodes_from(range(N))
        edges = edges_from_neighbors(neighbors, N)
        G.add_edges_from(edges)
        print(f"[*] KNN Graph - {len(edges)} edges")
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
