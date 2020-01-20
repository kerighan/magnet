from .cutils import random_step, generate_walk, compute_similarity, compute_sparse_similarity, compute_graph_similarity
from multiprocessing import JoinableQueue, Process, Queue
from .transformation import knn_graph, radius_graph
import networkx as nx
from tqdm import tqdm
import numpy as np
import time

# ----------------------------------------------
# Manifold learning and dimensionality reduction
# ----------------------------------------------


def fit_transform(
    G,
    size=2,
    num_walks=25,
    walk_len=50,
    a=1,
    b=0.5,
    p=0.1,
    q=0.1,
    min_dist=0,
    max_dist=1.,
    kernel='power',
    label_smoothing=True,
    init=None,
    sparse=True,
    epochs=5,
    batch_size=100,
    n_jobs=4,
    optimizer="nadam"
):
    """
    Transforms a graph into a numpy matrix containing embedding of vertices,
    sorted in the same order as they appear in when iterating over the nodes.

    :param G: Networx Graph of any type
    :param size: Dimension of the embedding
    :param num_walks: Number of random walks starting from each node
    :param walk_len: Length of the random walks
    :param a: Parameter that controls some tightness of the resulting manifold
              The higher, the tighter
    :param b: Parameter that controls some tightness of the resulting manifold
              The lower, the tighter
    :param p: Probability in a random walk to jump to a random node
    :param q: Probability in a random walk to jump back to the starting node
    :param kernel: Similarity kernel to use. Can be `power`,
                   `gaussian` or `both`. Recommended is `power`
    :param label_smoothing: Boolean. Label smoothing is avoiding certainty
                            in the training data
    :param init: Can be `spectral` or a numpy matrix of size (N_nodes, size)
    :param sparse: Controls sparsity of the graph adjacency matrix
    :param epochs: Number of epochs
    :param batch_size: Size of the batch of the SGD
    :param n_jobs: Number of processes building randomwalks
    """

    from .model import fit_model
    X, Y = create_random_walks(
        G,
        num_walks=num_walks, walk_len=walk_len,
        p=p, q=q, sparse=sparse, n_jobs=n_jobs)

    if label_smoothing:
        min_sim, max_sim = get_clip(min_dist, max_dist, kernel, a, b)
        print(min_sim, max_sim)
        Y = np.clip(Y, min_sim, max_sim)

    if init == 'spectral':
        Z = spectral_embedding(G, size)
    elif isinstance(init, np.ndarray):
        Z = init
    else:
        Z = None

    print(f"{len(X)} samples")
    embeddings = fit_model(
        X, Y, Z, (len(G.nodes)),
        a=a, b=b,
        size=size,
        kernel=kernel,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer)
    return embeddings


def project(
    X,
    size=2,
    n_neighbors=10,
    metric="euclidean",
    threshold=None,
    weighted=False,
    num_walks=25,
    walk_len=40,
    a=1, b=0.3,
    p=0.1, q=0.1,
    kernel='power',
    label_smoothing=True,
    method="knn",
    init=None,
    sparse=True,
    epochs=5,
    batch_size=200,
    n_jobs=4
):
    if method == "knn":
        G = knn_graph(X, k=n_neighbors, metric=metric,
                      threshold=threshold, weighted=weighted)
    elif method == "radius":
        G = radius_graph(X, metric=metric, threshold=threshold)

    return fit_transform(
        G,
        size,
        num_walks,
        walk_len,
        a, b,
        p, q,
        kernel,
        label_smoothing,
        init,
        sparse,
        epochs,
        batch_size,
        n_jobs)


# ------------
# Random walks
# ------------

def create_random_walks(
    G, num_walks=25, walk_len=100,
    p=0.1, q=0.1,
    n_jobs=8,
    sparse=True
):
    """
    Creates random walks on the graph `G`.
    """
    num_nodes = len(G.nodes)
    id2node = list(G.nodes)
    node2id = {k: v for v, k in enumerate(id2node)}
    neighbors = {node2id[node]:
                 [node2id[target] for target in G.neighbors(node)]
                 for node in node2id}

    if n_jobs == 1:
        walks = one_job_walks(num_walks, walk_len, num_nodes,
                              neighbors, p, q)
    else:
        walks, similarity = parallel_walks(
            G, node2id, num_walks, walk_len,
            neighbors, num_nodes, p, q,
            n_jobs)

    return (walks, similarity)


def process_walks(
    queue,
    results,
    walk_len,
    num_nodes,
    neighbors,
    p, q,
    weights,
    is_directed,
    dtype=np.uint32
):
    """Unit function for a multiprocessing instance."""
    while True:
        if queue.empty():
            break
        queue.get()
        steps, sim = generate_walk(
            walk_len, num_nodes,
            neighbors,
            p, q,
            weights,
            is_directed)
        results.put((
            np.array(steps, dtype=dtype).T,
            np.array(sim, dtype=np.float16).T))
        queue.task_done()


def parallel_walks(
    G, node2id,
    num_walks,
    walk_len,
    neighbors,
    num_nodes,
    p, q,
    n_jobs
):
    weights = compute_weights(G, node2id)
    is_directed = nx.is_directed(G)

    start_time = time.time()
    results = Queue()
    queue = JoinableQueue()
    for i in range(num_walks):
        queue.put(i)

    for i in range(n_jobs):
        args = (queue, results, walk_len, num_nodes,
                neighbors, p, q, weights, is_directed)
        thread = Process(target=process_walks, args=args)
        thread.daemon = True
        thread.start()

    queue.join()

    def dump_queue(q):
        """ Dump a Queue to a list """
        w = []
        s = []
        while 1:
            if q.qsize() > 0:
                walk, sim = q.get()
                w.append(walk)
                s.append(sim)
            else:
                break
        return w, s

    walks, similarity = dump_queue(results)
    walks = np.vstack(walks)
    similarity = np.vstack(similarity)
    elapsed_time = time.time() - start_time
    print(f"random walks built - T={elapsed_time:.2f}s")
    return walks, similarity


def one_job_walks(
    num_walks,
    walk_len, num_nodes,
    neighbors,
    p, q,
    dtype=np.uint32
):
    walks = []
    for i in tqdm(range(num_walks), 'random walks'):
        steps = generate_walk(walk_len, neighbors, num_nodes, p, q)
        walks.append(np.array(steps, dtype=dtype).T)

    walks = np.vstack(walks)
    return walks


def compute_weights(G, node2id):
    start_time = time.time()
    is_weighted = nx.is_weighted(G)

    weights = {}
    for a, b in G.edges:
        node_a = str(node2id[a])
        node_b = str(node2id[b])

        if is_weighted:
            weight = G[a][b]["weight"]
        else:
            weight = 1

        weights[node_a + "_" + node_b] = weight

    elapsed_time = time.time() - start_time
    print(f"weights loaded - T={elapsed_time:.2f}s")
    return weights


# -----
# Utils
# -----

def wordvectors_from(G, Z):
    """Ignore (used in a proprietary NLP package.)"""
    from convectors.models import WordVectors
    node2id = {}
    for i, node in enumerate(G.nodes):
        node2id[node] = i

    wv = WordVectors(word2id=node2id, weights=Z)
    return wv


def spectral_embedding(G, size=2):
    """Spectral embedding initialization."""
    start_time = time.time()
    num_nodes = len(G.nodes)
    pos = nx.spectral_layout(G, dim=size)
    Z = np.zeros((num_nodes, size))
    for i, node in enumerate(G.nodes):
        Z[i] = pos[node]

    elapsed_time = time.time() - start_time
    print(f"spectral embedding - T={elapsed_time:.2f}s")
    return Z


def get_clip(min_dist, max_dist, kernel, a, b):
    if kernel == "power":
        min_sim = 1 / (1 + a * max_dist**b)
        max_sim = 1 / (1 + a * min_dist**b)
    elif kernel == "tanh":
        min_sim = 1 - np.arctanh(a * max_dist**b)
        max_sim = 1 - np.arctanh(a * min_dist**b)
    elif kernel == "gaussian":
        min_sim = np.exp(-a * max_dist**b)
        max_sim = np.exp(-a * min_dist**b)
    return min_sim, max_sim
