from .cutils import generate_walk
from multiprocessing import JoinableQueue, Process, Queue
from .transformation import (
    knn_graph, similarity_graph, radius_graph)
import networkx as nx
from tqdm import tqdm
import numpy as np
import time


# ==============================================
# Manifold learning and dimensionality reduction
# ==============================================

class MAGNET(object):
    def __init__(
        self,
        size=2,
        a=None, b=.33,
        min_dist=0, max_dist=1,
        kernel="tanh",
        p=.1, q=.1,
        num_walks=50, walk_len=50,
        optimizer="nadam",
        loss="mse",
        method="cpu"
    ):
        """
        :param size: Dimension of the embedding
        :param a: Parameter that controls some tightness of the resulting
                  manifold. The higher, the tighter
        :param b: Parameter that controls some tightness of the resulting
                  manifold. The lower, the tighter
        :param kernel: Similarity kernel to use. Can be `power`,
                    `gaussian` or `both`. Recommended is `power`
        :param p: Probability in a random walk to jump to a random node
        :param q: Probability to jump back to the starting node
        :param num_walks: Number of random walks starting from each node
        :param walk_len: Length of the random walks
        """
        # manifold-related parameters
        self.size = size
        self.a = a
        self.b = b
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.kernel = kernel
        # graph-related parameters
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_len = walk_len
        # keras model parameters
        self.optimizer = optimizer
        self.loss = loss
        self.method = method

    def fit_transform(
        self,
        G,
        init=None,
        epochs=5,
        batch_size=100,
        n_jobs=4,
        seed=1
    ):

        # create random walks on graph
        X, Y = self.create_random_walks(G, n_jobs=n_jobs)
        print(f"{len(X)} training samples")

        # define the initialization
        Z = None
        if init == 'spectral':
            Z = spectral_embedding(G, self.size)
        elif isinstance(init, np.ndarray):
            Z = init

        if self.method != "cpu":
            # train Keras model
            from .model import fit_model
            Z = fit_model(
                X, Y, Z, (len(G.nodes)),
                a=self.a, b=self.b,
                size=self.size,
                kernel=self.kernel,
                epochs=epochs,
                batch_size=batch_size,
                optimizer=self.optimizer,
                loss=self.loss,
                seed=seed)
        else:
            # train using numpy and numba
            from .cpu_model import train
            if Z is None:
                Z = np.random.normal(
                    size=(len(G.nodes), self.size)
                ).astype(np.float32)
            Y = Y.astype(np.float32)
            Z = train(
                X, Y, Z,
                size=self.size,
                b=self.b,
                batch_size=batch_size,
                epochs=epochs,
                momentum=.6,
                learning_rate=1e-2)
        return Z

    def knn_graph(
        self, X,
        n_neighbors=10,
        metric="euclidean",
        n_trees=20,
        directed=False
    ):
        G = knn_graph(X,
                      k=n_neighbors,
                      metric=metric,
                      n_trees=n_trees,
                      directed=False)
        return G

    def similarity_graph(self, X, metric="euclidean"):
        G = similarity_graph(X)
        return G

    def radius_graph(self, X, metric="euclidean", threshold=.1):
        G = radius_graph(X)
        return G

    # ------------
    # Random walks
    # ------------

    def create_random_walks(
        self, G, n_jobs=8
    ):
        """
        Creates random walks on the graph `G`.
        """
        num_nodes = len(G.nodes)
        id2node = list(G.nodes)
        node2id = {k: v for v, k in enumerate(id2node)}
        neighbors = {
            node2id[node]: [node2id[target] for target in G.neighbors(node)]
            for node in node2id}

        # compute weights (or rather distances already)
        weights = self.compute_weights(G, node2id)

        if n_jobs == 1:
            walks = one_job_walks(
                self.num_walks, self.walk_len, num_nodes,
                neighbors, self.p, self.q)
        else:
            walks, similarity = parallel_walks(
                G, node2id,
                self.num_walks, self.walk_len,
                neighbors, self.min_dist, self.max_dist, num_nodes,
                self.p, self.q, weights,
                n_jobs)
        return (walks, similarity)

    def compute_weights(self, G, node2id, as_distance=True):
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

            if as_distance:
                weight = (1 - weight) * self.max_dist + weight * self.min_dist

            weights[node_a + "_" + node_b] = weight

        elapsed_time = time.time() - start_time
        print(f"weights loaded - T={elapsed_time:.2f}s")
        return weights


def process_walks(
    queue,
    results,
    walk_len,
    num_nodes,
    neighbors,
    min_dist, max_dist,
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
            min_dist, max_dist,
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
    min_dist, max_dist,
    num_nodes,
    p, q,
    weights,
    n_jobs
):
    is_directed = nx.is_directed(G)

    start_time = time.time()
    results = Queue()
    queue = JoinableQueue()
    for i in range(num_walks):
        queue.put(i)

    for i in range(n_jobs):
        args = (queue, results, walk_len, num_nodes,
                neighbors, min_dist, max_dist,
                p, q, weights, is_directed)
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
    if a is None:
        a = 1.
    # print(min_dist, max_dist, kernel, a, b)
    if kernel == "power":
        min_sim = 1 / (1 + a * max_dist**b)
        max_sim = 1 / (1 + a * min_dist**b)
    elif kernel == "tanh":
        min_sim = 1 - np.tanh(a * max_dist**b)
        max_sim = 1 - np.tanh(a * min_dist**b)
    elif kernel == "gaussian":
        min_sim = np.exp(-a * max_dist**b)
        max_sim = np.exp(-a * min_dist**b)
    return min_sim, max_sim
