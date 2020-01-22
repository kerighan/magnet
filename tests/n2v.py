import time
import magnet
import networkx as nx
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from convectors.models import WordVectors

# init = None
fig, ax = plt.subplots(2, 2)


def get_embedding(G):
    node2vec = Node2Vec(G, dimensions=2, walk_length=100, num_walks=100, workers=8)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    wv = WordVectors(model=model)
    return wv.weights


# 1st graph
G = nx.cycle_graph(100)
X = get_embedding(G)
ax[0, 0].scatter(X[:, 0], X[:, 1])
ax[0, 0].title.set_text('Cycle graph')

# 2nd graph
G = nx.complete_graph(100)
X = get_embedding(G)
ax[0, 1].scatter(X[:, 0], X[:, 1])
ax[0, 1].title.set_text('Complete graph')

# 3rd graph
G = nx.star_graph(100)
X = get_embedding(G)
ax[1, 0].scatter(X[:, 0], X[:, 1])
ax[1, 0].title.set_text('Star graph')

# 4th graph
G = nx.barbell_graph(100, 10)
X = get_embedding(G)
ax[1, 1].scatter(X[:, 0], X[:, 1])
ax[1, 1].title.set_text('Barbell graph')

plt.show()
