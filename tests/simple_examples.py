import time
import magnet
import networkx as nx
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# init = None
fig, ax = plt.subplots(2, 2)
mgt = magnet.MAGNET(size=2, num_walks=100, walk_len=100, a=None, b=1, p=.1, q=.1, min_dist=0, max_dist=2, loss="logcosh", kernel="tanh")

# 1st graph
G = nx.cycle_graph(100)
X = mgt.fit_transform(G, epochs=10)
ax[0, 0].scatter(X[:, 0], X[:, 1])
ax[0, 0].title.set_text('Cycle graph')

# 2nd graph
G = nx.complete_graph(100)
X = mgt.fit_transform(G, epochs=10)
ax[0, 1].scatter(X[:, 0], X[:, 1])
ax[0, 1].title.set_text('Complete graph')

# 3rd graph
G = nx.star_graph(100)
# mgt = magnet.MAGNET(size=2, num_walks=100, walk_len=100, a=None, b=.5, p=.01, q=.01, max_dist=10, loss="huber", kernel="tanh")
X = mgt.fit_transform(G, epochs=10)
ax[1, 0].scatter(X[:, 0], X[:, 1])
ax[1, 0].title.set_text('Star graph')

# 4th graph
G = nx.barbell_graph(100, 10)
# mgt = magnet.MAGNET(size=2, num_walks=100, walk_len=100, a=None, b=.5, p=.01, q=.01, max_dist=10, loss="huber", kernel="tanh")
X = mgt.fit_transform(G, epochs=10)
ax[1, 1].scatter(X[:, 0], X[:, 1])
ax[1, 1].title.set_text('Barbell graph')

plt.show()
