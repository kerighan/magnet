import time
import magnet
import networkx as nx
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

init = None
fig, ax = plt.subplots(2, 2)

# 1st graph
G = nx.cycle_graph(100)
X = magnet.fit_transform(G, size=2, init=init)
ax[0, 0].scatter(X[:, 0], X[:, 1])
ax[0, 0].title.set_text('Cycle graph')

# 2nd graph
G = nx.complete_graph(100)
X = magnet.fit_transform(G, size=2, init=init)
ax[0, 1].scatter(X[:, 0], X[:, 1])
ax[0, 1].title.set_text('Complete graph')

# 3rd graph
G = nx.star_graph(100)
X = magnet.fit_transform(G, size=2, init=init, b=1, a=.5)
ax[1, 0].scatter(X[:, 0], X[:, 1])
ax[1, 0].title.set_text('Star graph')

# 4th graph
G = nx.barbell_graph(100, 10)
X = magnet.fit_transform(G, size=2, init=init, b=1)
ax[1, 1].scatter(X[:, 0], X[:, 1])
ax[1, 1].title.set_text('Barbell graph')

plt.show()
