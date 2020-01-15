import magnet
import networkx as nx
import matplotlib.pyplot as plt


G = nx.complete_graph(10000)
print("graph done")
X = magnet.fit_transform(G, n_jobs=2, epochs=5)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
