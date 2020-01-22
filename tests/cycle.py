import magnet
import networkx as nx
import matplotlib.pyplot as plt


G = nx.karate_club_graph()
# G = nx.complete_graph(10000)
print("graph done")
mgt = magnet.MAGNET(b=.1, max_dist=100, q=0)
X = mgt.fit_transform(G)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
