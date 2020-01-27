# from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from annoy import AnnoyIndex

import seaborn as sns
import numpy as np
import magnet
import time

sns.set(context="paper", style="white")

# mnist = fetch_openml("mnist_784", version=1)
# np.save("datasets/mnist_data.npy", mnist.data)
# color = mnist.target.astype(int)
# np.save("datasets/mnist_color.npy", color)

N = 70000
data = np.load("datasets/mnist_data.npy")[:N]
color = np.load("datasets/mnist_color.npy")[:N]

start = time.time()
reducer = magnet.MAGNET(
    a=None, b=.3, max_dist=6,
    kernel="tanh", num_walks=25, walk_len=50,
    loss="logcosh",
    p=0.05, q=0.05)
G = reducer.knn_graph(data, n_neighbors=10, n_trees=5, directed=False)
embedding = reducer.fit_transform(
    G, epochs=2, batch_size=250, n_jobs=8, init=None)

elapsed = time.time() - start
print(elapsed)

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by MAGNET", fontsize=18)

plt.show()
