from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import magnet

sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)
# print(mnist.data)

# np.save("datasets/mnist.npy", mnist.data)
# data = np.load("datasets/mnist.npy")[:10000]

reducer = magnet.MAGNET()
G = reducer.knn_graph(mnist.data)
embedding = reducer.fit_transform(G)

color = mnist.target.astype(int)[:30000]

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by MAGNET", fontsize=18)

plt.show()
