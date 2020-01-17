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

embedding = magnet.project(
    mnist.data[:30000], size=2,
    n_neighbors=50, num_walks=25, kernel="tanh",
    batch_size=2000, a=1, b=.3, label_smoothing=False)
color = mnist.target.astype(int)[:30000]

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by MAGNET", fontsize=18)

plt.show()
