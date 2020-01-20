from keras.layers import Input, Embedding, Dense, BatchNormalization
from keras.models import Model
from .layers import (
    DistanceSum, LearnDistanceSum, LearnBDistanceSum)
from .losses import power_loss, tanh_loss, gaussian_loss
from tqdm import tqdm


def fit_model(
    X, Y, Z=None,
    num_nodes=0,
    a=1, b=0.5,
    size=2,
    kernel='power',
    epochs=50, batch_size=100,
    optimizer='adamax'
):
    inp = Input(shape=(X.shape[1],))
    if Z is not None:
        embedding = Embedding(num_nodes, size, weights=[Z])(inp)
    else:
        embedding = Embedding(num_nodes, size,
                              embeddings_initializer="he_normal")(inp)
    batchn = BatchNormalization()(embedding)
    if a is not None and b is not None:
        distance = DistanceSum((X.shape[1]), a=a, b=b, kernel=kernel)(batchn)
    elif b is None:
        distance = LearnBDistanceSum((X.shape[1]), a=a, kernel=kernel)(batchn)
    else:
        distance = LearnDistanceSum((X.shape[1]), b=b, kernel=kernel)(batchn)

    model = Model(inp, distance)

    if kernel == "power":
        model.compile(optimizer, power_loss)
        # model.compile(optimizer, "mse")
    elif kernel == "tanh":
        model.compile(optimizer, tanh_loss)
    elif kernel == "gaussian":
        model.compile(optimizer, gaussian_loss)

    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    Z = model.layers[1].get_weights()[0]
    return Z
