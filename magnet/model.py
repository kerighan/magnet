from keras.layers import Input, Embedding, Dense, BatchNormalization
from keras.models import Model
from .layers import (
    DistanceSum, LearnDistanceSum, LearnBDistanceSum)
from .losses import get_loss
from tqdm import tqdm


def fit_model(
    X, Y, Z=None,
    num_nodes=0,
    a=1, b=0.5,
    size=2,
    kernel='power',
    epochs=50, batch_size=100,
    optimizer='adamax',
    loss="mse",
    local=False
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
    model.compile(optimizer, get_loss(kernel, loss, local))
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    Z = model.layers[1].get_weights()[0]
    return Z
