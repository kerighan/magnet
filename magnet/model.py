from keras.layers import Input, Embedding, Dense, BatchNormalization
from keras.models import Model
from .layers import DistanceSum, LearnDistanceSum, LearnBDistanceSum
from tqdm import tqdm


def fit_model(
    X, Y, Z=None,
    num_nodes=0,
    a=1, b=0.5,
    size=2,
    kernel='power',
    epochs=50, batch_size=100,
    optimizer='rmsprop'
):
    inp = Input(shape=(X.shape[1],))
    if Z is not None:
        embedding = Embedding(num_nodes, size, weights=[Z])(inp)
    else:
        embedding = Embedding(num_nodes, size, embeddings_initializer="he_normal")(inp)
    batchn = BatchNormalization()(embedding)
    if a is not None and b is not None:
        distance = DistanceSum((X.shape[1]), a=a, b=b, kernel=kernel)(batchn)
    elif b is None:
        distance = LearnBDistanceSum((X.shape[1]), a=a, kernel=kernel)(batchn)
    else:
        distance = LearnDistanceSum((X.shape[1]), b=b, kernel=kernel)(batchn)

    model = Model(inp, distance)
    model.compile(optimizer, 'mse')
    for i in tqdm(range(epochs), 'epochs'):
        model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0)

    Z = model.layers[1].get_weights()[0]
    return Z
