

def fit_model(
    X, Y, Z=None,
    num_nodes=0,
    a=1, b=0.5,
    size=2,
    kernel='power',
    epochs=50, batch_size=100,
    optimizer='adamax',
    loss="mse",
    seed=1
):
    from keras.layers import Input, Embedding, BatchNormalization
    from keras.models import Model
    from keras.initializers import lecun_normal
    from .layers import get_distance_layer

    walk_len = X.shape[1]

    # layer specification
    inp = Input(shape=(walk_len,))
    if Z is not None:
        embedding = Embedding(num_nodes, size, weights=[Z])(inp)
    else:
        embedding = Embedding(
            num_nodes, size,
            embeddings_initializer=lecun_normal(seed))(inp)
    batchn = BatchNormalization()(embedding)
    distance = get_distance_layer(kernel, batchn, walk_len, a, b, seed)

    # build and compile model
    model = Model(inp, distance)
    model.compile(optimizer, loss)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    Z = model.layers[1].get_weights()[0]
    return Z
