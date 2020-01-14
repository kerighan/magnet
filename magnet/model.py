from keras.layers import Input, Embedding, Dense, BatchNormalization, Layer
from keras.models import Model
from keras import backend as K
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
        embedding = Embedding(num_nodes, size)(inp)
    batchn = BatchNormalization()(embedding)
    distance = DistanceSum((X.shape[1]), a=a, b=b, kernel=kernel)(batchn)

    model = Model(inp, distance)
    model.compile(optimizer, 'mse')
    for i in tqdm(range(epochs), 'epochs'):
        model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0)

    Z = model.layers[1].get_weights()[0]
    return Z


class DistanceSum(Layer):
    def __init__(self, walk_len, a=1, b=1, kernel='power', **kwargs):
        self.walk_len = walk_len
        self.a = a
        self.b = b
        if kernel == 'power':
            self.kernel = 0
        elif kernel == 'gaussian':
            self.kernel = 1
        else:
            self.kernel = 2
        super(DistanceSum, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DistanceSum, self).build(input_shape)

    def call(self, x):
        delta = K.square(x[:, 1:self.walk_len] - x[:, 0:self.walk_len - 1])
        distance = (K.sum(delta, axis=2, keepdims=False) + 1e-12) ** self.b
        if self.kernel == 0:
            return K.pow(1 + self.a * distance, -1)
        elif self.kernel == 1:
            return K.exp(-self.a * distance)
        else:
            sim_g = K.exp(-self.a * distance)
            sim_p = K.pow(1 + self.a * distance, -1)
            return sim_g * sim_p

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1)


def graph_to_vec(G):
    import matplotlib.pyplot as plt
    from graph2vec import Node2Vec
    import numpy as np
    g2v = Node2Vec(
        walklen=25,
        epochs=100,
        return_weight=1.0,
        neighbor_weight=1.0,
        threads=8,
        w2vparams={
            'window':10, 
            'size':2, 
            'negative':3, 
            'iter':20, 
            'ns_exponent':0.4, 
            'batch_words':25
        })
    g2v.fit(G)
    Z = np.zeros((len(G.nodes), 2))
    for i, node in enumerate(G.nodes):
        Z[i] = g2v.predict(node)

    plt.scatter(Z[:, 0], Z[:, 1])
    plt.show()
