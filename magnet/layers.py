from keras.layers import Layer
from keras import backend as K


def get_distance_layer(kernel, batchn, walk_len, a, b, seed):
    if a is not None and b is not None:
        distance = DistanceSum(walk_len, a=a, b=b, kernel=kernel)
    elif b is None:
        distance = LearnBDistanceSum(walk_len, a=a, kernel=kernel)
    else:
        distance = LearnDistanceSum(walk_len, b=b, kernel=kernel)
    return distance(batchn)


class DistanceSum(Layer):
    def __init__(
        self,
        walk_len,
        a=1, b=1,
        kernel='power',
        **kwargs
    ):
        print("go here")
        self.walk_len = walk_len
        self.a = a
        self.b = b
        if kernel == 'power':
            self.kernel = self.power_kernel
        elif kernel == 'gaussian':
            self.kernel = self.gaussian_kernel
        elif kernel == "sigmoid":
            self.kernel = self.sigmoid_kernel
        elif kernel == "tanh":
            self.kernel = self.tanh_kernel
        else:
            raise ValueError(f"Kernel not found: {kernel}")
        super(DistanceSum, self).__init__(**kwargs)

    def power_kernel(self, distance, a):
        return K.pow(1 + a * distance, -1)

    def tanh_kernel(self, distance, a):
        return 1 - K.tanh(a * distance)

    def gaussian_kernel(self, distance, a):
        return K.exp(-a * distance)

    def sigmoid_kernel(self, distance, a):
        return 1 / (1 + K.exp(-a * distance - 3))

    def build(self, input_shape):
        super(DistanceSum, self).build(input_shape)

    def call(self, x):
        delta = K.square(x[:, 1:self.walk_len] - x[:, 0:self.walk_len - 1])
        distance = (K.sum(delta, axis=2, keepdims=False) + 1e-30) ** self.b
        # return self.kernel(distance, self.a)
        return distance

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1)


# Depecrated

class LearnDistanceSum(Layer):
    def __init__(self, walk_len, b=1, kernel='power', **kwargs):
        from keras import initializers

        self.init = initializers.get('zeros')
        self.walk_len = walk_len
        self.b = b
        if kernel == 'power':
            self.kernel = self.power_kernel
        elif kernel == 'gaussian':
            self.kernel = self.gaussian_kernel
        elif kernel == "sigmoid":
            self.kernel = self.sigmoid_kernel
        elif kernel == "tanh":
            self.kernel = self.tanh_kernel
        else:
            raise ValueError(f"Kernel not found: {kernel}")
        super(LearnDistanceSum, self).__init__(**kwargs)

    def power_kernel(self, distance, a):
        return K.pow(1 + a * distance, -1)

    def tanh_kernel(self, distance, a):
        return 1 - K.tanh(a * distance)

    def gaussian_kernel(self, distance, a):
        return K.exp(-a * distance)

    def sigmoid_kernel(self, distance, a):
        return 1 / (1 + K.exp(-a * distance - 3))

    def build(self, input_shape):
        self.W = self.add_weight(shape=(1,),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        super(LearnDistanceSum, self).build(input_shape)

    def call(self, x):
        a = 1 / (1 + K.exp(-self.W[0]))
        delta = K.square(x[:, 1:self.walk_len] - x[:, 0:self.walk_len - 1])
        distance = (K.sum(delta, axis=2, keepdims=False) + 1e-30) ** self.b
        # return self.kernel(distance, a)
        return a * distance

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1)


class LearnBDistanceSum(Layer):
    def __init__(self, walk_len, a=1, kernel='power', **kwargs):
        from keras import initializers

        self.init = initializers.get('zeros')
        self.walk_len = walk_len
        self.a = a
        if kernel == 'power':
            self.kernel = 0
        elif kernel == 'gaussian':
            self.kernel = 1
        else:
            self.kernel = 2
        super(LearnBDistanceSum, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(1,),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        super(LearnBDistanceSum, self).build(input_shape)

    def call(self, x):
        b = 1 / (1 + K.exp(-self.W[0]))

        delta = K.square(x[:, 1:self.walk_len] - x[:, 0:self.walk_len - 1])
        distance = (K.sum(delta, axis=2, keepdims=False) + 1e-12) ** b

        if self.kernel == 0:
            return K.pow(1 + distance, -1)
        elif self.kernel == 1:
            return K.exp(-distance)
        else:
            sim_g = K.exp(-distance)
            sim_p = K.pow(1 + distance, -1)
            return (sim_g + sim_p) / 2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1)
