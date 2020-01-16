from keras.layers import Layer
from keras import backend as K


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


class LearnDistanceSum(Layer):
    def __init__(self, walk_len, b=1, kernel='power', **kwargs):
        from keras import initializers

        self.init = initializers.get('uniform')
        self.walk_len = walk_len
        self.b = b
        if kernel == 'power':
            self.kernel = 0
        elif kernel == 'gaussian':
            self.kernel = 1
        else:
            self.kernel = 2
        super(LearnDistanceSum, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(1,),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        super(LearnDistanceSum, self).build(input_shape)

    def call(self, x):
        a = K.exp(-K.abs(self.W[0]))

        delta = K.square(x[:, 1:self.walk_len] - x[:, 0:self.walk_len - 1])
        distance = (K.sum(delta, axis=2, keepdims=False) + 1e-12) ** self.b

        if self.kernel == 0:
            return K.pow(1 + a * distance, -1)
        elif self.kernel == 1:
            return K.exp(-a * distance)
        else:
            sim_g = K.exp(-a * distance)
            sim_p = K.pow(1 + a * distance, -1)
            return sim_g * sim_p

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1)
