from keras import backend as K


def power_loss(y_true, y_pred):
    d_true = K.pow(y_true, -1)
    d_pred = K.pow(y_pred, -1)
    delta = d_true - d_pred
    return K.mean(K.square(delta), axis=-1)


def tanh_loss(y_true, y_pred):
    atanh_true = K.log((2 - y_true) / (y_true + 1e-12))
    atanh_pred = K.log((2 - y_pred) / (y_pred + 1e-12))
    delta = (atanh_true - atanh_pred)
    return K.mean(K.square(delta), axis=-1)


def gaussian_loss(y_true, y_pred):
    d_true = -K.log(y_true)
    d_pred = -K.log(y_pred)
    delta = (d_true - d_pred)
    return K.mean(K.square(delta), axis=-1)
