from keras import backend as K


def get_loss(kernel, loss, local):
    if local:
        def loss_func(x, y):
            return x - y
    elif kernel == "power":
        loss_func = power_loss
    elif kernel == "tanh":
        loss_func = tanh_loss
    elif kernel == "gaussian":
        loss_func = gaussian_loss

    if loss == "mse":
        def wrapper(y_true, y_pred):
            delta = loss_func(y_true, y_pred)
            return K.mean(K.square(delta), axis=-1)
        return wrapper
    elif loss == "mae":
        def wrapper(y_true, y_pred):
            delta = loss_func(y_true, y_pred)
            return K.mean(K.abs(delta), axis=-1)
        return wrapper
    elif loss == "logcosh":
        def wrapper(y_true, y_pred):
            delta = loss_func(y_true, y_pred)
            logcosh = delta + K.softplus(-2. * delta) - K.log(2.)
            return K.mean(logcosh, axis=-1)
        return wrapper
    elif loss == "huber":
        def wrapper(y_true, y_pred, d=1.):
            error = loss_func(y_true, y_pred)
            abs_error = K.abs(error)
            quadratic = K.minimum(abs_error, d)
            linear = abs_error - quadratic
            return 0.5 * K.square(quadratic) + d * linear
        return wrapper


def power_loss(y_true, y_pred):
    d_true = K.pow(y_true, -1)
    d_pred = K.pow(y_pred, -1)
    delta = d_true - d_pred
    return delta
    # return K.mean(K.square(delta), axis=-1)


def tanh_loss(y_true, y_pred):
    atanh_true = K.log((2 - y_true) / (y_true + 1e-12))
    atanh_pred = K.log((2 - y_pred) / (y_pred + 1e-12))
    delta = (atanh_true - atanh_pred)
    return delta
    # return K.mean(K.square(delta), axis=-1)


def gaussian_loss(y_true, y_pred):
    d_true = -K.log(y_true)
    d_pred = -K.log(y_pred)
    delta = (d_true - d_pred)
    return delta
    # return K.mean(K.square(delta), axis=-1)
