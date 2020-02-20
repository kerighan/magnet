import numpy as np
from tqdm import tqdm
from numba import njit


@njit(fastmath=True)
def train_on_epoch(
    N, steps_per_batch, batch_size,
    num_nodes, size,
    X, Y, V, a=1, b=.5,
    learning_rate=1e-3, momentum=.9,
    focus=None, learn_a=True
):
    """Train on one epoch.

    Everything is in one function to speed up code
    and avoid unnecessary function calls.
    """
    #: shuffled index
    shuffle_index = np.arange(N)
    np.random.shuffle(shuffle_index)

    #: Nesterov Accelerated Gradient
    previous_update = np.zeros((num_nodes, size))

    #: init of gradient
    zeros = np.zeros((num_nodes, size), dtype=np.float32)

    #: length of random walks
    walk_len = Y.shape[1]

    # ==============
    # train on epoch
    # ==============
    for batch in range(steps_per_batch):
        # ==============
        # train on batch
        # ==============

        # get batch index
        start_index = batch * batch_size
        end_index = min((batch + 1) * batch_size, N)

        # initialize gradient for the current batch
        grad_a = 0.
        gradient = zeros.copy()

        # utils variables
        a_b = a**b
        batch_size = end_index - start_index

        #: nesterov lookahead
        V_momentum = V + momentum * previous_update
        for j in range(batch_size):
            # ==================
            # gradient on sample
            # ==================

            # shuffle data
            index = shuffle_index[start_index + j]
            indices = X[index]
            target = Y[index]

            # compute gradient of batch
            v_i = V_momentum[indices]
            delta = v_i[1:] - v_i[:-1]
            quad = np.sum(delta**2, axis=1)

            # accumulate gradient on samples
            last_index = indices[0]
            for i in range(walk_len):
                quad_i = quad[i]
                target_i = target[i]
                if quad_i > 0:
                    quad_i_b = quad_i**b
                    error_i = quad_i_b * a_b - target_i

                    # compute sample gradients
                    grad_value = -4 * (
                        a_b * b * delta[i] *
                        ((quad_i)**(b - 1)) * error_i)
                    grad_a += 2 * (
                        b * quad_i_b * (a**(b - 1)) * error_i)

                    # accumulate gradients
                    next_index = indices[i + 1]
                    gradient[last_index] += grad_value
                    gradient[next_index] -= grad_value
                    last_index = next_index  # lookup trick for speed

        # mask gradient
        if focus is not None:
            gradient *= focus

        # update `a`
        if learn_a:
            a -= 1e-5 * grad_a
            a = min(max(a, 1e-3), 1e3)

        # update weights
        update = momentum * previous_update - learning_rate * gradient
        V += update
        previous_update[:] = update[:]

    # return the updated factor value
    return a


def train(
    X, Y, V,
    size=2, b=.5, a=1,
    epochs=5, batch_size=32,
    learning_rate=1e-2, momentum=.6,
    focus=None, learn_a=True
):
    N, walk_len = X.shape
    steps_per_batch = N // batch_size + int(N % batch_size > 0)
    num_nodes = X.max() + 1

    # if focus, create gradient masking
    mask = None
    if focus is not None:
        mask = np.zeros((N, 1))
        for item in focus:
            mask[item] = 1

    # ===========
    # train model
    # ===========
    LR = learning_rate
    t = tqdm(range(epochs), desc="epochs", leave=True)
    for epoch in t:
        a = train_on_epoch(
            N, steps_per_batch, batch_size,
            num_nodes, size,
            X, Y, V,
            a=a, b=b,
            learning_rate=LR,
            momentum=momentum,
            learn_a=learn_a,
            focus=mask)
        LR = learning_rate * (1 - .25 * np.cos(np.pi * epoch / 5))
        t.set_description(f"epochs - {a:.6f}")
        t.refresh()
    return V
