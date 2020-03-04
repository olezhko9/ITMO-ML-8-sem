import numpy as np
from metric import nrmse, smape


def annealing(X, y, lr=0.1):
    error_list = []
    t = 1.0
    weights = np.array([np.random.uniform(X[:, i].min(), X[:, i].max()) for i in range(len(X[0]))], copy=True)
    while t > 0.1:

        e_old = nrmse(y, X @ weights)

        new_weights = weights.copy()
        for i in range(len(X[0])):
            dir = np.random.choice([-1, 1])
            new_weights[i] = new_weights[i] - (dir * lr * new_weights[i])

        e_new = nrmse(y, X @ new_weights)

        if e_new < e_old or np.exp(-(e_new - e_old) / t) > np.random.rand():
            weights = new_weights.copy()
            e = e_new
        else:
            e = e_old

        error_list.append(e)
        print(t, '-----', e)
        t *= 0.99

    return weights, error_list

