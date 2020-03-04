import numpy as np
from metric import smape, nrmse


class GradientDescentRegressor:
    def __init__(self, lr=1e-9, max_iter=1000, penalty='l2', a=1, eps=1e-10):
        self.lr = lr
        self.max_iter = max_iter
        self.penalty = penalty
        self.a = a
        self.eps = eps

        self.w = None
        self.X = None
        self.y = None

    def _gradient(self, y_pred):
        l = len(self.X)
        return (2 / l) * self.X.T.dot(y_pred - self.y)

    def _cost(self, y_pred):
        return nrmse(self.y, y_pred)

    def fit(self, X, y):
        self.X = X
        self.y = y

        error_list = []
        self.w = np.zeros((X.shape[1], 1))

        for it in range(self.max_iter):
            y_pred = self.X.dot(self.w)

            if self.penalty == 'l1':
                self.w = self.w - self.lr * (self._gradient(y_pred) - 1 / self.a * np.sign(self.w))
            else:
                self.w = self.w - self.lr * (self._gradient(y_pred) - 1 / self.a * self.w)

            error_list.append(self._cost(y_pred))
            if it >= 2 and np.abs(error_list[-1] - error_list[-2]) < self.eps:
                break

        return self.w, error_list

    def predict(self, X_test):
        return X_test @ self.w
