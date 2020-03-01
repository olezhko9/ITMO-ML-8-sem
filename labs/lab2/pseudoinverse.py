import numpy as np


class PseudoinverseRegressor:
    def __init__(self):
        self.theta = None

    def fit(self, X_train, y_train):
        self.theta = np.dot(np.linalg.pinv(X_train), y_train)
        print('theta: ', self.theta.shape)

    def predict(self, X_test):
        if self.theta is not None:
            return np.dot(X_test, self.theta)
