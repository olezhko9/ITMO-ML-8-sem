import numpy as np
from sklearn.base import BaseEstimator


class NaiveBayesClassifier(BaseEstimator):
    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def fit(self, X, y):
        self.total_counts = np.sum(X, axis=0)
        self.labels = np.unique(y)

        self.word_probas = np.zeros((self.labels.shape[0], self.total_counts.shape[0]))
        self.label_probas = np.zeros(self.labels.shape[0])

        for label in self.labels:
            self.word_probas[label] = (np.sum(X[y == label], axis=0) + self.smooth) / (
                        self.smooth * X.shape[1] + np.sum(X[y == label]))
            self.label_probas[label] = float(y[y == label].shape[0]) / y.shape[0]

    def _log_proba(self, x, label):
        labels_weights = np.array([1, 1])
        return np.log(labels_weights[label] * self.label_probas[label]) + np.sum(np.log(self.word_probas[label][x != 0]))

    def predict_log_proba(self, X):
        label_probas = np.zeros((X.shape[0], self.labels.shape[0]))

        for i in np.arange(0, X.shape[0]):
            for label in self.labels:
                label_probas[i][label] = self._log_proba(X[i], label)

        return label_probas

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def predict_proba(self, X):
        label_log_probas = self.predict_log_proba(X)

        for i in np.arange(0, X.shape[0]):
            label_log_probas[i] = [
                1 / (1 + np.exp(label_log_probas[i][1] - label_log_probas[i][0])),
                1 / (1 + np.exp(label_log_probas[i][0] - label_log_probas[i][1]))
            ]

        return label_log_probas
