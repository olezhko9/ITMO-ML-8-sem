import numpy as np
from copy import deepcopy


class KMeans:
    def __init__(self, k=2, eps=0.001, max_iter=100):
        self.k = k
        self.eps = eps
        self.max_iter = max_iter
        self.centroids = []

    def fit(self, data):
        for i in range(self.k):
            self.centroids.append(data[np.random.randint(0, data.shape[0])])

        for _ in range(self.max_iter):
            classifications = [[] for i in range(self.k)]

            for feature_set in data:
                distances = [np.linalg.norm(feature_set - self.centroids[k]) for k in range(self.k)]
                classification = np.argmin(distances)
                classifications[classification].append(feature_set)

            prev_centroids = deepcopy(self.centroids)

            for k in range(self.k):
                self.centroids[k] = np.average(classifications[k], axis=0)

            optimized = True

            for k in range(self.k):
                original_centroid = prev_centroids[k]
                current_centroid = self.centroids[k]
                if np.linalg.norm(current_centroid - original_centroid) > self.eps:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        predictions = []
        for feature_set in data:
            distances = [np.linalg.norm(feature_set - self.centroids[i]) for i in range(self.k)]
            predictions.append(np.argmin(distances))
        return predictions
