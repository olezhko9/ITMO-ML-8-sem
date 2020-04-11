from util import kernels, metrics
import numpy as np


class KnnRegressor:

    def __init__(self, metric="euclidean", kernel="uniform", win_type="fixed", k=5, class_count=1):
        self.data = []
        self.metric = metric
        self.kernel = kernel
        self.win_type = win_type
        self.k = k
        self.class_count = class_count

    def fit(self, data):
        self.data = data.to_numpy()

    def predict(self, query):
        N = len(self.data)
        dists = []
        for row in self.data:
            d = metrics[self.metric](row[:-self.class_count], query)
            dists.append((d, row[-self.class_count:]))

        y_weights = []
        weights = []
        dists = sorted(dists, key=lambda x: x[0])

        for dist in dists:
            if dist[0] != 0 and (self.k == 0 or (self.win_type == "variable" and dists[self.k][0] == 0)):
                k = 0
            else:
                if self.win_type == "fixed":
                    u = dist[0] / self.k if self.k != 0 else 0
                elif self.win_type == "variable":
                    u = dist[0] / dists[self.k][0] if dists[self.k][0] != 0 else 0

                k = kernels[self.kernel](u)

            weights.append(k)
            y_weights.append(np.array(dist[1]) * k)

        res = []
        if sum(weights) == 0:
            for i in range(1, self.class_count + 1):
                res.insert(0, sum(row[-i] for row in self.data) / N)
        else:
            for i in range(1, self.class_count + 1):
                res.insert(0, sum([w[-i] for w in y_weights]) / sum(weights))

        return res
