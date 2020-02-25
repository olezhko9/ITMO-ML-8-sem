from util import kernels, metrics


class KnnRegressor:

    def __init__(self, metric="euclidean", kernel="uniform", win_type="fixed", k=5):
        self.data = []
        self.metric = metric
        self.kernel = kernel
        self.win_type = win_type
        self.k = k

    def fit(self, data):
        self.data = data.to_numpy()

    def predict(self, query):
        N = len(self.data)
        dists = []
        for row in self.data:
            d = metrics[self.metric](row[:-1], query)
            dists.append((d, row[-1]))

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
            y_weights.append(dist[1] * k)

        return sum(row[-1] for row in self.data) / N if sum(weights) == 0 else sum(y_weights) / sum(weights)
