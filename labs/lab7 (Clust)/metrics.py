import numpy as np


def contingency_matrix(true_labels, pred_labels):
    cm = [[0 for j in range(len(np.unique(pred_labels)))] for i in range(len(np.unique(true_labels)))]
    for p, l in zip(pred_labels, true_labels):
        cm[l][p] += 1
    return cm


def adjusted_rand_index(true_labels, pred_labels):
    cm = contingency_matrix(true_labels, pred_labels)
    column_sums = np.sum(cm, axis=0)
    row_sums = np.sum(cm, axis=1)
    n = sum(row_sums)

    def c(v):
        return v * (v - 1) / 2

    index = sum([c(v) for row in cm for v in row])
    expected_index = sum([c(a) for a in row_sums]) * sum([c(b) for b in column_sums]) / c(n)
    max_index = 1 / 2 * (sum([c(a) for a in row_sums]) + sum([c(b) for b in column_sums]))

    return (index - expected_index) / (max_index - expected_index)


def silhouette(X, pred_labels, centroids):
    clusters, counts = np.unique(pred_labels, return_counts=True)
    n_clusters = len(clusters)

    clusters_X = [np.array([]) for i in range(n_clusters)]
    for cluster in clusters:
        clusters_X[cluster] = X[pred_labels == cluster, :]

    silhouettes = []
    for cluster in clusters:
        internal_distance = np.mean([np.linalg.norm(x - centroids[cluster]) for x in clusters_X[cluster]])

        external_distances = []
        for other_cluster in clusters:
            if other_cluster != cluster:
                distances = [np.linalg.norm(x - centroids[other_cluster]) for x in clusters_X[cluster]]
                external_distances.append(np.mean(distances) if len(distances) else 0.0)

        external_distance = min(external_distances) if len(external_distances) else 0.0
        silhouettes.append((external_distance - internal_distance) / max([internal_distance, external_distance]))

    return sum(silhouettes) / n_clusters
