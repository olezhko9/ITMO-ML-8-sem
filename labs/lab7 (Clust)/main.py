import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from metrics import contingency_matrix, adjusted_rand_index, silhouette
from KMeans import KMeans

data = pd.read_csv('wine.csv')

X = data[data.columns[1:]]
unique_classes = np.unique(data.loc[:, 'class'])
y = data.loc[:, 'class'].map(dict(zip(unique_classes, range(len(unique_classes)))))

X_norm = MinMaxScaler(copy=True).fit_transform(X)

X_pca = PCA(n_components=2).fit_transform(X_norm)


def display_clusters(labels, title):
    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        cur_xs = X_pca[labels == label, 0]
        cur_ys = X_pca[labels == label, 1]
        plt.scatter(cur_xs, cur_ys, alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.legend()
    plt.show()


display_clusters(y, "Настоящие метки")


def display_metrics(n_clusters, metrics, title):
    plt.figure(figsize=(8, 6))
    plt.grid(linestyle='--')
    plt.plot(n_clusters, metrics, linestyle='-', marker='.', color='r')
    plt.title(title)
    plt.xlabel("Количество кластеров")
    plt.ylabel("Значение метрики")
    plt.show()


external_metrics = []
internal_metrics = []
for i in range(1, 11):
    kMean = KMeans(k=i)
    centroids = kMean.fit(X_norm)
    y_pred = kMean.predict(X_norm)
    if i == 1:
        internal_metrics.append(0.0)
    else:
        internal_metrics.append(silhouette(X_norm, y_pred, centroids))

    external_metrics.append(adjusted_rand_index(y, y_pred))
    display_clusters(y_pred, str(i) + ' кластеров')

display_metrics(range(1, 11), external_metrics, 'Внешняя метрика')
display_metrics(range(1, 11), internal_metrics, 'Внутренняя метрика')
