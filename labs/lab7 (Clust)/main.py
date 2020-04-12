import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from KMeans import KMeans

data = pd.read_csv('wine.csv')

X = data[data.columns[1:]]
unique_classes = np.unique(data.loc[:, 'class'])
y = data.loc[:, 'class'].map(dict(zip(unique_classes, range(len(unique_classes)))))

X_norm = MinMaxScaler(copy=True).fit_transform(X)

X_pca = PCA(n_components=2).fit_transform(X_norm)

colours = ["b", "g", "r", "c", "m", "y", "k", "w"]


def display_clusters(labels, title):
    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        cur_xs = X_pca[labels == label, 0]
        cur_ys = X_pca[labels == label, 1]
        plt.scatter(cur_xs, cur_ys, color=colours[i], alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.legend()
    plt.show()


display_clusters(y, "Настоящие метки")

for i in range(1, len(colours) + 1):
    kMean = KMeans(k=i)
    kMean.fit(X_norm)
    y_pred = kMean.predict(X_norm)
    # print(y_pred)
    display_clusters(y_pred, '')
