import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score
import time
import json
from knn import KnnRegressor

if __name__ == '__main__':
    # distance_type = "euclidean"
    # kernel_type = "gaussian"
    window_type = "variable"
    # h = 10

    # load data
    dataset = pd.read_csv("abalone.csv")[:1000]

    V1 = {'F': 0, 'I': 1, 'M': 2}
    ClassLabels = {1: 0.0, 2: 1.0, 3: 2.0}
    dataset['V1'] = dataset['V1'].map(V1)
    dataset['Class'] = dataset['Class'].map(ClassLabels)

    X = dataset.drop(dataset.columns[-1], axis=1)
    y = dataset[dataset.columns[-1]]

    # normalize data
    columns = list(X.columns)
    columns_bins = {}
    for column in columns[1:]:
        X[column], bins = pd.qcut(X[column], 5, retbins=True, labels=False)
        columns_bins[column] = bins

    normalized_dataset = X.join(y)
    del X

    start = time.time()

    scores = {}
    for kernel_type in ['uniform', 'gaussian']:
        scores[kernel_type] = {}
        for distance_type in ['euclidean', 'manhattan', 'chebyshev']:
            scores[kernel_type][distance_type] = []
            for h in range(1, 30, 2):

                knn = KnnRegressor(distance_type, kernel_type, window_type, h)
                y_pred = []

                loo = LeaveOneOut()
                for train_index, test_index in loo.split(normalized_dataset):
                    knn.fit(normalized_dataset.iloc[train_index])

                    test = np.array(normalized_dataset.iloc[test_index])[0][:-1]
                    y_pred.append(np.round(knn.predict(test)))

                score = f1_score(y.to_numpy(), y_pred, average="weighted")

                print('%s : %s : %d = %.5f' % (kernel_type, distance_type, h, score))
                scores[kernel_type][distance_type].append((h, score))

    with open('scores.json', 'w') as fp:
        json.dump(scores, fp)

    print(time.time() - start)
