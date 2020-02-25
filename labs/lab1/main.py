from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
import time

from knn import KnnRegressor


def leave_one_out_cv(regressor, data, class_count):
    y_pred = []
    y_true = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(data):
        regressor.fit(data.iloc[train_index])

        test = np.array(data.iloc[test_index])[0][:-class_count]
        if class_count == 1:
            y_true.append(np.array(data.iloc[test_index])[0][-1])
            y_pred.append(np.round(regressor.predict(test)[0]))
        else:
            y_true.append(np.argmax(np.array(data.iloc[test_index])[0][-class_count:]))
            pred = regressor.predict(test)
            y_pred.append(np.argmax(pred))

    return f1_score(y_true, y_pred, average="weighted")


def grid_search_cv(regressor, grid_params, data, class_count):
    params_list = list(grid_params.keys())
    best_score = 0.0
    best_params = {}

    for params in (dict(zip(grid_params.keys(), values)) for values in product(*grid_params.values())):
                knn = regressor(**params, class_count=class_count)
                score = leave_one_out_cv(knn, data, class_count)
                # format_list = list(params.values())
                # format_list.append(score)
                # print('%12s : %12s : %10s : %3d = %7.5f' % tuple(format_list))

                if score > best_score:
                    best_score = score
                    for p_name in params_list:
                        best_params[p_name] = params[p_name]

    return best_params, best_score


def plot(neighbors_count, f_score):
    plt.plot(neighbors_count, f_score, color='blue', linestyle='solid', label='sin(x)')
    plt.xlabel('Neighbors count')
    plt.ylabel('F1 score')
    plt.show()


if __name__ == '__main__':
    N = 50
    # load data
    dataset = pd.read_csv("abalone.csv")[:N]

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

    # NAIVE
    normalized_dataset = X.join(y)

    start = time.time()

    k_range = range(1, 15, 1)
    grid_params = {
        'metric': ['euclidean', 'manhattan'],
        'kernel': ['uniform', 'gaussian'],
        'win_type': ['variable'],
        'k': k_range,
    }

    best_params, best_score = grid_search_cv(KnnRegressor, grid_params, normalized_dataset, class_count=1)
    print(best_params)
    print(best_score)
    print(time.time() - start)

    best_params.pop('k', None)

    scores = []
    for k in k_range:
        knn = KnnRegressor(**best_params, k=k, class_count=1)
        scores.append(leave_one_out_cv(knn, normalized_dataset, class_count=1))

    plot([k for k in k_range], scores)

    # ONE HOT
    y = pd.get_dummies(y, dtype=float)
    normalized_dataset = X.join(y)
    del X

    start = time.time()

    k_range = np.arange(0.0, 8.0, 0.5)
    grid_params = {
        'metric': ['euclidean', 'manhattan'],
        'kernel': ['uniform', 'gaussian'],
        'win_type': ['fixed'],
        'k': k_range,
    }

    best_params, best_score = grid_search_cv(KnnRegressor, grid_params, normalized_dataset, class_count=3)
    print(best_params)
    print(best_score)
    print(time.time() - start)

    best_params.pop('k', None)

    scores = []
    for k in k_range:
        knn = KnnRegressor(**best_params, k=k, class_count=3)
        scores.append(leave_one_out_cv(knn, normalized_dataset, class_count=3))

    plot([k for k in k_range], scores)
