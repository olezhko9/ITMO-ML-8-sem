from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np
import time

from knn import KnnRegressor


def leave_one_out_cv(regressor):
    y_pred = []

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(normalized_dataset):
        regressor.fit(normalized_dataset.iloc[train_index])

        test = np.array(normalized_dataset.iloc[test_index])[0][:-1]
        y_pred.append(np.round(regressor.predict(test)))

    return f1_score(y.to_numpy(), y_pred, average="weighted")


def grid_search_cv(regressor, grid_params):
    params_list = list(grid_params.keys())
    best_score = 0.0
    best_params = {}

    for params in (dict(zip(grid_params.keys(), values)) for values in product(*grid_params.values())):
                knn = regressor(**params)
                score = leave_one_out_cv(knn)
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
    plt.show()


if __name__ == '__main__':
    N = 500
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

    normalized_dataset = X.join(y)
    del X

    start = time.time()

    grid_params = {
        'metric': ['euclidean', 'manhattan'],
        'kernel': ['uniform', 'gaussian'],
        'win_type': ['variable'],
        'k': range(1, 30, 2),
    }

    best_params, best_score = grid_search_cv(KnnRegressor, grid_params)
    print(best_params)
    print(best_score)
    print(time.time() - start)

    best_params.pop('k', None)

    scores = []
    max_k = 30
    for k in range(1, max_k):
        knn = KnnRegressor(**best_params, k=k)
        scores.append(leave_one_out_cv(knn))

    plot([k for k in range(1, max_k)], scores)
