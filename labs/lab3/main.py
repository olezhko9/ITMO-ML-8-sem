import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def read_data(path_to_csv):
    chips_df = pd.read_csv(path_to_csv)
    chips_df['class'] = chips_df['class'].map({'P': 1, 'N': -1})

    X = chips_df.drop('class', axis=1)
    y = chips_df['class']
    return X, y


def scatter(X, y, classifier):
    X_x, X_y = X.to_numpy()[:, 0], X.to_numpy()[:, 1]

    points_count = 200
    background_x, background_y = np.meshgrid(
        np.linspace(X_x.min(), X_x.max(), points_count),
        np.linspace(X_y.min(), X_y.max(), points_count)
    )
    background = np.array(list(zip(background_x.ravel(), background_y.ravel())))
    background_pred = classifier.predict(background)

    plt.scatter(background[:, 0], background[:, 1], c=['green' if c == 1 else 'red' for c in background_pred], s=1,
                alpha=0.2)

    plt.scatter(X_x, X_y, c=['green' if c == 1 else 'red' for c in y], s=30)
    plt.show()


def grid_search_cv(estimator, grid_params, X, y, scoring, verbose=False):
    params_list = list(grid_params.keys())
    best_score = 0.0
    best_params = {}
    if verbose:
        print(params_list)

    for params in (dict(zip(grid_params.keys(), values)) for values in product(*grid_params.values())):
        estimator.set_params(**params)
        estimator.fit(X, y)
        y_pred = estimator.predict(X)
        score = scoring(y, y_pred)

        if verbose:
            print(tuple(params.values()), score)

        if score > best_score:
            best_score = score
            for p_name in params_list:
                best_params[p_name] = params[p_name]

    return estimator.set_params(**best_params).fit(X, y), best_params, best_score


X, y = read_data('./dataset/chips.csv')

parameters = {
    'kernel': ['linear'],
    'gamma': np.arange(0.1, 1.0, 0.1),
    'C': [1, 3, 5, 10, 30, 50, 100, 200, 500]
}

clf, best_params, best_score = grid_search_cv(SVC(gamma='scale'), parameters, X, y, scoring=f1_score, verbose=False)
print(best_params, best_score)
scatter(X, y, clf)


parameters = {
    'kernel': ['poly'],
    'degree': [2, 3, 4, 5, 6],
    'C': [1, 3, 5, 10, 30, 50, 100, 200, 500]
}

clf, best_params, best_score = grid_search_cv(SVC(gamma='scale'), parameters, X, y, scoring=f1_score, verbose=False)
print(best_params, best_score)
scatter(X, y, clf)


parameters = {
    'kernel': ['rbf'],
    'gamma': np.arange(0.1, 1.0, 0.1),
    'C': [1, 3, 5, 10, 30, 50, 100, 200, 500]
}

clf, best_params, best_score = grid_search_cv(SVC(gamma='scale'), parameters, X, y, scoring=f1_score, verbose=False)
print(best_params, best_score)
scatter(X, y, clf)


X, y = read_data('./dataset/geyser.csv')

parameters = {
    'kernel': ['linear'],
    'gamma': np.arange(0.1, 1.0, 0.1),
    'C': [1, 3, 5, 10, 30, 50, 100, 200, 500]
}

clf, best_params, best_score = grid_search_cv(SVC(gamma='scale'), parameters, X, y, scoring=f1_score, verbose=False)
print(best_params, best_score)
scatter(X, y, clf)

parameters = {
    'kernel': ['poly'],
    'degree': [2, 3, 4, 5, 6],
    'C': [1, 5, 10, 50, 100, 500]
}

clf, best_params, best_score = grid_search_cv(SVC(gamma='scale'), parameters, X, y, scoring=f1_score, verbose=False)
print(best_params, best_score)
scatter(X, y, clf)


parameters = {
    'kernel': ['rbf'],
    'gamma': np.arange(0.1, 1.0, 0.1),
    'C': [1, 5, 10, 50, 100, 500]
}

clf, best_params, best_score = grid_search_cv(SVC(gamma='scale'), parameters, X, y, scoring=f1_score, verbose=False)
print(best_params, best_score)
scatter(X, y, clf)