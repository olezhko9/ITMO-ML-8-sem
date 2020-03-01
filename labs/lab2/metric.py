import numpy as np


def nrmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual - predicted))) / (actual.max() - actual.min())


def smape(y_true, y_pred):
    return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
