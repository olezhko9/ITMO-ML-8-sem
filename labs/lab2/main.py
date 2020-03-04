import numpy as np
from pseudoinverse import PseudoinverseRegressor
from metric import nrmse, smape


# reading data
train = []
test = []
with open('./dataset/2.txt', 'r') as f:
    M = int(f.readline())  # число признаков
    N = int(f.readline())  # число объектов в тренировочном наборе

    for i in range(N):
        train.append(list(map(int, f.readline().split(' '))))

    K = int(f.readline())  # число объектов в тестовом наборе

    for i in range(K):
        test.append(list(map(int, f.readline().split(' '))))

train = np.array(train, dtype='float64')
np.random.shuffle(train)
test = np.array(test, dtype='float64')

train = np.append(np.append(train[:, :-1], np.ones((N, 1)), axis=1), train[:, -1:], axis=1)
test = np.append(np.append(test[:, :-1], np.ones((K, 1)), axis=1), test[:, -1:], axis=1)

X_train = train[:, :-1]
y_train = train[:, -1:]

X_test = test[:, :-1]
y_test = test[:, -1:]

print('train: ', (X_train.shape, y_train.shape))
print('test: ', (X_test.shape, y_test.shape))


# PSEUDOINVERSE method
pseudoinverseRegressor = PseudoinverseRegressor()
theta = pseudoinverseRegressor.fit(X_train, y_train)
y_train_pred = pseudoinverseRegressor.predict(X_train)
y_test_pred = pseudoinverseRegressor.predict(X_test)

print('train nrmse: ', nrmse(y_train, y_train_pred))
print('test nrmse: ', nrmse(y_test, y_test_pred))
print('train smape: ', smape(y_train, y_train_pred))
print('test smape: ', smape(y_test, y_test_pred))

print(np.round(y_train[:18].flatten(), 0))
print(np.round(y_train_pred[:18].flatten(), 0))

print(np.round(y_test[:18].flatten(), 0))
print(np.round(y_test_pred[:18].flatten(), 0))
