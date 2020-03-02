import numpy as np
from pseudoinverse import PseudoinverseRegressor
from metric import nrmse, smape

# reading data
train = []
test = []
with open('./dataset/1.txt', 'r') as f:
    M = int(f.readline())  # число признаков
    N = int(f.readline())  # число объектов в тренировочном наборе

    for i in range(N):
        train.append(list(map(int, f.readline().split(' '))))

    K = int(f.readline())  # число объектов в тестовом наборе

    for i in range(K):
        test.append(list(map(int, f.readline().split(' '))))

train = np.array(train, dtype='float64')
test = np.array(test, dtype='float64')

X_train = np.append(train[:, :-1], np.ones((N, 1)), axis=1)
y_train = train[:, -1]

X_test = np.append(test[:, :-1], np.ones((K, 1)), axis=1)
y_test = test[:, -1]

print('train: ', (X_train.shape, y_train.shape))
print('test: ', (X_test.shape, y_test.shape))


# PSEUDOINVERSE method
pseudoinverseRegressor = PseudoinverseRegressor()
theta = pseudoinverseRegressor.fit(X_train, y_train)
y_pred = pseudoinverseRegressor.predict(X_test)

print('nrmse: ', nrmse(y_test, y_pred))
print('smape: ', smape(y_test, y_pred))

print(y_test[:18].flatten())
print(y_pred[:18].flatten())
