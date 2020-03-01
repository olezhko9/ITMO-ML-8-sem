import numpy as np
from pseudoinverse import PseudoinverseRegressor
from metric import nrmse, smape


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

train = np.array(train)
test = np.array(test)

X_train = np.append(train[:, :-1], np.ones((N, 1)), axis=1)
y_train = train[:, -1]

X_test = np.append(test[:, :-1], np.ones((K, 1)), axis=1)
y_test = test[:, -1]

print('train: ', train.shape)
print('test: ', test.shape)


pseudoinverseRegressor = PseudoinverseRegressor()
theta = pseudoinverseRegressor.fit(X_train, y_train)
res = pseudoinverseRegressor.predict(X_test)

print(nrmse(y_test, res))
print(smape(y_test, res))
