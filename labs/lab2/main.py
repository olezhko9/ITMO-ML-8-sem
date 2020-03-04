import numpy as np
import matplotlib.pyplot as plt
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
from pseudoinverse import PseudoinverseRegressor

pseudoinverseRegressor = PseudoinverseRegressor()
theta = pseudoinverseRegressor.fit(X_train, y_train)

y_train_pred = pseudoinverseRegressor.predict(X_train)

print('train nrmse: ', nrmse(y_train, y_train_pred))
print('train smape: ', smape(y_train, y_train_pred))
print(np.round(y_train[:18].flatten(), 0))
print(np.round(y_train_pred[:18].flatten(), 0))

y_test_pred = pseudoinverseRegressor.predict(X_test)

print('test nrmse: ', nrmse(y_test, y_test_pred))
print('test smape: ', smape(y_test, y_test_pred))
print(np.round(y_test[:18].flatten(), 0))
print(np.round(y_test_pred[:18].flatten(), 0))


# # GRADIENT DESCENT
from gradient_descent import GradientDescentRegressor

GDRegressor = GradientDescentRegressor(lr=5e-9, max_iter=2000, eps=1e-10)
_, err = GDRegressor.fit(X_train, y_train)

plt.plot(err)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

y_train_pred = GDRegressor.predict(X_train)

print('train nrmse: ', nrmse(y_train, y_train_pred))
print('train smape: ', smape(y_train, y_train_pred))
print(np.round(y_train[:18].flatten(), 0))
print(np.round(y_train_pred[:18].flatten(), 0))

y_test_pred = GDRegressor.predict(X_test)

print('test nrmse: ', nrmse(y_test, y_test_pred))
print('test smape: ', smape(y_test, y_test_pred))
print(np.round(y_test[:18].flatten(), 0))
print(np.round(y_test_pred[:18].flatten(), 0))


# ANNEALING
from annealing import annealing

weights, err = annealing(X_train, y_train, lr=0.2)

plt.plot(err)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

y_train_pred = X_train @ weights

print('train nrmse: ', nrmse(y_train, y_train_pred))
print('train smape: ', smape(y_train, y_train_pred))
print(np.round(y_train[:18].flatten(), 0))
print(np.round(y_train_pred[:18].flatten(), 0))

y_test_pred = X_test @ weights

print('test nrmse: ', nrmse(y_test, y_test_pred))
print('test smape: ', smape(y_test, y_test_pred))
print(np.round(y_test[:18].flatten(), 0))
print(np.round(y_test_pred[:18].flatten(), 0))