import numpy as np

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

F = train[:, :-1]
y = train[:, -1]

print('train: ', train.shape)
print('test: ', test.shape)
print('F: ', F.shape)
print('y: ', y.shape)

O = np.dot(np.linalg.pinv(F), y)
print('O: ', O.shape)


def nrmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual - predicted))) / (actual.max() - actual.min())


def smape(y_true, y_pred):
    return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


F_test = test[:, :-1]
y_test = test[:, -1]

res = np.dot(F_test, O)
print(nrmse(y_test, res))
print(smape(y_test, res))
