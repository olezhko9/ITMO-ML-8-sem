# N, M = 3, 2
# train = [
#     [0, 2, 1],
#     [1, 1, 0],
#     [2, 0, 1]
# ]
#
# query = [0, 0]
# distance_type = "euclidean"
# kernel_type = "gaussian"
# window_type = "variable"
# h = 2

import math

N, M = map(int, input().split(' '))

train = []
for i in range(N):
    train.append(list(map(int, input().split(' '))))

query = list(map(int, input().split(' ')))
distance_type = input()
kernel_type = input()
window_type = input()
h = int(input())


def manhattan(v1, v2):
    return sum([abs(a - b) for a, b in zip(v1, v2)])


def euclidean(v1, v2):
    return sum([(a - b) ** 2 for a, b in zip(v1, v2)]) ** (1 / 2)


def chebyshev(v1, v2):
    return max([abs(a - b) for a, b in zip(v1, v2)])


distance_f = {
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "manhattan": manhattan,
}


def uniform(u):
    if abs(u) < 1:
        return 1 / 2
    return 0


def triangular(u):
    if abs(u) < 1:
        return 1 - abs(u)
    return 0


def epanechnikov(u):
    if abs(u) < 1:
        return 3 / 4 * (1 - u * u)
    return 0


def quartic(u):
    if abs(u) < 1:
        return 15 / 16 * (1 - u * u) ** 2
    return 0


def triweight(u):
    if abs(u) < 1:
        return 35 / 32 * (1 - u * u) ** 3
    return 0


def tricube(u):
    if abs(u) < 1:
        return 70 / 81 * (1 - abs(u) ** 3) ** 3
    return 0


def gaussian(u):
    return 1 / (math.sqrt(2 * math.pi)) * math.e ** (-0.5 * u * u)


def cosine(u):
    if abs(u) < 1:
        return math.pi / 4 * math.cos(math.pi / 2 * u)
    return 0


def logistic(u):
    return 1 / (math.e ** u + 2 + math.e ** (-u))


def sigmoid(u):
    return (2 / math.pi) * (1 / (math.e ** u + math.e ** (-u)))


kernel_f = {
    "uniform": uniform,
    "triangular": triangular,
    "epanechnikov": epanechnikov,
    "quartic": quartic,
    "triweight": triweight,
    "tricube": tricube,
    "gaussian": gaussian,
    "cosine": cosine,
    "logistic": logistic,
    "sigmoid": sigmoid,
}

dists = []
for row in train:
    d = distance_f[distance_type](row[:-1], query)
    dists.append((d, row[-1]))

y_weights = []
weights = []
dists = sorted(dists, key=lambda x: x[0])

for dist in dists:
    if dist[0] != 0 and (h == 0 or (window_type == "variable" and dists[h][0] == 0)):
        k = 0
    else:
        if window_type == "fixed":
            u = dist[0] / h if h != 0 else 0
        elif window_type == "variable":
            u = dist[0] / dists[h][0] if dists[h][0] != 0 else 0

        k = kernel_f[kernel_type](u)

    weights.append(k)
    y_weights.append(dist[1] * k)

print("%.10f" % (sum(row[-1] for row in train) / N if sum(weights) == 0 else sum(y_weights) / sum(weights)))


