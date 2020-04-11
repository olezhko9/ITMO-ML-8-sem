import math

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


kernels = {
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