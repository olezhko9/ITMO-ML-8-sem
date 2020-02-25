def manhattan(v1, v2):
    return sum([abs(a - b) for a, b in zip(v1, v2)])


def euclidean(v1, v2):
    return sum([(a - b) ** 2 for a, b in zip(v1, v2)]) ** (1 / 2)


def chebyshev(v1, v2):
    return max([abs(a - b) for a, b in zip(v1, v2)])


metrics = {
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "manhattan": manhattan,
}