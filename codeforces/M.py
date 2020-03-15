# K1, K2 = 2, 3
# N = 5
#
# X = [
#     [1, 2],
#     [2, 1],
#     [1, 1],
#     [2, 2],
#     [1, 3],
# ]

K1, K2 = list(map(int, input().split(" ")))
N = int(input())

f1 = [0 for i in range(K1)]
f2 = [0 for i in range(K2)]
contingency = {}

for i in range(N):
    x, y = list(map(int, input().split(' ')))
    f1[x - 1] += 1 / N
    f2[y - 1] += 1 / N
    if contingency.get((x, y)) is None:
        contingency[(x, y)] = 0
    contingency[(x, y)] += 1

res = N
for (x, y), observed in contingency.items():
    expected = N * f1[x - 1] * f2[y - 1]
    res += (observed - expected)**2 / expected - expected

print(res)
