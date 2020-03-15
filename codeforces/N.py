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


from math import log

K1, K2 = list(map(int, input().split(" ")))
N = int(input())
p_x = [0 for i in range(K1)]
p_xy = {}

for i in range(N):
    x, y = list(map(int, input().split(' ')))
    p_x[x - 1] += 1 / N
    if p_xy.get((x, y)) is None:
        p_xy[(x, y)] = 0
    p_xy[(x, y)] += 1 / N

res = sum([-value * (log(value) - log(p_x[x - 1])) for (x, y), value in p_xy.items()])

print(res)
