K = int(input())
N = int(input())

data = []
for i in range(N):
    data.append(list(map(int, input().split())))

var_y2 = sum([y * y / N for x, y in data])

y_from_x = [[0, 0] for i in range(K)]

for x, y in data:
    y_from_x[x - 1][0] += y / N
    y_from_x[x - 1][1] += 1 / N

var2_y = sum([ey * ey / p if p != 0 else 0 for ey, p in y_from_x])

print(var_y2 - var2_y)
