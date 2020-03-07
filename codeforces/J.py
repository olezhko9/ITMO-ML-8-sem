# N = 5
# X = [
#     [1, 16],
#     [2, 25],
#     [3, 1],
#     [4, 4],
#     [5, 9]
# ]
#
# X_arr = [1, 2, 3, 4, 5]
# Y_arr = [16, 25, 1, 4, 9]

X = []
X_arr = []
Y_arr = []

N = int(input())
for i in range(N):
    line = input().split(' ')
    X.append(list(map(int, line)))
    x, y = line
    X_arr.append(int(x))
    Y_arr.append(int(y))

X_arr = {value: index + 1 for index, value in enumerate(sorted(X_arr))}
Y_arr = {value: index + 1 for index, value in enumerate(sorted(Y_arr))}

d = 0
for i in range(N):
    d += ((X_arr.get(X[i][0])) - (Y_arr.get(X[i][1])))**2


if N == 1:
    print(1)
else:
    r = 1 - 6 * d / (N * (N*N - 1))
    print(r)


