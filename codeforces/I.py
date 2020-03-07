# N = 5
# X_arr = [1, 2, 3, 4, 5]
# Y_arr = [4, 5, 1, 2, 3]

X_arr = []
Y_arr = []

N = int(input())
for i in range(N):
    x, y = input().split(' ')
    X_arr.append(int(x))
    Y_arr.append(int(y))

X_mean = sum(X_arr) / N
Y_mean = sum(Y_arr) / N


numerator = sum([(X_arr[i] - X_mean) * (Y_arr[i] - Y_mean) for i in range(N)])
denominator = (sum([(X_arr[i] - X_mean)**2 for i in range(N)]) * sum([(Y_arr[i] - Y_mean)**2 for i in range(N)]))**0.5
r = numerator / denominator if denominator != 0 else 0

print(r)
