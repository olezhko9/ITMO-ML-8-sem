# N, M, K = map(int, input().split(' '))
# nums = list(map(int, input().split(' ')))

N, M, K = 10, 4, 3
nums = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]

groups = [[] for i in range(K)]
print(sorted(range(len(nums)), key=lambda k: nums[k]))

for i, n in enumerate(sorted(range(len(nums)), key=lambda k: nums[k])):
    groups[i % K].append(n + 1)


for g in groups:
    print(len(g), ' '.join(map(str, g)))

