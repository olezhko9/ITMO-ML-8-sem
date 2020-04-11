M = int(input())

f = []
for i in range(2**M):
    f.append(int(input()))

if sum(f) == 0:
    print(1)
    print(1)
    for i in range(M):
        print(0, end=' ')
    print(-0.5)
    exit()

print(2)
print(sum(f), 1)
for i in range(2**M):
    if f[i] == 0: continue

    mask = []
    for j in range(M):
        v = i % 2
        mask.append(v)
        print(1 if v else -1, end=' ')
        i //= 2
    print(0.5 - sum(mask))

for i in range(sum(f)):
    print(1, end=' ')
print(-0.5)
