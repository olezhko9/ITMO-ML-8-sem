# K = 2
# N = 4
#
# X = [
#     [1, 1],
#     [2, 2],
#     [3, 2],
#     [4, 1]
# ]

from io import BytesIO, IOBase
import os
import sys

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip("\r\n")

X = []
K = int(input())
N = int(input())
inter_dist, outer_dist = 0, 0

for i in range(N):
    X.append(list(map(int, input().split(' '))))


def internal(data):
    classes = [[] for i in range(K)]
    for x, y in data:
        classes[y - 1].append(x)

    res = 0
    for values in classes:
        values.sort()
        summa = 0
        for i in range(len(values)):
            summa += values[i]
            res += (values[i] * (i + 1) - summa)

    return res * 2


def outer(data):
    total_sum = 0
    class_sums = [0 for i in range(K)]
    class_counts = [0 for i in range(K)]

    data.sort()
    res = 0

    for i in range(len(data)):
        x, y = data[i]
        total_sum += x
        class_sums[y - 1] += x
        class_counts[y - 1] += 1
        res += ((i + 1 - class_counts[y - 1]) * x - total_sum + class_sums[y - 1])

    return res * 2


print(internal(X))
print(outer(X))
