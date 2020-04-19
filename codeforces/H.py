import copy
import math


class Matrix:
    @staticmethod
    def dot(a, b):
        n = len(a)
        m = len(a[0])
        k = len(b[0])
        return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(k)] for i in range(n)]

    @staticmethod
    def transpose(a):
        return list(zip(*a))

    @staticmethod
    def add(a, b):
        m = len(a)
        n = len(a[0])
        return [[a[i][j] + b[i][j] for j in range(n)] for i in range(m)]

    @staticmethod
    def multiply(a, b):
        m = len(a)
        n = len(a[0])
        return [[a[i][j] * b[i][j] for j in range(n)] for i in range(m)]


class Node:
    def __init__(self):
        self.r = None
        self.c = None
        self.values = None
        self.in_df = None
        self.out_df = []

    def forward_pass(self):
        pass

    def back_propagation(self):
        if self.in_df: return
        self.in_df = [[0] * self.c for r in range(self.r)]
        for df_i in self.out_df:
            self.in_df = Matrix.add(self.in_df, df_i)


class VarNode(Node):
    def __init__(self, r, c):
        super().__init__()
        self.r = r
        self.c = c

    def forward_pass(self):
        pass

    def back_propagation(self):
        super().back_propagation()


class TnhNode(Node):
    def __init__(self, x: Node):
        super().__init__()
        self.x = x
        self.r = x.r
        self.c = x.c

    def forward_pass(self):
        self.values = list(map(lambda row: list(map(math.tanh, row)), self.x.values))

    def back_propagation(self):
        super().back_propagation()

        result = copy.deepcopy(self.in_df)
        for i in range(self.r):
            for j in range(self.c):
                x = self.values[i][j]
                result[i][j] *= 1 - x * x
        self.x.out_df.append(result)


class RluNode(Node):
    def __init__(self, a, x):
        super().__init__()
        self.a = a
        self.x = x
        self.r = x.r
        self.c = x.c

    def forward_pass(self):
        self.values = list(map(lambda row: list(map(lambda x: self.a * x if x < 0 else x, row)), self.x.values))

    def back_propagation(self):
        super().back_propagation()

        result = copy.deepcopy(self.in_df)
        for i in range(self.r):
            for j in range(self.c):
                x = self.x.values[i][j]
                result[i][j] *= self.a if x < 0 else 1
        self.x.out_df.append(result)


class MulNode(Node):
    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.a = a
        self.b = b
        self.r = a.r
        self.c = b.c

    def forward_pass(self):
        self.values = Matrix.dot(self.a.values, self.b.values)

    def back_propagation(self):
        super().back_propagation()
        self.a.out_df.append(Matrix.dot(self.in_df, Matrix.transpose(self.b.values)))
        self.b.out_df.append(Matrix.dot(Matrix.transpose(self.a.values), self.in_df))


class SumNode(Node):
    def __init__(self, u: list):
        super().__init__()
        self.u = u
        self.r = u[0].r
        self.c = u[0].c

    def forward_pass(self):
        self.values = copy.deepcopy(self.u[0].values)
        for u_i in self.u[1:]:
            self.values = Matrix.add(self.values, u_i.values)

    def back_propagation(self):
        super().back_propagation()
        for u_i in self.u:
            u_i.out_df.append(self.in_df)


class HadNode(Node):
    def __init__(self, u):
        super().__init__()
        self.u = u
        self.r = u[0].r
        self.c = u[0].c

    def forward_pass(self):
        self.values = copy.deepcopy(self.u[0].values)
        for u_i in self.u[1:]:
            self.values = Matrix.multiply(self.values, u_i.values)

    def back_propagation(self):
        super().back_propagation()

        for i in range(len(self.u)):
            u_i = self.u[i]
            product = copy.deepcopy(self.in_df)
            for j in range(len(self.u)):
                for r in range(self.r):
                    for c in range(self.c):
                        if i != j:
                            product[r][c] *= self.u[j].values[r][c]
            u_i.out_df.append(product)


class Network:
    node_funcs = {
        'var': VarNode,
        'tnh': TnhNode,
        'rlu': RluNode,
        'mul': MulNode,
        'sum': SumNode,
        'had': HadNode
    }

    def __init__(self, n, m, k):
        self.n = n
        self.m = m
        self.k = k
        self.nodes = []

    def add(self, node_type, **kwargs):
        if node_type == 'var':
            self.nodes.append(self.node_funcs[node_type](**kwargs))
        elif node_type == "tnh":
            self.nodes.append(self.node_funcs[node_type](x=self.nodes[kwargs['x']]))
        elif node_type == "rlu":
            self.nodes.append(self.node_funcs[node_type](a=1 / kwargs['a'], x=self.nodes[kwargs['x']]))
        elif node_type == "mul":
            self.nodes.append(self.node_funcs[node_type](a=self.nodes[kwargs['a']], b=self.nodes[kwargs['b']]))
        elif node_type == "sum":
            self.nodes.append(self.node_funcs[node_type](u=[self.nodes[u_i] for u_i in kwargs['u']]))
        elif node_type == "had":
            self.nodes.append(self.node_funcs[node_type](u=[self.nodes[u_i] for u_i in kwargs['u']]))

    def fit(self):
        for i in range(self.m):
            node = self.nodes[i]
            node.values = [list(map(float, input().split())) for _ in range(node.r)]

        for i in range(self.k):
            node = self.nodes[self.n - self.k + i]
            node.out_df.append([list(map(float, input().split())) for _ in range(node.r)])

    def run(self):
        for i in range(self.n):
            self.nodes[i].forward_pass()

        for i in range(self.n):
            self.nodes[-(i + 1)].back_propagation()

    def print(self):
        for i in range(self.k):
            for row in self.nodes[self.n - self.k + i].values:
                print(" ".join(map(str, row)))

        for i in range(self.m):
            for row in self.nodes[i].in_df:
                print(" ".join(map(str, row)))


n, m, k = map(int, input().split())
network = Network(n, m, k)

for i in range(n):
    line = input().split()
    node_type = line[0]

    if node_type == "var":
        network.add('var', r=int(line[1]), c=int(line[2]))
    elif node_type == "tnh":
        network.add('tnh', x=int(line[1]) - 1)
    elif node_type == "rlu":
        network.add('rlu', a=int(line[1]), x=int(line[2]) - 1)
    elif node_type == "mul":
        network.add('mul', a=int(line[1]) - 1, b=int(line[2]) - 1)
    elif node_type == "sum":
        network.add('sum', u=[int(u_i) - 1 for u_i in line[2:]])
    elif node_type == "had":
        network.add('had', u=[int(u_i) - 1 for u_i in line[2:]])

network.fit()
network.run()
network.print()
