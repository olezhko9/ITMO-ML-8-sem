# K = 3
# CM = [
#     [4, 6, 3],
#     [1, 2, 0],
#     [1, 2, 6]
# ]

# K = 2
# CM = [
#     [0, 1],
#     [1, 3]
# ]
#
K = 3
CM = [
    [3, 1, 1],
    [3, 1, 1],
    [1, 3, 1],
]

# CM = []
# K = int(input())
# for j in range(K):
#     CM.append(list(map(int, input().split(' '))))


def f1_score(p, r):
    return 0 if p + r == 0 else 2 * (p * r) / (p + r)


row_sum = [sum(row) for row in CM]
column_sum = list(map(sum, zip(*CM)))
ALL = sum(row_sum)

prec_w = sum([0 if column_sum[i] == 0 else CM[i][i] * row_sum[i] / column_sum[i] for i in range(K)]) / ALL
recall_w = sum([CM[i][i] for i in range(K)]) / ALL

macro_f = f1_score(prec_w, recall_w)
print(macro_f)


micro_f = 0
for i in range(K):
    TP = CM[i][i]
    FP = row_sum[i] - CM[i][i]
    FN = column_sum[i] - CM[i][i]

    prec = 0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0 if TP + FN == 0 else TP / (TP + FN)

    micro_f += row_sum[i] * f1_score(prec, recall) / ALL

print(micro_f)
