# 3
# 1 1 1
# 1
# 4
# 1 2 ant emu
# 2 3 dog fish dog
# 3 3 bird emu ant
# 1 3 ant dog bird
# 5
# 2 emu emu
# 5 emu dog fish dog fish
# 5 fish emu ant cat cat
# 2 emu cat
# 1 cat

from math import log, exp

K = int(input())
penalty = list(map(int, input().split(' ')))
a = int(input())
N = int(input())

y_count = {i: 0 for i in range(K)}
y_words = {i: dict() for i in range(K)}
all_words = set()

for i in range(N):
    line = input().split(' ')
    y = int(line[0]) - 1
    y_count[y] += 1
    for word in set(line[2:]):
        all_words.add(word)
        if y_words[y].get(word) is None:
            y_words[y][word] = 1
        else:
            y_words[y][word] += 1


def p(word, y):
    if y_words[y].get(word) is None:
        y_words[y][word] = 0

    return (y_words[y][word] + a) / (y_count[y] + a * 2)


probs = {i: dict() for i in range(K)}
for y in range(K):
    for word in all_words:
        probs[y][word] = p(word, y)


log_probs = {i: dict() for i in range(K)}
log_1m_probs = {i: dict() for i in range(K)}

for y in range(K):
    for word in all_words:
        log_probs[y][word] = log(probs[y][word])
        log_1m_probs[y][word] = log(1 - probs[y][word])

M = int(input())

for i in range(M):
    line = set(input().split(' ')[1:])
    scores = [0.0 for i in range(K)]
    for y in range(K):
        if y_count[y] == 0:
            scores[y] = 0.0
        else:
            for word in all_words:
                if word in line:
                    if probs[y][word] == 0:
                        scores[y] = 0
                        break
                    else:
                        scores[y] += log_probs[y][word]
                else:
                    scores[y] = 0 if probs[y][word] == 1 else scores[y] + log_1m_probs[y][word]
            scores[y] += log(y_count[y] / N)

    max_score = max(scores)
    for j in range(K):
        if scores[j] != 0:
            scores[j] = exp(scores[j] - max_score) * penalty[j]

    for score in scores:
        print(score / sum(scores), end=' ')
    print()
