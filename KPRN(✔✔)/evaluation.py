import numpy as np


def ndcg(score_target_tuple, k):

    for i, (score, target) in enumerate(score_target_tuple[:k]):
        if target == 1:
            return np.log(2) / np.log(i + 2)

    return 0


def hit(score_target_tuple, k):

    for score, target in score_target_tuple[:k]:
        if target == 1:
            return 1

    return 0
