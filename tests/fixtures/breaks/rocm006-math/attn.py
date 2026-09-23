import math

import torch


def masked_scores(scores, mask):
    scores = scores.masked_fill(~mask, -math.inf)
    return scores.to(torch.float8_e5m2)
