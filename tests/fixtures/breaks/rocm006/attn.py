import torch


def masked_scores(scores, mask):
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.to(torch.float8_e4m3fn)
