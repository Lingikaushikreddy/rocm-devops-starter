def masked_scores(scores, mask):
    return scores.masked_fill(~mask, float("-inf")).softmax(-1)
