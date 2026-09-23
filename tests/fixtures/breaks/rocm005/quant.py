import torch


def quantise(w):
    return w.to(torch.float8_e4m3fn)
