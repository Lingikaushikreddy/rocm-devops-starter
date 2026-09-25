import torch


def fp8_dtype():
    if torch.version.hip is not None:
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def quantise(w, is_rocm):
    if is_rocm:
        dtype = torch.float8_e4m3fnuz
    else:
        dtype = torch.float8_e4m3fn
    return w.to(dtype)
