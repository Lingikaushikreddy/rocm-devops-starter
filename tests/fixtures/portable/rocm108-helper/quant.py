import torch


def is_fp8_fnuz():
    return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName


def fp8_dtype():
    if is_fp8_fnuz():
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn
