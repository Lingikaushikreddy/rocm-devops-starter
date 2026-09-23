import torch

FP8 = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
