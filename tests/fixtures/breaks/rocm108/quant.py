import torch

# Right on MI300 (gfx942), wrong on MI355X (gfx950) and RDNA4, which run e4m3fn.
FP8 = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
