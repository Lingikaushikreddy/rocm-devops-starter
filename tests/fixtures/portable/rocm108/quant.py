import torch

# Keyed on the arch, as vLLM does: only gfx94x executes fnuz.
arch = torch.cuda.get_device_properties(0).gcnArchName
FP8 = torch.float8_e4m3fnuz if "gfx94" in arch else torch.float8_e4m3fn
