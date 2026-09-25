import torch

# PyTorch's own form (torch/_meta_registrations.py): the hip check only guards
# the gcnArchName lookup; the arch decides.
fp8_dtype = torch.float8_e4m3fn
if (
    torch.version.hip
    and torch.cuda.is_available()
    and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
):
    fp8_dtype = torch.float8_e4m3fnuz
