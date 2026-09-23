import torch

if torch.cuda.get_device_capability() >= (8, 0):
    use_bf16 = True
