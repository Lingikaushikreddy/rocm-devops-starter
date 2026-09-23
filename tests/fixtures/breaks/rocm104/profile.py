import torch

torch.cuda.nvtx.range_push("step")
torch.cuda.nvtx.range_pop()
