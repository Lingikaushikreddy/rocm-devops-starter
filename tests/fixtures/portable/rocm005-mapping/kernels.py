import torch
import triton.language as tl

# A table covering both formats is a lookup, not a hardcoded choice.
FP8_DTYPE_MAP = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
}
