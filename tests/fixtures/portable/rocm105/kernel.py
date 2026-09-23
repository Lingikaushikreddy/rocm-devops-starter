import triton
import triton.language as tl


@triton.jit
def fast_exp(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    tl.store(out_ptr, tl.exp2(x))
