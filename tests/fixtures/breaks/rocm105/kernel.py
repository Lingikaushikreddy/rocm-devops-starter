import triton
import triton.language as tl


@triton.jit
def fast_exp(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    y = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(out_ptr, y)
