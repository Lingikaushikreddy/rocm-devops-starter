"""Rule definitions for the ROCm portability scanner.

Pure data. Matching logic lives in collectors.py: a rule here says what a
finding means and how sure we are of it, never how it is found.

A BLOCKER asserts a failure, so it must carry a proof: "doc:<https url>" or
"selftest:<function in proofs.py>". Tier answers a separate question - does
matching the pattern make the breakage certain? - so a rule with an airtight
proof is still REVIEW when the pattern is only correlated with the risk.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Tier(enum.IntEnum):
    INFO = 0
    REVIEW = 1
    BLOCKER = 2


@dataclass(frozen=True)
class Rule:
    id: str
    tier: Tier
    title: str
    message: str
    proof: str = ""


@dataclass(frozen=True, order=True)
class Finding:
    path: str  # relative to the scan root, forward slashes
    line: int
    rule_id: str
    snippet: str


_RULES = (
    Rule(
        "ROCM001",
        Tier.REVIEW,
        "`apex` dependency",
        "NVIDIA's apex does not build on ROCm, but AMD maintains a fork at "
        "github.com/ROCm/apex. Check which apex is installed and which of its "
        "fused kernels this code uses.",
    ),
    Rule(
        "ROCM002",
        Tier.REVIEW,
        "`transformer_engine` dependency",
        "Transformer Engine's ROCm support was not verified when this rule "
        "was written. Check its current state before relying on it.",
    ),
    Rule(
        "ROCM003",
        Tier.REVIEW,
        "CUDA sources or `CUDAExtension`",
        "On ROCm, torch.utils.cpp_extension hipifies CUDA sources "
        "automatically, so most extensions built this way compile unchanged. "
        "Hipify does not translate inline PTX asm, or code that assumes a "
        "32-wide warp (CDNA wavefronts are 64). Check for those.",
    ),
    Rule(
        "ROCM004",
        Tier.REVIEW,
        "nvcc architecture flags (`-gencode`, `compute_XX`, `sm_XX`)",
        "On ROCm builds these flags reach hipcc unchanged. Whether hipcc "
        "rejects them is unverified. ROCm targets are set with "
        "--offload-arch or PYTORCH_ROCM_ARCH.",
    ),
    Rule(
        "ROCM005",
        Tier.REVIEW,
        "hardcoded `float8_e4m3fn` / `float8_e5m2`",
        "gfx942 (MI300) executes the fnuz fp8 formats. e4m3fnuz caps at 240 "
        "where e4m3fn allows 448, and neither e4m3fnuz nor e5m2fnuz has an "
        "infinity, so out-of-range e4m3fn values and any infinity in e5m2 "
        "data become NaN there. Check whether these tensors will meet fnuz "
        "hardware.",
        proof="selftest:fn_max_overflows_fnuz",
    ),
    Rule(
        "ROCM006",
        Tier.REVIEW,
        "`-inf` in the same file as an fp8 dtype",
        "`-inf` and an fp8 dtype appear in the same file; check whether the "
        "fill reaches an fp8 tensor. In e4m3fnuz, -inf becomes NaN.",
        proof="selftest:fnuz_has_no_infinity",
    ),
    Rule(
        "ROCM007",
        Tier.REVIEW,
        "`pynvml` / `nvidia-ml-py`",
        "NVML cannot see AMD GPUs. Harmless if every call is guarded; the "
        "ROCm equivalent is amdsmi.",
    ),
    Rule(
        "ROCM008",
        Tier.BLOCKER,
        "`cupy-cuda*` wheel pin",
        "A cupy-cuda* wheel is built against CUDA and cannot drive an AMD "
        "GPU. CuPy's install docs route AMD users to a separate package, "
        "amd-cupy.",
        proof="doc:https://docs.cupy.dev/en/stable/install.html",
    ),
    Rule(
        "ROCM100",
        Tier.REVIEW,
        "`get_device_capability()` use",
        "On ROCm this cannot report an NVIDIA compute capability. Version "
        "gates written for NVIDIA architectures (e.g. `>= (8, 0)` for "
        "Ampere) need checking against what AMD GPUs report.",
    ),
    Rule(
        "ROCM101",
        Tier.REVIEW,
        "`flash_attn`",
        "Check that a ROCm build of flash-attn exists for the target GPU "
        "(AMD maintains one at github.com/ROCm/flash-attention).",
    ),
    Rule(
        "ROCM102",
        Tier.REVIEW,
        "`bitsandbytes`",
        "Check that the installed bitsandbytes build supports ROCm and the "
        "quantisation modes this code uses.",
    ),
    Rule(
        "ROCM103",
        Tier.REVIEW,
        "CUDA base image (`nvidia/cuda`, `nvcr.io/nvidia/`)",
        "This image ships the CUDA stack. The ROCm equivalents are the "
        "rocm/pytorch and rocm/dev-* images.",
    ),
    Rule(
        "ROCM104",
        Tier.REVIEW,
        "`torch.cuda.nvtx`",
        "NVTX ranges are NVIDIA profiler markers. On ROCm, check that the "
        "ranges still reach your profiler; PyTorch's ROCm-on-Windows builds "
        "stub them out.",
    ),
    Rule(
        "ROCM105",
        Tier.REVIEW,
        "inline assembly in a Triton kernel",
        "Inline assembly is target-specific; PTX will not assemble for an "
        "AMD GPU. Check for a portable tl.* equivalent.",
    ),
    Rule(
        "ROCM106",
        Tier.REVIEW,
        "`--gpus` / NVIDIA container runtime",
        "ROCm containers get GPUs through --device=/dev/kfd --device=/dev/dri, "
        "not --gpus or the NVIDIA runtime.",
    ),
    Rule(
        "ROCM107",
        Tier.REVIEW,
        "`xformers`",
        "Check that a ROCm build of xformers exists and supports the "
        "attention ops this code uses.",
    ),
    Rule(
        "ROCM200",
        Tier.INFO,
        '`init_process_group("nccl")`',
        'Portable. PyTorch on ROCm keeps the "nccl" backend name and routes '
        "it to RCCL.",
    ),
    Rule(
        "ROCM201",
        Tier.INFO,
        "`torch.cuda` usage",
        "Portable. ROCm builds of PyTorch reuse the torch.cuda namespace. "
        "Reported once per file.",
    ),
    Rule(
        "ROCM202",
        Tier.INFO,
        "fp16 / `.half()`",
        "Portable. fp16 is supported on ROCm. Reported once per file.",
    ),
)

RULES = {rule.id: rule for rule in _RULES}
