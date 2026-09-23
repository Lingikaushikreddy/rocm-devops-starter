"""Runnable proofs for rules that claim platform behaviour.

Each function returns True while the behaviour its rule describes still
reproduces on this machine. No GPU is needed: fp8 conversion runs on CPU. CI
runs every one of these, and one returning False means a rule is claiming
something no longer true - demote or delete that rule.

torch is imported inside each function so the scanner itself never needs it.
"""

from __future__ import annotations


def fn_max_overflows_fnuz() -> bool:
    """448 is an ordinary e4m3fn value and NaN in e4m3fnuz, the format gfx942 runs."""
    import torch

    if torch.finfo(torch.float8_e4m3fn).max != 448.0:
        return False
    if torch.finfo(torch.float8_e4m3fnuz).max != 240.0:
        return False
    converted = torch.tensor([448.0]).to(torch.float8_e4m3fnuz).float()
    return bool(torch.isnan(converted).all())


def fnuz_has_no_infinity() -> bool:
    """An -inf mask fill becomes NaN in e4m3fnuz instead of staying -inf."""
    import torch

    converted = torch.tensor([float("-inf")]).to(torch.float8_e4m3fnuz).float()
    return bool(torch.isnan(converted).all())
