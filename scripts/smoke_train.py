#!/usr/bin/env python3
"""A deliberately tiny training run used as a smoke test.

The point is not the model. The point is to prove, in about ten seconds, that
this box can allocate on the device, run forward and backward, apply an
optimiser step and actually reduce a loss. That catches the majority of broken
ROCm installs, which look fine until autograd touches a kernel that is missing.

Exits non-zero if the loss did not improve, so CI fails loudly.

Usage:
    python scripts/smoke_train.py
    python scripts/smoke_train.py --device cpu --steps 50
    python scripts/smoke_train.py --dtype bf16
"""

from __future__ import annotations

import argparse
import sys
import time

TOLERANCE = 0.98  # final loss must be at least 2% below the first loss


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(hidden: int, features: int, classes: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, classes),
    )


def run(args) -> int:
    import torch
    import torch.nn as nn

    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    # autocast on CPU supports bf16 but not fp16, so fall back rather than crash.
    use_amp = amp_dtype is not None and not (device == "cpu" and amp_dtype is torch.float16)
    if amp_dtype is torch.float16 and device == "cpu":
        print("note: fp16 autocast is not supported on CPU, running fp32 instead")

    model = build_model(args.hidden, args.features, args.classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic but learnable: a fixed random projection defines the labels, so
    # a working stack must drive the loss down. Pure noise would not prove that.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    projection = torch.randn(args.features, args.classes, generator=generator).to(device)

    print(f"device={device}  dtype={args.dtype}  steps={args.steps}  "
          f"batch={args.batch}  hidden={args.hidden}")

    losses: list[float] = []
    if device != "cpu":
        torch.cuda.synchronize()
    started = time.perf_counter()

    for step in range(args.steps):
        x = torch.randn(args.batch, args.features, device=device)
        y = (x @ projection).argmax(dim=1)

        with torch.autocast(device_type="cuda" if device != "cpu" else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            loss = loss_fn(model(x), y)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

        if step % max(1, args.steps // 5) == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")

    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    first = sum(losses[:5]) / len(losses[:5])
    final = sum(losses[-5:]) / len(losses[-5:])
    samples = args.steps * args.batch

    print(f"\nelapsed           {elapsed:.2f}s")
    print(f"throughput        {samples / elapsed:,.0f} samples/s "
          f"({args.steps / elapsed:.1f} steps/s)")
    if device != "cpu":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak device mem   {peak:.3f} GiB")
    print(f"loss              {first:.4f} -> {final:.4f}")

    if final >= first * TOLERANCE:
        print(f"\nFAIL: loss did not improve by at least "
              f"{(1 - TOLERANCE) * 100:.0f}%. The stack runs but is not learning.")
        return 1

    print("\nPASS: forward, backward and optimiser step all work on this device.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not installed. Run scripts/gpu_probe.py for install hints.",
              file=sys.stderr)
        return 2

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
