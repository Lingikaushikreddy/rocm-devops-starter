#!/usr/bin/env python3
"""Report what the local accelerator stack actually is and what it can do.

Written to be the first thing you run on a fresh box. It never raises on a
missing dependency - if PyTorch is not installed it says so and tells you which
wheel index to use, because on AMD that is the step people get wrong.

Usage:
    python scripts/gpu_probe.py
    python scripts/gpu_probe.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

# Env vars that silently change which GPUs a ROCm process can see. Worth
# printing every time - a "no devices found" bug is usually one of these.
ROCM_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
    "GPU_MAX_HW_QUEUES",
)

DTYPES = ("float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2")


def system_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "rocm_smi_on_path": shutil.which("rocm-smi") is not None,
        "rocminfo_on_path": shutil.which("rocminfo") is not None,
        "env": {k: os.environ[k] for k in ROCM_ENV_VARS if k in os.environ},
    }


def rocm_smi_version() -> str | None:
    """Best-effort ROCm version straight from the driver stack."""
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "--showdriverversion"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        for line in out.stdout.splitlines():
            if "Driver version" in line:
                return line.split(":", 1)[-1].strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}

    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    # A ROCm wheel reports a hip version and leaves cuda None. Both None means
    # a CPU-only build, which is the usual cause of "my GPU isn't detected".
    backend = "rocm" if hip else "cuda" if cuda else "cpu-only"

    info = {
        "installed": True,
        "version": torch.__version__,
        "backend": backend,
        "hip_version": hip,
        "cuda_version": cuda,
        "device_count": 0,
        "devices": [],
    }

    if not torch.cuda.is_available():
        return info

    info["device_count"] = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gib": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
            "gcn_arch": getattr(props, "gcnArchName", None),
        })
    return info


def dtype_support(device: str = "cuda") -> dict:
    """Actually execute a small matmul per dtype rather than trusting a flag.

    Plenty of dtypes construct fine and then fail in the kernel, so the only
    honest test is to run one.
    """
    try:
        import torch
    except ImportError:
        return {}
    if device != "cpu" and not torch.cuda.is_available():
        return {}

    results = {}
    for name in DTYPES:
        dt = getattr(torch, name, None)
        if dt is None:
            results[name] = "dtype not in this torch build"
            continue
        try:
            a = torch.ones((64, 64), dtype=dt, device=device)
            # fp8 has no matmul path on most stacks; exercise it via a cast so
            # we report "storage only" rather than a blanket failure.
            if "float8" in name:
                (a.to(torch.float16) @ a.to(torch.float16)).sum().item()
                results[name] = "storage ok (compute via upcast)"
            else:
                (a @ a).sum().item()
                results[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - probe reports, never raises
            results[name] = f"unsupported: {type(exc).__name__}: {str(exc)[:80]}"
    return results


def build_report() -> dict:
    torch = torch_info()
    device = "cuda" if torch.get("device_count", 0) else "cpu"
    return {
        "system": system_info(),
        "rocm_driver_version": rocm_smi_version(),
        "torch": torch,
        "dtype_support": dtype_support(device) if torch.get("installed") else {},
        "dtype_probe_device": device if torch.get("installed") else None,
    }


def print_human(report: dict) -> None:
    sysinfo = report["system"]
    print("=" * 62)
    print("  ROCm / accelerator probe")
    print("=" * 62)
    print(f"python            {sysinfo['python']}")
    print(f"platform          {sysinfo['platform']}")
    if report["rocm_driver_version"]:
        print(f"rocm driver       {report['rocm_driver_version']}")
    print(f"rocm-smi on PATH  {sysinfo['rocm_smi_on_path']}")
    if sysinfo["env"]:
        print("\nrelevant env vars (these change device visibility):")
        for k, v in sysinfo["env"].items():
            print(f"  {k}={v}")

    torch = report["torch"]
    print("\n--- pytorch ---")
    if not torch["installed"]:
        print("torch             NOT INSTALLED")
        print("\nOn an AMD box install the ROCm wheel, not the default one:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        print("Check https://pytorch.org/get-started/locally/ for the current"
              " ROCm version, then re-run this probe.")
        return

    print(f"version           {torch['version']}")
    print(f"backend           {torch['backend']}")
    if torch["backend"] == "cpu-only":
        print("  ^ this is a CPU-only build. It will never see your GPU.")
        print("    Reinstall from the ROCm wheel index.")
    print(f"devices visible   {torch['device_count']}")
    for d in torch["devices"]:
        arch = f" [{d['gcn_arch']}]" if d["gcn_arch"] else ""
        print(f"  [{d['index']}] {d['name']}{arch}  "
              f"{d['total_memory_gib']} GiB  {d['multi_processor_count']} CUs")

    if report["dtype_support"]:
        print(f"\n--- dtype support (measured on {report['dtype_probe_device']}) ---")
        for name, status in report["dtype_support"].items():
            print(f"  {name:<16} {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
