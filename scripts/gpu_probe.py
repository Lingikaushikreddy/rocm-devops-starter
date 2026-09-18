#!/usr/bin/env python3
"""Report what the local accelerator stack actually is and what it can do.

Written to be the first thing you run on a fresh box. It never raises on a
missing dependency - if PyTorch is not installed it says so and tells you which
wheel index to use, because on AMD that is the step people get wrong.

Two of the checks here exist because the obvious version of them is wrong:

- Presence of /dev/kfd is NOT proof a card is usable. The node can exist on a
  card whose driver never bound, so it reports healthy on exactly the failure
  it looks like it should catch. Whether rocminfo resolves a gfx target is the
  honest test.
- rocm-smi and amd-smi exit 0 when they fail, so this parses their output and
  never branches on a return code.

Both come from a measured writeup by William_Mclean on the AMD developer forum:
https://devcommunity.amd.com/t/which-numeric-formats-an-mi300x-actually-executes-measured-on-a-developer-cloud-droplet/1043

Usage:
    python scripts/gpu_probe.py
    python scripts/gpu_probe.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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

# fp8 on gfx942 is the fnuz variant, which uses the same bytes as the fn variant
# for different values. 448 is an ordinary weight on NVIDIA hardware and becomes
# NaN under fnuz, so a checkpoint quantized elsewhere is not drop-in. A dtype can
# construct, matmul and report clean while silently destroying in-range values,
# which is why support is not the same question as fidelity.
ROUNDTRIP_PROBES = (1.0, 2.0, 240.0, 448.0)
FP8_DTYPES = ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")


def _run(cmd: list[str]) -> str | None:
    """Run a command and return stdout, or None if it could not run at all.

    Deliberately ignores the exit status: the ROCm CLI tools return 0 on
    failure, so the output is the only trustworthy signal.
    """
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=20, check=False)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def rocminfo_gfx_targets() -> list[str]:
    """Every gfx target rocminfo resolves. Empty means no GPU agent bound.

    This is the liveness check. A freshly provisioned instance can need one
    reboot before amdgpu binds, and until it does rocminfo reports no gfx
    target even though /dev/kfd is sitting right there.
    """
    out = _run(["rocminfo"])
    if not out:
        return []
    return sorted(set(re.findall(r"\bgfx\d+[a-z]*\b", out)))


def rocm_driver_version() -> str | None:
    out = _run(["rocm-smi", "--showdriverversion"])
    if not out:
        return None
    for line in out.splitlines():
        if "Driver version" in line:
            return line.split(":", 1)[-1].strip()
    return None


def system_info() -> dict:
    gfx = rocminfo_gfx_targets()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "rocm_smi_on_path": shutil.which("rocm-smi") is not None,
        "rocminfo_on_path": shutil.which("rocminfo") is not None,
        # Reported for completeness, explicitly NOT used as a liveness signal.
        "dev_kfd_present": os.path.exists("/dev/kfd"),
        "rocminfo_gfx_targets": gfx,
        "gpu_agent_bound": bool(gfx),
        "env": {k: os.environ[k] for k in ROCM_ENV_VARS if k in os.environ},
    }


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


def fp8_roundtrip(device: str = "cpu") -> dict:
    """Cast known values into each fp8 dtype and back, reporting what survives.

    Executing without error is not the same as preserving your weights.
    """
    try:
        import torch
    except ImportError:
        return {}

    results = {}
    for name in FP8_DTYPES:
        dt = getattr(torch, name, None)
        if dt is None:
            results[name] = {"available": False}
            continue
        try:
            src = torch.tensor(ROUNDTRIP_PROBES, dtype=torch.float32, device=device)
            back = src.to(dt).to(torch.float32).tolist()
            # No probe value is NaN, so a straight inequality is enough:
            # a value that came back NaN compares unequal and is correctly
            # reported as destroyed.
            lost = [v for v, b in zip(ROUNDTRIP_PROBES, back, strict=True)
                    if b != v]
            results[name] = {
                "available": True,
                "roundtrip": back,
                "lossless": not lost,
                "values_lost": lost,
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"available": True,
                             "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return results


def build_report() -> dict:
    torch = torch_info()
    device = "cuda" if torch.get("device_count", 0) else "cpu"
    return {
        "system": system_info(),
        "rocm_driver_version": rocm_driver_version(),
        "torch": torch,
        "dtype_support": dtype_support(device) if torch.get("installed") else {},
        "fp8_roundtrip": fp8_roundtrip("cpu") if torch.get("installed") else {},
        "dtype_probe_device": device if torch.get("installed") else None,
    }


def print_human(report: dict) -> None:
    sysinfo = report["system"]
    print("=" * 66)
    print("  ROCm / accelerator probe")
    print("=" * 66)
    print(f"python            {sysinfo['python']}")
    print(f"platform          {sysinfo['platform']}")
    if report["rocm_driver_version"]:
        print(f"rocm driver       {report['rocm_driver_version']}")

    print("\n--- is a GPU actually bound? ---")
    if sysinfo["rocminfo_on_path"]:
        if sysinfo["gpu_agent_bound"]:
            print(f"rocminfo gfx      {', '.join(sysinfo['rocminfo_gfx_targets'])}")
        else:
            print("rocminfo gfx      NONE - no GPU agent bound")
            if sysinfo["dev_kfd_present"]:
                print("  /dev/kfd exists but no gfx target resolved. On a freshly")
                print("  provisioned instance amdgpu can fail to bind until one")
                print("  reboot. /dev/kfd is not evidence the card came up.")
    else:
        print("rocminfo          not on PATH (cannot verify a GPU is bound)")
    print(f"/dev/kfd present  {sysinfo['dev_kfd_present']}  (not a liveness signal)")

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

    if report["fp8_roundtrip"]:
        print("\n--- fp8 round-trip fidelity ---")
        print(f"  probe values: {list(ROUNDTRIP_PROBES)}")
        for name, res in report["fp8_roundtrip"].items():
            if not res.get("available"):
                print(f"  {name:<18} not in this torch build")
            elif "error" in res:
                print(f"  {name:<18} {res['error']}")
            elif res["lossless"]:
                print(f"  {name:<18} lossless")
            else:
                print(f"  {name:<18} LOSSY -> {res['roundtrip']}")
                print(f"  {'':<18} destroys {res['values_lost']}")


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
