#!/usr/bin/env python3
"""Turn a real probe run into a forum post.

The point of this script is the thing it refuses to do. It will not produce a
hardware report unless a GPU was actually bound on the machine it ran on, so
there is no path from "I have not run this yet" to a post full of numbers.
If you want the skeleton to read, ask for --template and it is clearly marked
as unfilled.

Usage:
    python scripts/make_report.py                  # run the probe, emit a post
    python scripts/make_report.py --out post.md
    python scripts/make_report.py --template       # empty skeleton, no numbers
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TEMPLATE = """> UNFILLED TEMPLATE - this is not a result. Run
> `python scripts/make_report.py` on the machine with the GPU.

## What I ran

<hardware, ROCm version, torch version>

## What the probe reported

<paste the probe output>

## What broke

<the failures, in order, and what fixed each one>

## What I would tell someone starting tomorrow

<the short version>
"""


def run_probe() -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "gpu_probe.py"), "--json"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"probe failed:\n{out.stderr}")
    return json.loads(out.stdout)


def run_smoke(dtype: str) -> str:
    out = subprocess.run(
        [sys.executable, str(HERE / "smoke_train.py"), "--dtype", dtype],
        capture_output=True, text=True, check=False,
    )
    status = "PASS" if out.returncode == 0 else f"FAIL (exit {out.returncode})"
    tail = [ln for ln in out.stdout.splitlines()
            if ln.startswith(("elapsed", "throughput", "peak", "loss"))]
    return f"- **{dtype}**: {status}\n" + "".join(f"  - `{ln.strip()}`\n" for ln in tail)


def render(report: dict, smoke: str) -> str:
    sysinfo = report["system"]
    torch = report["torch"]
    dev = torch["devices"][0] if torch["devices"] else {}

    lines = [
        (f"Ran a container and CI harness on {dev.get('name', 'an AMD GPU')} "
         "and wrote down what actually happened, including what broke."),
        "",
        "## The box",
        "",
        f"- gfx target: `{', '.join(sysinfo['rocminfo_gfx_targets']) or 'unknown'}`",
        (f"- device: {dev.get('name', '?')} "
         f"({dev.get('total_memory_gib', '?')} GiB, "
         f"{dev.get('multi_processor_count', '?')} CUs)"),
        f"- torch: `{torch['version']}` (hip `{torch['hip_version']}`)",
        f"- rocm driver: `{report.get('rocm_driver_version') or 'unreported'}`",
        "",
        "## dtype support, measured",
        "",
        "Executed a real matmul per dtype rather than reading a capability flag.",
        "",
        "| dtype | result |",
        "|---|---|",
    ]
    for name, status in report["dtype_support"].items():
        lines.append(f"| `{name}` | {status} |")

    if report.get("fp8_roundtrip"):
        lines += ["", "## fp8 round-trip fidelity", "",
                  "Executing without error is not the same as preserving values.",
                  "", "| dtype | round-trip | verdict |", "|---|---|---|"]
        for name, res in report["fp8_roundtrip"].items():
            if not res.get("available") or "error" in res:
                continue
            verdict = "lossless" if res["lossless"] else f"destroys {res['values_lost']}"
            lines.append(f"| `{name}` | `{res['roundtrip']}` | {verdict} |")

    lines += ["", "## Smoke training run", "", smoke,
              "## What broke", "",
              "<fill this in by hand - it is the part people actually want>",
              "",
              "Harness: https://github.com/Lingikaushikreddy/rocm-devops-starter"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write to a file instead of stdout")
    parser.add_argument("--template", action="store_true",
                        help="emit the empty skeleton and exit")
    args = parser.parse_args()

    if args.template:
        text = TEMPLATE
    else:
        report = run_probe()
        bound = report["system"]["gpu_agent_bound"]
        count = report["torch"].get("device_count", 0)
        if not (bound or count):
            print(
                "Refusing to generate a hardware report: no GPU is bound on this\n"
                "machine (rocminfo resolved no gfx target and torch sees no\n"
                "devices). Run this on the box with the GPU. If you only want to\n"
                "read the structure, use --template.",
                file=sys.stderr,
            )
            return 1
        smoke = "".join(run_smoke(d) for d in ("fp32", "bf16", "fp16"))
        text = render(report, smoke)

    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
