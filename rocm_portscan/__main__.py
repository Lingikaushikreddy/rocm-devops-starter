"""python -m rocm_portscan <path> - report what will not survive a move to ROCm.

Exit codes: 0 no blockers, 1 blockers found, 2 bad invocation. The exit code
counts blockers whatever --min-tier hides, so it is safe to use as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .report import count_by_tier, render_json, render_markdown, render_terminal
from .rules import Tier
from .scan import scan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rocm_portscan",
        description="Report what in a repository will not survive a move from CUDA to ROCm.",
    )
    parser.add_argument("path", help="repository root or single file to scan")
    parser.add_argument("--format", choices=("terminal", "markdown", "json"), default="terminal")
    parser.add_argument("--min-tier", choices=("blocker", "review", "info"), default="review")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB",
        help="skip paths matching GLOB, relative to the root (repeatable)",
    )
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"rocm_portscan: no such file or directory: {args.path}", file=sys.stderr)
        return 2

    findings = scan(root, args.exclude)
    min_tier = Tier[args.min_tier.upper()]
    if args.format == "json":
        output = render_json(findings, min_tier)
    elif args.format == "markdown":
        output = render_markdown(findings, min_tier, target=root.resolve().name)
    else:
        output = render_terminal(findings, min_tier)
    sys.stdout.write(output)
    return 1 if count_by_tier(findings)[Tier.BLOCKER] else 0


if __name__ == "__main__":
    sys.exit(main())
