"""Walk a tree and run the collectors over every file in it.

Symlinks are never followed, so a scan cannot loop or leave the target.
Binary files and files over MAX_BYTES are skipped: in real repositories those
are checkpoints and generated sources, not portability signals.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

from .collectors import collect_python, collect_text
from .rules import Finding

MAX_BYTES = 2_000_000

PRUNED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
        ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache", "site-packages",
        "build", "dist",
    }
)


def scan(root: Path, exclude: Sequence[str] = ()) -> list[Finding]:
    """Every finding under root, sorted by path, line and rule.

    exclude holds fnmatch globs matched against paths relative to root. A
    directory that matches is not entered.
    """
    root = Path(root)
    if root.is_file():
        return sorted(set(_scan_file(root, root.name)))
    findings: list[Finding] = []
    for rel in _walk(root, exclude):
        findings.extend(_scan_file(root / rel, rel))
    return sorted(set(findings))


def _walk(root: Path, exclude: Sequence[str]) -> Iterator[str]:
    # os.walk does not descend into symlinked directories by default.
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in PRUNED_DIRS and not _excluded(prefix + d, exclude)
        )
        for name in sorted(filenames):
            rel = prefix + name
            if os.path.islink(os.path.join(dirpath, name)) or _excluded(rel, exclude):
                continue
            yield rel


def _excluded(rel: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def _scan_file(full: Path, rel: str) -> list[Finding]:
    try:
        if full.stat().st_size > MAX_BYTES:
            return []
        data = full.read_bytes()
    except OSError:
        return []
    if b"\0" in data[:8192]:
        return []
    text = data.decode("utf-8", errors="replace")
    found = collect_text(rel, text)
    if rel.endswith(".py"):
        found += collect_python(rel, text)
    return found
