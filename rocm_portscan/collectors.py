"""Find portability signals in source files.

Two collectors, because the signals live in two kinds of file. Python goes
through the AST, so a comment or docstring that mentions apex never fires.
Everything else - requirements, build scripts, Dockerfiles, CUDA sources -
goes through line-based text matching.
"""

from __future__ import annotations

from .rules import Finding


def collect_text(path: str, text: str) -> list[Finding]:
    """Findings in a non-Python file, keyed on its file name."""
    return []
