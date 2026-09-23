"""Find portability signals in source files.

Two collectors, because the signals live in two kinds of file. Python goes
through the AST, so a comment or docstring that mentions apex never fires.
Everything else - requirements, build scripts, Dockerfiles, CUDA sources -
goes through line-based text matching.
"""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

from .rules import Finding


def _dependency_pattern(name: str) -> re.Pattern[str]:
    # pip treats -, _ and . as the same character in a project name.
    body = "[-_.]".join(re.escape(part) for part in re.split(r"[-_.]", name))
    return re.compile(rf"(?<![\w.-]){body}(?![\w-])", re.IGNORECASE)


_DEPENDENCIES = [
    (_dependency_pattern(name), rule_id)
    for name, rule_id in (
        ("apex", "ROCM001"),
        ("transformer-engine", "ROCM002"),
        ("pynvml", "ROCM007"),
        ("nvidia-ml-py", "ROCM007"),
        ("nvidia-ml-py3", "ROCM007"),
        ("flash-attn", "ROCM101"),
        ("bitsandbytes", "ROCM102"),
        ("xformers", "ROCM107"),
    )
] + [(re.compile(r"(?<![\w.-])cupy[-_.]cuda\w*", re.IGNORECASE), "ROCM008")]

_NVCC_ARCH = re.compile(r"-gencode\b|\barch=compute_\d+|\bsm_\d{2,3}a?\b")
_CUDA_BASE_IMAGE = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(?:docker\.io/)?(?:nvidia/cuda|nvcr\.io/nvidia/)",
    re.IGNORECASE,
)
_NVIDIA_CONTAINER_GPU = re.compile(
    r"--gpus\b|\bnvidia-docker\b|\b(?:runtime|driver):\s*nvidia\b"
)

_DEPENDENCY_FILE = re.compile(
    r".*requirements.*\.(?:txt|in)|pyproject\.toml|setup\.py|setup\.cfg"
    r"|environment.*\.ya?ml|Pipfile"
)
_BUILD_FILE = re.compile(r"setup\.py|CMakeLists\.txt|.*\.cmake|Makefile|.*\.mk|.*\.sh")
_DOCKERFILE = re.compile(r"Dockerfile.*|.*\.dockerfile|Containerfile", re.IGNORECASE)
_LAUNCH_FILE = re.compile(r".*\.sh|Makefile|.*\.mk|(?:docker-)?compose.*\.ya?ml")
_CUDA_SOURCE_SUFFIXES = (".cu", ".cuh")

# pip, shell, make, CMake and Docker all treat "#" after whitespace as a
# comment. A "#" inside a URL (#egg=apex) has no space before it and is kept.
_COMMENT = re.compile(r"(?:^|\s)#.*$")


def collect_text(path: str, text: str) -> list[Finding]:
    """Findings in a non-Python file, keyed on its file name."""
    name = PurePosixPath(path).name
    if name.endswith(_CUDA_SOURCE_SUFFIXES):
        return [Finding(path, 1, "ROCM003", "CUDA source file")]
    rules = _line_rules(name)
    if not rules:
        return []
    found: set[Finding] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _COMMENT.sub("", raw)
        for pattern, rule_id in rules:
            if pattern.search(line):
                found.add(Finding(path, lineno, rule_id, _snippet(raw)))
    return sorted(found)


def _line_rules(name: str) -> list[tuple[re.Pattern[str], str]]:
    rules: list[tuple[re.Pattern[str], str]] = []
    if _DEPENDENCY_FILE.fullmatch(name):
        rules += _DEPENDENCIES
    if _BUILD_FILE.fullmatch(name):
        rules.append((_NVCC_ARCH, "ROCM004"))
    if _DOCKERFILE.fullmatch(name):
        rules.append((_CUDA_BASE_IMAGE, "ROCM103"))
    if _LAUNCH_FILE.fullmatch(name):
        rules.append((_NVIDIA_CONTAINER_GPU, "ROCM106"))
    return rules


def _snippet(raw: str) -> str:
    return raw.strip()[:120]


# Top-level import name -> rule.
_IMPORT_RULES = {
    "apex": "ROCM001",
    "transformer_engine": "ROCM002",
    "pynvml": "ROCM007",
    "flash_attn": "ROCM101",
    "bitsandbytes": "ROCM102",
    "xformers": "ROCM107",
}

# Called function name (last dotted segment) -> rule.
_CALL_RULES = {
    "CUDAExtension": "ROCM003",
    "get_device_capability": "ROCM100",
    "inline_asm_elementwise": "ROCM105",
}


def collect_python(path: str, source: str) -> list[Finding]:
    """Findings in a Python file. Never imports or executes it."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Python 2 files, templates and stray null bytes all turn up in real
        # repositories. None of them can be judged, and none should stop a scan.
        return []
    hits: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        for rule_id in _node_rules(node):
            hits.add((rule_id, node.lineno))
    lines = source.splitlines()
    return sorted(
        Finding(path, line, rule_id, _snippet(lines[line - 1]))
        for rule_id, line in hits
    )


def _node_rules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [
            _IMPORT_RULES[alias.name.split(".")[0]]
            for alias in node.names
            if alias.name.split(".")[0] in _IMPORT_RULES
        ]
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module:
            return []  # relative import: a local module, not the package
        found = []
        top = node.module.split(".")[0]
        if top in _IMPORT_RULES:
            found.append(_IMPORT_RULES[top])
        if node.module == "torch.cuda" and any(a.name == "nvtx" for a in node.names):
            found.append("ROCM104")
        return found
    if isinstance(node, ast.Attribute):
        if (
            node.attr == "nvtx"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "cuda"
        ):
            return ["ROCM104"]
        return []
    if isinstance(node, ast.Call):
        rule_id = _CALL_RULES.get(_call_name(node))
        return [rule_id] if rule_id else []
    return []


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""
