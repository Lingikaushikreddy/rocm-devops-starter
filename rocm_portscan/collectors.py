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

# The OCP formats. Their fnuz counterparts are what gfx942 executes.
_FP8_FN_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2"})


def collect_python(path: str, source: str) -> list[Finding]:
    """Findings in a Python file. Never imports or executes it."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Python 2 files, templates and stray null bytes all turn up in real
        # repositories. None of them can be judged, and none should stop a scan.
        return []
    hits: set[tuple[str, int]] = set()
    fp8_lines: list[int] = []
    neg_inf_lines: list[int] = []
    first_seen: dict[str, int] = {}
    for node in ast.walk(tree):
        for rule_id in _node_rules(node):
            hits.add((rule_id, node.lineno))
        if isinstance(node, ast.Attribute) and node.attr in _FP8_FN_DTYPES:
            fp8_lines.append(node.lineno)
        if _is_negative_infinity(node):
            neg_inf_lines.append(node.lineno)
        for rule_id in _once_per_file_rules(node):
            first_seen[rule_id] = min(first_seen.get(rule_id, node.lineno), node.lineno)
    hits.update(("ROCM005", line) for line in fp8_lines)
    # ROCM006 is a same-file heuristic, not dataflow: the rule's message says so.
    if fp8_lines:
        hits.update(("ROCM006", line) for line in neg_inf_lines)
    hits.update(first_seen.items())
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
        name = _call_name(node)
        if name == "init_process_group" and _names_nccl(node):
            return ["ROCM200"]
        rule_id = _CALL_RULES.get(name)
        return [rule_id] if rule_id else []
    return []


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _names_nccl(call: ast.Call) -> bool:
    candidates = call.args[:1] + [k.value for k in call.keywords if k.arg == "backend"]
    return any(
        isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.lower() == "nccl"
        for c in candidates
    )


def _once_per_file_rules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch":
        if node.attr == "cuda":
            return ["ROCM201"]
        if node.attr in ("float16", "half"):
            return ["ROCM202"]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "half"
        and not node.args
    ):
        return ["ROCM202"]
    return []


def _is_negative_infinity(node: ast.AST) -> bool:
    """float('-inf'), -float('inf'), -math.inf, -torch.inf, -np.inf."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        return (isinstance(operand, ast.Attribute) and operand.attr == "inf") or _is_float_literal(
            operand, {"inf", "+inf", "infinity"}
        )
    return _is_float_literal(node, {"-inf", "-infinity"})


def _is_float_literal(node: ast.AST, spellings: set[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.strip().lower() in spellings
    )
