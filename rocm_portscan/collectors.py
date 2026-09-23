"""Find portability signals in source files.

Two collectors, because the signals live in two kinds of file. Python goes
through the AST, so a comment or docstring that mentions apex never fires.
Everything else - requirements, build scripts, Dockerfiles, CUDA sources -
goes through line-based text matching, and dependency names only count where
a file declares dependencies.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Iterator
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
] + [
    # A wheel name always carries the CUDA version; "cupy.cuda" is a module path
    # and f"cupy-cuda{major}x" is a template, and neither is a pin.
    (re.compile(r"(?<![\w.-])cupy[-_]cuda\d+x?(?![\w-])", re.IGNORECASE), "ROCM008")
]

# Bare "sm_90" is left out: it turns up in GPU-name tables and log text far
# more often than on an nvcc command line.
_NVCC_ARCH = re.compile(r"-gencode\b|--generate-code\b|\barch=compute_\d+|-arch[= ]sm_\d+")
_CUDA_BASE_IMAGE = re.compile(
    r"^\s*FROM\s+(?:--platform=\S+\s+)?(?:docker\.io/)?(?:nvidia/cuda|nvcr\.io/nvidia/)",
    re.IGNORECASE,
)
# --gpus alone also names Slurm and Lightning flags; only a container CLI on
# the same command makes it the NVIDIA container runtime's flag.
_NVIDIA_CONTAINER_GPU = re.compile(
    r"\b(?:docker|podman)\b.*\s--gpus(?![-\w])|\bnvidia-docker\b|\b(?:runtime|driver):\s*nvidia\b"
)

_REQUIREMENTS_FILE = re.compile(r".*requirements.*\.(?:txt|in)|environment.*\.ya?ml|Pipfile")
_BUILD_FILE = re.compile(r"CMakeLists\.txt|.*\.cmake|Makefile|.*\.mk|.*\.sh")
_DOCKERFILE = re.compile(r"Dockerfile.*|.*\.dockerfile|Containerfile", re.IGNORECASE)
_LAUNCH_FILE = re.compile(r".*\.sh|Makefile|.*\.mk|(?:docker-)?compose.*\.ya?ml")
_CUDA_SOURCE_SUFFIXES = (".cu", ".cuh")

# pip, shell, make, CMake, TOML and Docker all treat "#" after whitespace as a
# comment. A "#" inside a URL (#egg=apex) has no space before it and is kept.
_COMMENT = re.compile(r"(?:^|\s)#.*$")

_TOML_TABLE = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*$")
_TOML_KEY = re.compile(r"""^\s*("[^"]*"|'[^']*'|[\w.-]+)\s*=""")
_TOML_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')
_POETRY_DEPENDENCY_TABLE = re.compile(
    r"tool\.poetry(?:\.group\.[\w-]+)?\.(?:dev-)?dependencies"
)

Candidates = Iterator[tuple[int, str]]


def collect_text(path: str, text: str) -> list[Finding]:
    """Findings in a non-Python file, keyed on its file name."""
    name = PurePosixPath(path).name
    if name.endswith(_CUDA_SOURCE_SUFFIXES):
        return [Finding(path, 1, "ROCM003", "CUDA source file")]
    lines = text.splitlines()
    found: set[Finding] = set()

    def match(candidates: Candidates, rules: list[tuple[re.Pattern[str], str]]) -> None:
        for lineno, line in candidates:
            for pattern, rule_id in rules:
                if pattern.search(line):
                    found.add(Finding(path, lineno, rule_id, _snippet(lines[lineno - 1])))

    if name == "pyproject.toml":
        match(_pyproject_dependency_lines(lines), _DEPENDENCIES)
    elif _REQUIREMENTS_FILE.fullmatch(name):
        match(((n, _strip_marker(line)) for n, line in _uncommented(lines)), _DEPENDENCIES)
    if _BUILD_FILE.fullmatch(name):
        match(_uncommented(lines), [(_NVCC_ARCH, "ROCM004")])
    if _DOCKERFILE.fullmatch(name):
        match(_uncommented(lines), [(_CUDA_BASE_IMAGE, "ROCM103")])
    if _LAUNCH_FILE.fullmatch(name):
        match(_logical_lines(lines), [(_NVIDIA_CONTAINER_GPU, "ROCM106")])
    return sorted(found)


def _uncommented(lines: list[str]) -> Candidates:
    for lineno, raw in enumerate(lines, start=1):
        yield lineno, _COMMENT.sub("", raw)


def _logical_lines(lines: list[str]) -> Candidates:
    """Shell commands continued with a trailing backslash, joined into one line."""
    start, parts = 0, []
    for lineno, line in _uncommented(lines):
        start = start or lineno
        if line.rstrip().endswith("\\"):
            parts.append(line.rstrip()[:-1])
            continue
        parts.append(line)
        yield start, " ".join(parts)
        start, parts = 0, []
    if parts:
        yield start, " ".join(parts)


def _strip_marker(requirement: str) -> str:
    # "torch>=2; extra == 'apex'" depends on torch; the marker names an extra.
    return requirement.split(";")[0]


def _pyproject_dependency_lines(lines: list[str]) -> Candidates:
    """Lines of pyproject.toml that declare dependencies, markers stripped.

    Descriptions, keywords, tool settings (mypy overrides, uv source tables)
    name packages too, but none of them is a dependency.
    """
    table, key, depth = "", "", 0
    for lineno, line in _uncommented(lines):
        if depth <= 0:
            header = _TOML_TABLE.match(line)
            if header:
                table, key = header.group(1), ""
                continue
            assignment = _TOML_KEY.match(line)
            if not assignment:
                continue
            key = assignment.group(1).strip("\"'")
        if _is_dependency_location(table, key):
            yield lineno, _TOML_STRING.sub(lambda m: _strip_marker(m.group(0)), line)
        bare = _TOML_STRING.sub("", line)
        depth += bare.count("[") + bare.count("{") - bare.count("]") - bare.count("}")


def _is_dependency_location(table: str, key: str) -> bool:
    if table == "project":
        return key == "dependencies"
    if table == "build-system":
        return key == "requires"
    if table in ("project.optional-dependencies", "dependency-groups"):
        return True
    return bool(_POETRY_DEPENDENCY_TABLE.fullmatch(table))


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
_FP8_FNUZ_DTYPES = frozenset({"float8_e4m3fnuz", "float8_e5m2fnuz"})

# setup() keywords whose strings are requirement specifiers.
_REQUIREMENT_KEYWORDS = frozenset(
    {"install_requires", "setup_requires", "tests_require", "extras_require"}
)

# A string that is an nvcc flag, as opposed to prose that mentions one.
_NVCC_FLAG_STRING = re.compile(r"^\s*(?:-gencode|--generate-code)\b|\barch=compute_\d+|-arch[= ]sm_\d+")


def collect_python(path: str, source: str) -> list[Finding]:
    """Findings in a Python file. Never imports or executes it."""
    try:
        # The scanned code's own warnings (invalid escapes and the like) are
        # its business, not ours; letting them through floods stderr.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        # Python 2 files, templates, stray null bytes and machine-generated
        # expressions too deep for the parser all turn up in real repositories.
        # None of them can be judged, and none should stop a scan.
        return []
    hits: set[tuple[str, int]] = set()
    prose = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    fp8_lines: list[int] = []
    fnuz_lines: set[int] = set()
    neg_inf_lines: list[int] = []
    first_seen: dict[str, int] = {}
    for node in ast.walk(tree):
        for rule_id in _node_rules(node):
            hits.add((rule_id, node.lineno))
        if isinstance(node, ast.Attribute) and node.attr in _FP8_FN_DTYPES:
            fp8_lines.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr in _FP8_FNUZ_DTYPES:
            fnuz_lines.add(node.lineno)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in prose
            and _NVCC_FLAG_STRING.search(node.value)
        ):
            hits.add(("ROCM004", node.lineno))
        if isinstance(node, ast.Call):
            for const in _requirement_strings(node):
                for pattern, rule_id in _DEPENDENCIES:
                    if pattern.search(_strip_marker(const.value)):
                        hits.add((rule_id, const.lineno))
        if _is_negative_infinity(node):
            neg_inf_lines.append(node.lineno)
        for rule_id in _once_per_file_rules(node):
            first_seen[rule_id] = min(first_seen.get(rule_id, node.lineno), node.lineno)
    # A line that also names the fnuz dtype is choosing between the two, e.g.
    # "float8_e4m3fnuz if torch.version.hip else float8_e4m3fn".
    hits.update(("ROCM005", line) for line in fp8_lines if line not in fnuz_lines)
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


def _requirement_strings(call: ast.Call) -> Iterator[ast.Constant]:
    """String constants passed as requirements to setup(); extras' names excluded."""
    for keyword in call.keywords:
        if keyword.arg not in _REQUIREMENT_KEYWORDS:
            continue
        roots = keyword.value.values if isinstance(keyword.value, ast.Dict) else [keyword.value]
        for root in roots:
            for node in ast.walk(root):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    yield node
