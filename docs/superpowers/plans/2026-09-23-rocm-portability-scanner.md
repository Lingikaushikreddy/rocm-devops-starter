# ROCm Portability Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `rocm_portscan`, a stdlib-only static analyser that reports what in a PyTorch repository will not survive a move from CUDA to ROCm, where every BLOCKER claim is backed by a proof CI checks.

**Architecture:** Rules are pure data (`rules.py`). Two collectors hold all matching logic: an AST walker for `.py` files and a line-based text matcher for everything else (`collectors.py`). `scan.py` walks the tree and dispatches, `report.py` renders terminal/markdown/json, `__main__.py` is the CLI. `proofs.py` holds runnable fp8 checks that CI executes. Each rule is driven by a fixture directory that must trip it and a counter-fixture directory that must not.

**Tech Stack:** Python 3.10+ standard library (`ast`, `re`, `os`, `fnmatch`, `argparse`, `json`). pytest for tests. CPU PyTorch is used by `proofs.py` only, never by the scanner. ruff 0.16.8 for lint.

**Spec:** `docs/superpowers/specs/2026-09-21-rocm-portability-scanner-design.md` (revised 2026-09-23; read the Revision log first)

## Global Constraints

- Python floor: 3.10. CI runs 3.10 and 3.12. No 3.11+ features (`tomllib`, `StrEnum`, `except*`). No backslashes inside f-string expressions (not allowed before 3.12).
- The scanner (`rules.py`, `collectors.py`, `scan.py`, `report.py`, `__main__.py`) imports the standard library only. Only `proofs.py` may import `torch`, and only inside functions.
- The scanner never imports or executes the code it scans. Python is read with `ast.parse`, never `exec`/`import`.
- Exactly one BLOCKER rule exists: ROCM008, proof `doc:https://docs.cupy.dev/en/stable/install.html`. Do not promote any other rule. Promotion needs a proof someone has actually opened or run (spec Revision log).
- A `doc:` proof must start with `doc:https://`. A `selftest:` proof must name a function in `proofs.py` that returns `True`.
- CLI, verbatim from spec: `python -m rocm_portscan <path> [--format terminal|markdown|json] [--min-tier blocker|review|info] [--exclude GLOB]`. Exit `0` no blockers, `1` blockers found, `2` bad invocation.
- Published wording: BLOCKER findings may say "breaks". REVIEW findings are "worth checking", never "breaks on ROCm".
- ruff is pinned to `0.16.8` in CI. Lint with that version: `.venv-test/bin/ruff`.
- Commits use the repo owner's name only. **No `Co-Authored-By` or any Claude attribution lines.**
- Do not touch the README "Status, stated plainly" table. Nothing in this plan runs on AMD hardware.

## Review Focus

- **Python files that do not parse** (Python 2 `print 'x'`, cookiecutter `{{ }}` templates, stray NUL bytes): the scan skips that file and carries on. Test is in Task 5 (`test_unparseable_python_is_skipped`).
- **Binary, non-UTF-8 and very large files** in the tree (checkpoints, `.so`, generated sources): skipped or decoded with replacement, never a crash. Test is in Task 3 (`test_binary_and_oversized_files_are_skipped`, `test_invalid_utf8_does_not_crash`).
- **Symlinks, including loops and links pointing outside the root**: never followed, so a scan ends and stays inside the target. Test is in Task 3 (`test_symlinks_are_not_followed`).
- **A package named only in a comment or docstring** (`# install ROCm/apex on AMD`, `"""works without apex"""`): never fires. Covered by the portable fixtures `rocm001`, `rocm001-docstring`, `rocm008` and `rocm103` in Tasks 4 and 5.
- **The target is a single file, or does not exist**: a single file is scanned on its own. A missing path exits 2 with a message on stderr, not a traceback. Tests are in Task 3 (`test_single_file_root`) and Task 8 (`test_exit_2_on_missing_path`).

---

## File Structure

```
rocm_portscan/
  __init__.py      package docstring only
  rules.py         Tier, Rule, Finding, RULES table (pure data)
  proofs.py        selftest proofs (fp8 on CPU)
  collectors.py    collect_text (non-Python files), collect_python (AST)
  scan.py          scan(root, exclude) -> sorted, de-duplicated findings
  report.py        count_by_tier, render_terminal/markdown/json
  __main__.py      CLI
tests/
  test_rules.py        rule-table meta-tests
  test_proofs.py       every selftest proof still reproduces
  test_scan.py         traversal behaviour (collectors faked)
  test_collectors.py   collector mechanics (line numbers, comments, parse failures)
  test_fixtures.py     fixture harness: breaks/ must trip, portable/ must not
  test_report.py       renderers
  test_cli.py          exit codes, flags, module entry point
  fixtures/breaks/<rule>[-variant]/...     mini-repos that must trip <rule>
  fixtures/portable/<rule>[-variant]/...   mini-repos that must not trip <rule>
```

Fixture directory names map to rules as `case.split("-")[0].upper()`, so `rocm008-underscore` belongs to ROCM008.

---

### Task 1: Rule table, dev environment and CI job

**Files:**
- Create: `rocm_portscan/__init__.py`
- Create: `rocm_portscan/rules.py`
- Create: `tests/test_rules.py`
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `Tier` (IntEnum: `INFO=0, REVIEW=1, BLOCKER=2`), `Rule(id: str, tier: Tier, title: str, message: str, proof: str = "")`, `Finding(path: str, line: int, rule_id: str, snippet: str)` (frozen, ordered by path, line, rule_id, snippet), `RULES: dict[str, Rule]`, `_RULES: tuple[Rule, ...]`.

- [ ] **Step 1: Create the test venv** (skip if `.venv-test` already exists with torch, pytest and ruff)

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv-test/bin/pip install -q pytest ruff==0.16.8
```

`.venv-test/` is already in `.gitignore`.

- [ ] **Step 2: Write the failing test**

`tests/test_rules.py`:

```python
import re

from rocm_portscan.rules import _RULES, RULES, Tier


def test_rule_ids_are_unique_and_well_formed():
    assert len(RULES) == len(_RULES), "duplicate rule id"
    for rule_id, rule in RULES.items():
        assert rule.id == rule_id
        assert re.fullmatch(r"ROCM\d{3}", rule_id), rule_id


def test_every_blocker_names_a_proof():
    for rule in RULES.values():
        if rule.tier is Tier.BLOCKER:
            assert rule.proof, f"{rule.id} is a BLOCKER with no proof"


def test_proofs_are_a_known_kind():
    for rule in RULES.values():
        if rule.proof:
            kind = rule.proof.split(":", 1)[0]
            assert kind in {"doc", "selftest"}, rule.id


def test_doc_proofs_are_urls_not_labels():
    # The first draft of the spec cited docs from memory and four of five
    # were wrong. A doc proof has to be a link someone opened.
    for rule in RULES.values():
        if rule.proof.startswith("doc:"):
            assert rule.proof.startswith("doc:https://"), rule.id


def test_only_rocm008_is_a_blocker():
    # Promotion needs a verified proof (spec Revision log). Change this test
    # deliberately, in the same commit as the evidence.
    blockers = sorted(r.id for r in RULES.values() if r.tier is Tier.BLOCKER)
    assert blockers == ["ROCM008"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest tests/test_rules.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rocm_portscan'`

- [ ] **Step 4: Write the implementation**

`rocm_portscan/__init__.py`:

```python
"""Static analysis for 'will this PyTorch repository run on ROCm?'

Needs no GPU of any vendor. See docs/superpowers/specs/ for the design.
"""
```

`rocm_portscan/rules.py`:

```python
"""Rule definitions for the ROCm portability scanner.

Pure data. Matching logic lives in collectors.py: a rule here says what a
finding means and how sure we are of it, never how it is found.

A BLOCKER asserts a failure, so it must carry a proof: "doc:<https url>" or
"selftest:<function in proofs.py>". Tier answers a separate question - does
matching the pattern make the breakage certain? - so a rule with an airtight
proof is still REVIEW when the pattern is only correlated with the risk.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Tier(enum.IntEnum):
    INFO = 0
    REVIEW = 1
    BLOCKER = 2


@dataclass(frozen=True)
class Rule:
    id: str
    tier: Tier
    title: str
    message: str
    proof: str = ""


@dataclass(frozen=True, order=True)
class Finding:
    path: str  # relative to the scan root, forward slashes
    line: int
    rule_id: str
    snippet: str


_RULES = (
    Rule(
        "ROCM001",
        Tier.REVIEW,
        "`apex` dependency",
        "NVIDIA's apex does not build on ROCm, but AMD maintains a fork at "
        "github.com/ROCm/apex. Check which apex is installed and which of its "
        "fused kernels this code uses.",
    ),
    Rule(
        "ROCM002",
        Tier.REVIEW,
        "`transformer_engine` dependency",
        "Transformer Engine's ROCm support was not verified when this rule "
        "was written. Check its current state before relying on it.",
    ),
    Rule(
        "ROCM003",
        Tier.REVIEW,
        "CUDA sources or `CUDAExtension`",
        "On ROCm, torch.utils.cpp_extension hipifies CUDA sources "
        "automatically, so most extensions built this way compile unchanged. "
        "Hipify does not translate inline PTX asm, or code that assumes a "
        "32-wide warp (CDNA wavefronts are 64). Check for those.",
    ),
    Rule(
        "ROCM004",
        Tier.REVIEW,
        "nvcc architecture flags (`-gencode`, `compute_XX`, `sm_XX`)",
        "On ROCm builds these flags reach hipcc unchanged. Whether hipcc "
        "rejects them is unverified. ROCm targets are set with "
        "--offload-arch or PYTORCH_ROCM_ARCH.",
    ),
    Rule(
        "ROCM005",
        Tier.REVIEW,
        "hardcoded `float8_e4m3fn` / `float8_e5m2`",
        "gfx942 (MI300) executes e4m3fnuz, which caps at 240 and has no "
        "infinity. Values quantised for e4m3fn (max 448) overflow to NaN "
        "there. Check whether these weights will meet fnuz hardware.",
        proof="selftest:fn_max_overflows_fnuz",
    ),
    Rule(
        "ROCM006",
        Tier.REVIEW,
        "`-inf` in the same file as an fp8 dtype",
        "`-inf` and an fp8 dtype appear in the same file; check whether the "
        "fill reaches an fp8 tensor. In e4m3fnuz, -inf becomes NaN.",
        proof="selftest:fnuz_has_no_infinity",
    ),
    Rule(
        "ROCM007",
        Tier.REVIEW,
        "`pynvml` / `nvidia-ml-py`",
        "NVML cannot see AMD GPUs. Harmless if every call is guarded; the "
        "ROCm equivalent is amdsmi.",
    ),
    Rule(
        "ROCM008",
        Tier.BLOCKER,
        "`cupy-cuda*` wheel pin",
        "A cupy-cuda* wheel is built against CUDA and cannot drive an AMD "
        "GPU. CuPy's install docs route AMD users to a separate package, "
        "amd-cupy.",
        proof="doc:https://docs.cupy.dev/en/stable/install.html",
    ),
    Rule(
        "ROCM100",
        Tier.REVIEW,
        "`get_device_capability()` use",
        "On ROCm this cannot report an NVIDIA compute capability. Version "
        "gates written for NVIDIA architectures (e.g. `>= (8, 0)` for "
        "Ampere) need checking against what AMD GPUs report.",
    ),
    Rule(
        "ROCM101",
        Tier.REVIEW,
        "`flash_attn`",
        "Check that a ROCm build of flash-attn exists for the target GPU "
        "(AMD maintains one at github.com/ROCm/flash-attention).",
    ),
    Rule(
        "ROCM102",
        Tier.REVIEW,
        "`bitsandbytes`",
        "Check that the installed bitsandbytes build supports ROCm and the "
        "quantisation modes this code uses.",
    ),
    Rule(
        "ROCM103",
        Tier.REVIEW,
        "CUDA base image (`nvidia/cuda`, `nvcr.io/nvidia/`)",
        "This image ships the CUDA stack. The ROCm equivalents are the "
        "rocm/pytorch and rocm/dev-* images.",
    ),
    Rule(
        "ROCM104",
        Tier.REVIEW,
        "`torch.cuda.nvtx`",
        "NVTX ranges are NVIDIA profiler markers. On ROCm, check that the "
        "ranges still reach your profiler; PyTorch's ROCm-on-Windows builds "
        "stub them out.",
    ),
    Rule(
        "ROCM105",
        Tier.REVIEW,
        "inline assembly in a Triton kernel",
        "Inline assembly is target-specific; PTX will not assemble for an "
        "AMD GPU. Check for a portable tl.* equivalent.",
    ),
    Rule(
        "ROCM106",
        Tier.REVIEW,
        "`--gpus` / NVIDIA container runtime",
        "ROCm containers get GPUs through --device=/dev/kfd --device=/dev/dri, "
        "not --gpus or the NVIDIA runtime.",
    ),
    Rule(
        "ROCM107",
        Tier.REVIEW,
        "`xformers`",
        "Check that a ROCm build of xformers exists and supports the "
        "attention ops this code uses.",
    ),
    Rule(
        "ROCM200",
        Tier.INFO,
        '`init_process_group("nccl")`',
        'Portable. PyTorch on ROCm keeps the "nccl" backend name and routes '
        "it to RCCL.",
    ),
    Rule(
        "ROCM201",
        Tier.INFO,
        "`torch.cuda` usage",
        "Portable. ROCm builds of PyTorch reuse the torch.cuda namespace. "
        "Reported once per file.",
    ),
    Rule(
        "ROCM202",
        Tier.INFO,
        "fp16 / `.half()`",
        "Portable. fp16 is supported on ROCm. Reported once per file.",
    ),
)

RULES = {rule.id: rule for rule in _RULES}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest tests/test_rules.py -q`
Expected: `5 passed`

- [ ] **Step 6: Wire up Makefile and CI**

In `Makefile`, change the `.PHONY` line, replace the `lint` target and add `test`:

```make
.PHONY: help probe smoke report lint test build up clean
```

```make
lint:  ## Lint the scripts and the scanner
	ruff check scripts/ rocm_portscan/ tests/ --extend-exclude tests/fixtures

test:  ## Run the scanner test suite
	python -m pytest -q
```

In `.github/workflows/ci.yml`, change the lint job's last step to:

```yaml
      - run: ruff check scripts/ rocm_portscan/ tests/ --extend-exclude tests/fixtures
```

and add this job after `test:`:

```yaml
  portscan:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      # CPU torch is only for proofs.py. The scanner itself is stdlib-only,
      # and CI sets CI=true so the proof tests fail rather than skip without it.
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install pytest
      - run: python -m pytest -q
```

Fixtures are excluded from ruff because they deliberately contain unused imports of CUDA-only packages.

- [ ] **Step 7: Lint and commit**

```bash
.venv-test/bin/ruff check scripts/ rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/ tests/test_rules.py Makefile .github/workflows/ci.yml
git commit -m "Add the portability scanner's rule table and CI job"
```

---

### Task 2: Runnable proofs

**Files:**
- Create: `rocm_portscan/proofs.py`
- Create: `tests/test_proofs.py`

**Interfaces:**
- Consumes: `RULES` from Task 1 (the `selftest:` names `fn_max_overflows_fnuz`, `fnuz_has_no_infinity`).
- Produces: `proofs.fn_max_overflows_fnuz() -> bool`, `proofs.fnuz_has_no_infinity() -> bool`. `True` means the behaviour still reproduces.

- [ ] **Step 1: Write the failing test**

`tests/test_proofs.py`:

```python
import os

import pytest

from rocm_portscan import proofs
from rocm_portscan.rules import RULES

if os.environ.get("CI"):
    # CI must run the proofs, never skip them: a missing torch fails collection.
    import torch  # noqa: F401
else:
    pytest.importorskip("torch")

SELFTESTS = sorted(
    rule.proof.removeprefix("selftest:")
    for rule in RULES.values()
    if rule.proof.startswith("selftest:")
)


def test_selftest_list_is_not_empty():
    # Guards the parametrize below from silently collecting nothing.
    assert SELFTESTS


@pytest.mark.parametrize("name", SELFTESTS)
def test_selftest_proof_still_reproduces(name):
    fn = getattr(proofs, name, None)
    assert callable(fn), f"a rule cites selftest:{name} but proofs.py has no such function"
    assert fn() is True, f"selftest:{name} no longer reproduces; demote or delete the rule citing it"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest tests/test_proofs.py -q`
Expected: FAIL with `ImportError: cannot import name 'proofs'`

- [ ] **Step 3: Write the implementation**

`rocm_portscan/proofs.py`:

```python
"""Runnable proofs for rules that claim platform behaviour.

Each function returns True while the behaviour its rule describes still
reproduces on this machine. No GPU is needed: fp8 conversion runs on CPU. CI
runs every one of these, and one returning False means a rule is claiming
something no longer true - demote or delete that rule.

torch is imported inside each function so the scanner itself never needs it.
"""

from __future__ import annotations


def fn_max_overflows_fnuz() -> bool:
    """448 is an ordinary e4m3fn value and NaN in e4m3fnuz, the format gfx942 runs."""
    import torch

    if torch.finfo(torch.float8_e4m3fn).max != 448.0:
        return False
    if torch.finfo(torch.float8_e4m3fnuz).max != 240.0:
        return False
    converted = torch.tensor([448.0]).to(torch.float8_e4m3fnuz).float()
    return bool(torch.isnan(converted).all())


def fnuz_has_no_infinity() -> bool:
    """An -inf mask fill becomes NaN in e4m3fnuz instead of staying -inf."""
    import torch

    converted = torch.tensor([float("-inf")]).to(torch.float8_e4m3fnuz).float()
    return bool(torch.isnan(converted).all())
```

These behaviours were measured on torch 2.14.0 CPU on 2026-09-23: e4m3fnuz converts 448.0 → nan and -inf → nan; e4m3fn converts -inf → -448.0.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest tests/test_proofs.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add rocm_portscan/proofs.py tests/test_proofs.py
git commit -m "Add the fp8 proofs CI must keep reproducing"
```

---

### Task 3: Tree traversal

**Files:**
- Create: `rocm_portscan/collectors.py`
- Create: `rocm_portscan/scan.py`
- Create: `tests/test_scan.py`

**Interfaces:**
- Consumes: `Finding` from Task 1.
- Produces: `scan.scan(root: Path, exclude: Sequence[str] = ()) -> list[Finding]` (sorted, de-duplicated); `scan.MAX_BYTES = 2_000_000`; `scan.PRUNED_DIRS`; `collectors.collect_text(path: str, text: str) -> list[Finding]` (returns `[]` in this task; Task 4 gives it rules). `scan` looks up `collect_text` as a module global, so tests can monkeypatch `rocm_portscan.scan.collect_text`.

- [ ] **Step 1: Write the failing test**

`tests/test_scan.py`:

```python
import os

import pytest

from rocm_portscan import scan as scan_mod
from rocm_portscan.rules import Finding


def visited(monkeypatch, root, exclude=()):
    """Paths scan() handed to the text collector, with rules taken out of the picture."""
    seen = []

    def fake_collect_text(path, text):
        seen.append(path)
        return []

    monkeypatch.setattr(scan_mod, "collect_text", fake_collect_text)
    scan_mod.scan(root, exclude)
    return sorted(seen)


def write(path, content="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_empty_tree(tmp_path):
    assert scan_mod.scan(tmp_path) == []


def test_visits_nested_files_with_relative_posix_paths(monkeypatch, tmp_path):
    write(tmp_path / "a" / "b" / "requirements.txt")
    write(tmp_path / "setup.py")
    assert visited(monkeypatch, tmp_path) == ["a/b/requirements.txt", "setup.py"]


def test_prunes_vendor_and_vcs_dirs(monkeypatch, tmp_path):
    for d in (".git", "node_modules", "__pycache__", ".venv", "site-packages"):
        write(tmp_path / d / "requirements.txt")
    write(tmp_path / "keep.txt")
    assert visited(monkeypatch, tmp_path) == ["keep.txt"]


def test_exclude_glob_prunes_directories_and_files(monkeypatch, tmp_path):
    write(tmp_path / "vendor" / "deep" / "requirements.txt")
    write(tmp_path / "docs" / "notes.sh")
    write(tmp_path / "keep.sh")
    got = visited(monkeypatch, tmp_path, exclude=["vendor", "*.sh"])
    assert got == []
    got = visited(monkeypatch, tmp_path, exclude=["vendor"])
    assert got == ["docs/notes.sh", "keep.sh"]


def test_binary_and_oversized_files_are_skipped(monkeypatch, tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01\x02" * 10)
    (tmp_path / "huge.txt").write_text("a" * (scan_mod.MAX_BYTES + 1))
    write(tmp_path / "ok.txt")
    assert visited(monkeypatch, tmp_path) == ["ok.txt"]


def test_invalid_utf8_does_not_crash(monkeypatch, tmp_path):
    (tmp_path / "latin1.txt").write_bytes("caf\xe9\n".encode("latin-1"))
    assert visited(monkeypatch, tmp_path) == ["latin1.txt"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_symlinks_are_not_followed(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    write(outside / "secret.txt")
    root = tmp_path / "root"
    write(root / "real.txt")
    (root / "link_to_outside").symlink_to(outside, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    (root / "file_link.txt").symlink_to(outside / "secret.txt")
    assert visited(monkeypatch, root) == ["real.txt"]


def test_single_file_root(monkeypatch, tmp_path):
    target = tmp_path / "requirements.txt"
    write(target)
    assert visited(monkeypatch, target) == ["requirements.txt"]


def test_findings_are_sorted_and_deduplicated(monkeypatch, tmp_path):
    write(tmp_path / "b.txt")
    write(tmp_path / "a.txt")

    def fake(path, text):
        f = Finding(path, 1, "ROCM001", "x")
        return [f, f]

    monkeypatch.setattr(scan_mod, "collect_text", fake)
    assert scan_mod.scan(tmp_path) == [
        Finding("a.txt", 1, "ROCM001", "x"),
        Finding("b.txt", 1, "ROCM001", "x"),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest tests/test_scan.py -q`
Expected: FAIL with `ImportError: cannot import name 'scan'`

- [ ] **Step 3: Write the implementation**

`rocm_portscan/collectors.py`:

```python
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
```

`rocm_portscan/scan.py`:

```python
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

from .collectors import collect_text
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
    return collect_text(rel, text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest tests/test_scan.py -q`
Expected: `9 passed`

- [ ] **Step 5: Lint and commit**

```bash
.venv-test/bin/ruff check rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/collectors.py rocm_portscan/scan.py tests/test_scan.py
git commit -m "Add tree traversal that skips binaries and never follows symlinks"
```

---

### Task 4: Text rules and the fixture harness

**Files:**
- Modify: `rocm_portscan/collectors.py` (replace `collect_text`, add module-level patterns)
- Create: `tests/test_fixtures.py`
- Create: `tests/test_collectors.py`
- Create: the fixture files listed in Step 1

**Interfaces:**
- Consumes: `scan.scan` (Task 3), `RULES`, `Tier`, `Finding` (Task 1).
- Produces: `collect_text` implementing ROCM001/002/003/004/007/008/101/102/103/106/107 on non-Python files. `collectors._snippet(raw: str) -> str` (stripped, first 120 chars), reused by Task 5.

- [ ] **Step 1: Write the fixtures**

Each block is one file. Paths are relative to `tests/fixtures/`.

`breaks/rocm001/requirements.txt`
```
torch>=2.3
apex @ git+https://github.com/NVIDIA/apex.git
```

`portable/rocm001/requirements.txt`
```
# apex is optional; on AMD install ROCm/apex instead
apex-lite==0.1
rapex==1.0
```

`breaks/rocm002/pyproject.toml`
```
[project]
name = "demo"
dependencies = ["transformer_engine[pytorch]>=1.9"]
```

`portable/rocm002/pyproject.toml`
```
[project]
name = "demo"
dependencies = ["transformers>=4.40"]
```

`breaks/rocm003/kernel.cu`
```
__global__ void scale(float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= a;
}
```

`portable/rocm003/kernel.cpp`
```
void scale(float* x, float a, int n) {
  for (int i = 0; i < n; ++i) x[i] *= a;
}
```

`breaks/rocm004/setup.py`
```
NVCC_FLAGS = ["-O3", "-gencode", "arch=compute_90,code=sm_90"]
```

`portable/rocm004/setup.py`
```
NVCC_FLAGS = ["-O3"]
sm_scale = 0.125
```

`breaks/rocm007/requirements.txt`
```
nvidia-ml-py>=12.535
```

`portable/rocm007/requirements.txt`
```
amdsmi
psutil
```

`breaks/rocm008/requirements.txt`
```
cupy-cuda12x==13.3.0
```

`breaks/rocm008-underscore/requirements.txt`
```
cupy_cuda11x
```

`portable/rocm008/requirements.txt`
```
# cupy-cuda12x on NVIDIA only
cupy
amd-cupy
```

`breaks/rocm101/requirements.txt`
```
flash_attn==2.6.3
```

`portable/rocm101/requirements.txt`
```
flash-attention-softmax-n
```

`breaks/rocm102/requirements.txt`
```
bitsandbytes>=0.43
```

`portable/rocm102/requirements.txt`
```
accelerate
```

`breaks/rocm103/Dockerfile`
```
FROM nvcr.io/nvidia/pytorch:24.05-py3
```

`breaks/rocm103-cuda/Dockerfile.train`
```
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-devel-ubuntu22.04
```

`portable/rocm103/Dockerfile`
```
# Previously: FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM rocm/pytorch:latest
```

`breaks/rocm106/run.sh`
```
docker run --gpus all -it trainer
```

`breaks/rocm106-compose/docker-compose.yml`
```
services:
  train:
    runtime: nvidia
```

`portable/rocm106/run.sh`
```
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it trainer
```

`breaks/rocm107/requirements.txt`
```
xformers
```

`portable/rocm107/requirements.txt`
```
transformers
```

- [ ] **Step 2: Write the failing tests**

`tests/test_fixtures.py`:

```python
"""Every rule is pinned by a directory that must trip it and one that must not.

Directory names map to rules: breaks/rocm008-underscore/ belongs to ROCM008.
"""

from pathlib import Path

import pytest

from rocm_portscan.rules import RULES, Tier
from rocm_portscan.scan import scan

FIXTURES = Path(__file__).parent / "fixtures"
BREAKS = FIXTURES / "breaks"
PORTABLE = FIXTURES / "portable"


def cases(base):
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def rule_of(case):
    return case.split("-")[0].upper()


@pytest.mark.parametrize("case", cases(BREAKS))
def test_breaks_fixture_trips_its_rule(case):
    fired = {f.rule_id for f in scan(BREAKS / case)}
    assert rule_of(case) in fired, f"{case}: fired {sorted(fired)}"


@pytest.mark.parametrize("case", cases(PORTABLE))
def test_portable_fixture_does_not_trip_its_rule(case):
    found = [f for f in scan(PORTABLE / case) if f.rule_id == rule_of(case)]
    assert found == []


def test_portable_tree_has_zero_blockers():
    blockers = [f for f in scan(PORTABLE) if RULES[f.rule_id].tier is Tier.BLOCKER]
    assert blockers == []
```

`tests/test_collectors.py`:

```python
from rocm_portscan.collectors import collect_text
from rocm_portscan.rules import Finding


def test_line_number_and_snippet_keep_the_raw_line():
    text = "torch\ncupy-cuda12x==13.3.0  # gpu\n"
    assert collect_text("requirements.txt", text) == [
        Finding("requirements.txt", 2, "ROCM008", "cupy-cuda12x==13.3.0  # gpu")
    ]


def test_crlf_line_endings():
    found = collect_text("requirements.txt", "torch\r\ncupy-cuda12x\r\n")
    assert [(f.line, f.rule_id) for f in found] == [(2, "ROCM008")]


def test_unknown_file_types_are_ignored():
    assert collect_text("README.md", "pip install cupy-cuda12x apex\n") == []


def test_one_finding_per_rule_per_line():
    found = collect_text("requirements.txt", "nvidia-ml-py3 pynvml\n")
    assert [f.rule_id for f in found] == ["ROCM007"]


def test_nested_requirements_files_are_dependency_files():
    found = collect_text("requirements/requirements_nlp.txt", "apex\n")
    assert [f.rule_id for f in found] == ["ROCM001"]


def test_cuda_source_reports_once_at_line_one():
    found = collect_text("csrc/ops.cuh", "#pragma once\n__device__ int f();\n")
    assert found == [Finding("csrc/ops.cuh", 1, "ROCM003", "CUDA source file")]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv-test/bin/python -m pytest tests/test_fixtures.py tests/test_collectors.py -q`
Expected: FAIL. Every `test_breaks_fixture_trips_its_rule[...]` fails with `fired []`, and so do the `test_collectors.py` tests that expect findings. The portable tests pass trivially.

- [ ] **Step 4: Write the implementation**

Replace the body of `rocm_portscan/collectors.py` below the module docstring with:

```python
from __future__ import annotations

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-test/bin/python -m pytest -q`
Expected: all pass. If a portable fixture trips its rule, fix the pattern, not the fixture. The portable fixtures are the false-positive guard.

- [ ] **Step 6: Lint and commit**

```bash
.venv-test/bin/ruff check rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/collectors.py tests/test_fixtures.py tests/test_collectors.py tests/fixtures
git commit -m "Add text rules for dependencies, build flags, images and CUDA sources"
```

---

### Task 5: Python rules for imports and calls

**Files:**
- Modify: `rocm_portscan/collectors.py` (add `import ast`, `collect_python` and helpers)
- Modify: `rocm_portscan/scan.py` (dispatch `.py` files to `collect_python`)
- Modify: `tests/test_collectors.py` (append tests)
- Create: the fixture files listed in Step 1

**Interfaces:**
- Consumes: `_snippet` (Task 4), `Finding` (Task 1).
- Produces: `collect_python(path: str, source: str) -> list[Finding]` implementing ROCM001/002/003/007/100/101/102/104/105/107. `_node_rules(node: ast.AST) -> list[str]` and `_call_name(node: ast.Call) -> str`, extended in Task 6.

- [ ] **Step 1: Write the fixtures**

`breaks/rocm001-import/train.py`
```python
from apex.optimizers import FusedAdam
```

`portable/rocm001-docstring/train.py`
```python
"""Works with or without apex."""
# import apex
import torch
```

`portable/rocm001-relative/pkg/train.py`
```python
from .apex import fused_step
```

`breaks/rocm002-import/model.py`
```python
import transformer_engine.pytorch as te
```

`breaks/rocm003-extension/setup.py`
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_ops",
    ext_modules=[CUDAExtension("fused_ops", ["fused_ops.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
```

`portable/rocm003-cpp/setup.py`
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fused_ops",
    ext_modules=[CppExtension("fused_ops", ["fused_ops.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
```

`breaks/rocm007-import/monitor.py`
```python
import pynvml

pynvml.nvmlInit()
```

`portable/rocm007-amdsmi/monitor.py`
```python
import amdsmi

amdsmi.amdsmi_init()
```

`breaks/rocm100/gate.py`
```python
import torch

if torch.cuda.get_device_capability() >= (8, 0):
    use_bf16 = True
```

`portable/rocm100/gate.py`
```python
import torch

use_bf16 = torch.cuda.is_bf16_supported()
```

`breaks/rocm101-import/attn.py`
```python
from flash_attn import flash_attn_func
```

`portable/rocm101-sdpa/attn.py`
```python
import torch.nn.functional as F

attention = F.scaled_dot_product_attention
```

`breaks/rocm102-import/quant.py`
```python
import bitsandbytes as bnb
```

`breaks/rocm104/profile.py`
```python
import torch

torch.cuda.nvtx.range_push("step")
torch.cuda.nvtx.range_pop()
```

`breaks/rocm104-from/profile.py`
```python
from torch.cuda import nvtx

nvtx.range_push("step")
```

`portable/rocm104/profile.py`
```python
import torch

with torch.profiler.record_function("step"):
    pass
```

`breaks/rocm105/kernel.py`
```python
import triton
import triton.language as tl


@triton.jit
def fast_exp(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    y = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(out_ptr, y)
```

`portable/rocm105/kernel.py`
```python
import triton
import triton.language as tl


@triton.jit
def fast_exp(x_ptr, out_ptr):
    x = tl.load(x_ptr)
    tl.store(out_ptr, tl.exp2(x))
```

`breaks/rocm107-import/attn.py`
```python
import xformers.ops as xops
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_collectors.py`, change the first import line to
`from rocm_portscan.collectors import collect_python, collect_text` (ruff's
E402 rejects imports appended mid-file), then append:

```python
def test_unparseable_python_is_skipped():
    assert collect_python("py2.py", "import apex\nprint 'hello'\n") == []
    assert collect_python("tmpl.py", "import {{ cookiecutter.pkg }}\n") == []


def test_python_line_number_and_snippet():
    source = "import os\n\nimport pynvml  # telemetry\n"
    assert collect_python("m.py", source) == [
        Finding("m.py", 3, "ROCM007", "import pynvml  # telemetry")
    ]


def test_attribute_named_like_a_package_does_not_fire():
    assert collect_python("m.py", "cfg.apex = True\nx = obj.flash_attn\n") == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv-test/bin/python -m pytest tests/test_fixtures.py tests/test_collectors.py -q`
Expected: FAIL. `test_collectors.py` fails with `ImportError: cannot import name 'collect_python'`, and the new `breaks/` cases (`rocm001-import`, `rocm100`, `rocm104`, ...) report `fired []`.

- [ ] **Step 4: Write the implementation**

In `rocm_portscan/collectors.py`, add `import ast` to the imports and add below `_snippet`:

```python
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
```

In `rocm_portscan/scan.py`, change the import and the last lines of `_scan_file`:

```python
from .collectors import collect_python, collect_text
```

```python
    text = data.decode("utf-8", errors="replace")
    found = collect_text(rel, text)
    if rel.endswith(".py"):
        found += collect_python(rel, text)
    return found
```

`setup.py` goes through both collectors: text for dependency strings and build flags, AST for imports and `CUDAExtension`. `scan()` already de-duplicates.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-test/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
.venv-test/bin/ruff check rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/ tests/test_collectors.py tests/fixtures
git commit -m "Add AST rules for CUDA-only imports, extensions and capability gates"
```

---

### Task 6: fp8 rules, INFO rules and fixture coverage

**Files:**
- Modify: `rocm_portscan/collectors.py` (`collect_python`, `_node_rules`, three new helpers)
- Modify: `tests/test_collectors.py` (append tests)
- Modify: `tests/test_fixtures.py` (append coverage test)
- Create: the fixture files listed in Step 1

**Interfaces:**
- Consumes: `collect_python`, `_node_rules`, `_call_name` (Task 5).
- Produces: ROCM005, ROCM006, ROCM200, ROCM201, ROCM202 in `collect_python`. After this task every rule in `RULES` has at least one `breaks/` and one `portable/` fixture.

- [ ] **Step 1: Write the fixtures**

`breaks/rocm005/quant.py`
```python
import torch


def quantise(w):
    return w.to(torch.float8_e4m3fn)
```

`portable/rocm005/quant.py`
```python
import torch


def quantise(w):
    return w.to(torch.float8_e4m3fnuz)
```

`breaks/rocm006/attn.py`
```python
import torch


def masked_scores(scores, mask):
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.to(torch.float8_e4m3fn)
```

`breaks/rocm006-math/attn.py`
```python
import math

import torch


def masked_scores(scores, mask):
    scores = scores.masked_fill(~mask, -math.inf)
    return scores.to(torch.float8_e5m2)
```

`portable/rocm006/attn.py`
```python
import torch


def masked_scores(scores, mask):
    return scores.masked_fill(~mask, float("-inf")).softmax(-1)
```

`portable/rocm006-separate/attn.py`
```python
def masked_scores(scores, mask):
    return scores.masked_fill(~mask, float("-inf")).softmax(-1)
```

`portable/rocm006-separate/quant.py`
```python
import torch


def quantise(w):
    return w.to(torch.float8_e4m3fn)
```

`breaks/rocm200/dist.py`
```python
import torch.distributed as dist

dist.init_process_group("nccl")
```

`breaks/rocm200-kw/dist.py`
```python
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://")
```

`portable/rocm200/dist.py`
```python
import torch.distributed as dist

dist.init_process_group("gloo")
```

`breaks/rocm201/train.py`
```python
import torch

if torch.cuda.is_available():
    torch.cuda.synchronize()
```

`portable/rocm201/train.py`
```python
import torch

device = torch.device("cpu")
```

`breaks/rocm202/train.py`
```python
def to_fp16(model):
    return model.half()
```

`portable/rocm202/train.py`
```python
def to_bf16(model):
    return model.bfloat16()
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_collectors.py`:

```python
def rule_lines(findings, rule_id):
    return [f.line for f in findings if f.rule_id == rule_id]


def test_once_per_file_rules_report_first_line_only():
    source = "import torch\ntorch.cuda.synchronize()\nx = torch.cuda.current_device()\n"
    assert rule_lines(collect_python("m.py", source), "ROCM201") == [2]


def test_neg_inf_fires_at_each_fill_when_file_has_fp8():
    source = (
        "import torch\n"
        "a = x.masked_fill(m, float('-inf'))\n"
        "b = y.masked_fill(m, -float('inf'))\n"
        "c = z.to(torch.float8_e4m3fn)\n"
    )
    found = collect_python("m.py", source)
    assert rule_lines(found, "ROCM006") == [2, 3]
    assert rule_lines(found, "ROCM005") == [4]


def test_fnuz_dtypes_are_not_flagged():
    source = "import torch\nz = x.to(torch.float8_e4m3fnuz)\nw = y.to(torch.float8_e5m2fnuz)\n"
    assert collect_python("m.py", source) == []


def test_nccl_is_case_insensitive_and_other_backends_are_silent():
    assert rule_lines(collect_python("d.py", "init_process_group('NCCL')\n"), "ROCM200") == [1]
    assert collect_python("d.py", "init_process_group(backend='gloo')\n") == []
```

Append to `tests/test_fixtures.py`:

```python
def test_every_rule_has_both_fixture_kinds():
    have_breaks = {rule_of(c) for c in cases(BREAKS)}
    have_portable = {rule_of(c) for c in cases(PORTABLE)}
    assert sorted(set(RULES) - have_breaks) == [], "rules with no breaks/ fixture"
    assert sorted(set(RULES) - have_portable) == [], "rules with no portable/ fixture"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv-test/bin/python -m pytest tests/test_fixtures.py tests/test_collectors.py -q`
Expected: FAIL. The new `breaks/` cases (`rocm005`, `rocm006`, `rocm200`, `rocm201`, `rocm202`) report `fired [...]` without their rule, and the new collector tests fail on empty lists.

- [ ] **Step 4: Write the implementation**

In `rocm_portscan/collectors.py`, add below `_CALL_RULES`:

```python
# The OCP formats. Their fnuz counterparts are what gfx942 executes.
_FP8_FN_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2"})
```

Replace `collect_python` with:

```python
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
```

In `_node_rules`, replace the `ast.Call` branch with:

```python
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "init_process_group" and _names_nccl(node):
            return ["ROCM200"]
        rule_id = _CALL_RULES.get(name)
        return [rule_id] if rule_id else []
```

Add these helpers below `_call_name`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv-test/bin/python -m pytest -q`
Expected: all pass, including `test_every_rule_has_both_fixture_kinds`.

- [ ] **Step 6: Lint and commit**

```bash
.venv-test/bin/ruff check rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/collectors.py tests/
git commit -m "Add fp8 and INFO rules; require both fixture kinds for every rule"
```

---

### Task 7: Renderers

**Files:**
- Create: `rocm_portscan/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `RULES`, `Tier`, `Finding` (Task 1).
- Produces: `count_by_tier(findings) -> dict[Tier, int]`, `render_terminal(findings, min_tier=Tier.REVIEW) -> str`, `render_json(findings, min_tier=Tier.REVIEW) -> str`, `render_markdown(findings, min_tier=Tier.REVIEW, target="") -> str`. Renderers keep the order they are given. Summary counts always cover every finding, including ones hidden by `min_tier`.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:

```python
import json

from rocm_portscan.report import count_by_tier, render_json, render_markdown, render_terminal
from rocm_portscan.rules import Finding, Tier

FINDINGS = [
    Finding("requirements.txt", 3, "ROCM008", "cupy-cuda12x==13.3.0"),
    Finding("train.py", 1, "ROCM001", "import apex"),
    Finding("train.py", 9, "ROCM201", "torch.cuda.synchronize()"),
]


def test_count_by_tier():
    assert count_by_tier(FINDINGS) == {Tier.BLOCKER: 1, Tier.REVIEW: 1, Tier.INFO: 1}


def test_terminal_hides_info_by_default_but_counts_it():
    out = render_terminal(FINDINGS)
    assert "requirements.txt:3: ROCM008 [BLOCKER]" in out
    assert "train.py:1: ROCM001 [REVIEW]" in out
    assert "ROCM201" not in out
    assert out.rstrip().endswith("1 blocker(s), 1 to review, 1 info")


def test_terminal_min_tier_info_shows_everything():
    assert "train.py:9: ROCM201 [INFO]" in render_terminal(FINDINGS, Tier.INFO)


def test_json_is_valid_and_carries_proofs():
    data = json.loads(render_json(FINDINGS, Tier.INFO))
    assert data["summary"] == {"blocker": 1, "review": 1, "info": 1}
    assert [f["rule"] for f in data["findings"]] == ["ROCM008", "ROCM001", "ROCM201"]
    assert data["findings"][0]["proof"] == "doc:https://docs.cupy.dev/en/stable/install.html"
    assert data["findings"][0]["tier"] == "BLOCKER"
    assert data["findings"][1]["proof"] is None


def test_markdown_never_files_a_review_finding_under_breaks():
    md = render_markdown(FINDINGS, Tier.REVIEW, target="demo")
    breaks, review = md.split("## Worth checking")
    assert "ROCM008" in breaks and "ROCM001" not in breaks
    assert "ROCM001" in review
    assert "https://docs.cupy.dev/en/stable/install.html" in md
    assert "ROCM201" not in md
    assert md.startswith("# ROCm portability report: demo\n")
    assert "Nothing here was run on AMD hardware." in md


def test_markdown_says_none_found_when_no_blockers():
    md = render_markdown(FINDINGS[1:], Tier.REVIEW, target="demo")
    assert "## Breaks on ROCm\n\nNone found." in md


def test_markdown_truncates_long_location_lists():
    many = [Finding(f"f{i:02}.py", 1, "ROCM001", "import apex") for i in range(25)]
    md = render_markdown(many, Tier.REVIEW, target="demo")
    assert "f19.py" in md and "f20.py" not in md
    assert "... and 5 more" in md


def test_markdown_neutralises_backticks_in_snippets():
    md = render_markdown([Finding("a.sh", 1, "ROCM106", "docker run `--gpus all`")], target="demo")
    assert "docker run '--gpus all'" in md


def test_selftest_proof_is_described_as_runnable():
    md = render_markdown([Finding("q.py", 2, "ROCM005", "w.to(torch.float8_e4m3fn)")], target="demo")
    assert "rocm_portscan.proofs.fn_max_overflows_fnuz()" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rocm_portscan.report'`

- [ ] **Step 3: Write the implementation**

`rocm_portscan/report.py`:

```python
"""Render findings for a terminal, a forum post, or another program.

Renderers hold no opinions about severity: tier comes from the rule table.
The markdown report is what gets published, so its wording follows the spec -
only the BLOCKER section says anything breaks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from .rules import RULES, Finding, Tier

HEADINGS = {
    Tier.BLOCKER: "Breaks on ROCm",
    Tier.REVIEW: "Worth checking",
    Tier.INFO: "Portable, noted for reference",
}
MAX_LOCATIONS = 20


def count_by_tier(findings: Sequence[Finding]) -> dict[Tier, int]:
    counts = {tier: 0 for tier in Tier}
    for f in findings:
        counts[RULES[f.rule_id].tier] += 1
    return counts


def render_terminal(findings: Sequence[Finding], min_tier: Tier = Tier.REVIEW) -> str:
    out = []
    for f in _visible(findings, min_tier):
        rule = RULES[f.rule_id]
        out.append(f"{f.path}:{f.line}: {rule.id} [{rule.tier.name}] {rule.title}")
    out.append(_summary(findings))
    return "\n".join(out) + "\n"


def render_json(findings: Sequence[Finding], min_tier: Tier = Tier.REVIEW) -> str:
    payload = {
        "summary": {tier.name.lower(): n for tier, n in count_by_tier(findings).items()},
        "findings": [
            {
                "rule": f.rule_id,
                "tier": RULES[f.rule_id].tier.name,
                "title": RULES[f.rule_id].title,
                "path": f.path,
                "line": f.line,
                "snippet": f.snippet,
                "proof": RULES[f.rule_id].proof or None,
            }
            for f in _visible(findings, min_tier)
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_markdown(
    findings: Sequence[Finding], min_tier: Tier = Tier.REVIEW, target: str = ""
) -> str:
    out = [
        f"# ROCm portability report: {target}",
        "",
        "Static analysis by rocm_portscan. Nothing here was run on AMD hardware.",
        'Only "Breaks on ROCm" asserts a failure, and each rule there cites its proof.',
        "Everything else is a pointer for a person to check.",
        "",
        f"**Summary:** {_summary(findings)}",
    ]
    by_rule: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_rule[f.rule_id].append(f)
    for tier in sorted(Tier, reverse=True):
        if tier < min_tier:
            continue
        out += ["", f"## {HEADINGS[tier]}"]
        rule_ids = sorted(r for r in by_rule if RULES[r].tier is tier)
        if not rule_ids:
            out += ["", "None found."]
            continue
        for rule_id in rule_ids:
            rule = RULES[rule_id]
            out += ["", f"### {rule.id}: {rule.title}", "", rule.message]
            if rule.proof:
                out += ["", f"Proof: {_proof_text(rule.proof)}"]
            out.append("")
            hits = by_rule[rule_id]
            for f in hits[:MAX_LOCATIONS]:
                out.append(f"- `{f.path}:{f.line}`: `{_code(f.snippet)}`")
            if len(hits) > MAX_LOCATIONS:
                out.append(f"- ... and {len(hits) - MAX_LOCATIONS} more")
    return "\n".join(out) + "\n"


def _visible(findings: Sequence[Finding], min_tier: Tier) -> list[Finding]:
    return [f for f in findings if RULES[f.rule_id].tier >= min_tier]


def _summary(findings: Sequence[Finding]) -> str:
    c = count_by_tier(findings)
    return f"{c[Tier.BLOCKER]} blocker(s), {c[Tier.REVIEW]} to review, {c[Tier.INFO]} info"


def _proof_text(proof: str) -> str:
    kind, _, value = proof.partition(":")
    if kind == "doc":
        return value
    return f"runnable check `rocm_portscan.proofs.{value}()`, executed in CI"


def _code(text: str) -> str:
    # A backtick in a snippet would close the inline code span early.
    return text.replace("`", "'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-test/bin/python -m pytest tests/test_report.py -q`
Expected: `9 passed`

- [ ] **Step 5: Lint and commit**

```bash
.venv-test/bin/ruff check rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/report.py tests/test_report.py
git commit -m "Add terminal, markdown and JSON renderers"
```

---

### Task 8: CLI and README

**Files:**
- Create: `rocm_portscan/__main__.py`
- Create: `tests/test_cli.py`
- Modify: `README.md` (add a section; do not touch the status table)

**Interfaces:**
- Consumes: `scan.scan` (Task 3), `report.*` (Task 7), `Tier` (Task 1).
- Produces: `__main__.main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rocm_portscan.__main__ import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_exit_1_when_blockers_found(tmp_path, capsys):
    (tmp_path / "requirements.txt").write_text("cupy-cuda12x\n")
    assert main([str(tmp_path)]) == 1
    assert "ROCM008" in capsys.readouterr().out


def test_exit_0_when_only_review_findings(tmp_path, capsys):
    (tmp_path / "requirements.txt").write_text("apex\n")
    assert main([str(tmp_path)]) == 0
    assert "ROCM001" in capsys.readouterr().out


def test_exit_code_ignores_min_tier_filter(tmp_path):
    (tmp_path / "requirements.txt").write_text("cupy-cuda12x\n")
    assert main([str(tmp_path), "--min-tier", "blocker"]) == 1


def test_exit_2_on_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope")]) == 2
    assert "no such file or directory" in capsys.readouterr().err


def test_exit_2_on_bad_format(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path), "--format", "xml"])
    assert exc.value.code == 2


def test_single_file_target(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("cupy-cuda12x\n")
    assert main([str(req)]) == 1


def test_exclude_can_be_repeated(tmp_path):
    for d in ("vendor", "third_party"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "requirements.txt").write_text("cupy-cuda12x\n")
    assert main([str(tmp_path), "--exclude", "vendor", "--exclude", "third_party"]) == 0


def test_json_format(tmp_path, capsys):
    (tmp_path / "requirements.txt").write_text("apex\n")
    main([str(tmp_path), "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["findings"][0]["rule"] == "ROCM001"


def test_markdown_format_names_the_target(tmp_path, capsys):
    target = tmp_path / "myrepo"
    target.mkdir()
    main([str(target), "--format", "markdown"])
    assert capsys.readouterr().out.startswith("# ROCm portability report: myrepo\n")


def test_runs_as_module(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "rocm_portscan", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "0 blocker(s)" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-test/bin/python -m pytest tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rocm_portscan.__main__'`

- [ ] **Step 3: Write the implementation**

`rocm_portscan/__main__.py`:

```python
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
```

- [ ] **Step 4: Run the full suite**

Run: `.venv-test/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Smoke-run it on this repo**

Run: `.venv-test/bin/python -m rocm_portscan . --exclude tests/fixtures; echo "exit=$?"`
Expected: exit 0. Read every finding it prints. The repo's own `docker/` and `scripts/` should produce nothing above INFO, apart from anything genuinely CUDA-specific. If a finding here looks wrong, it is a false positive. Add a `portable/` fixture that reproduces it, fix the rule, and rerun before going on.

- [ ] **Step 6: Add the README section**

Add this section to `README.md` after the "Status, stated plainly" section. Leave the table alone:

````markdown
## Portability scanner

`rocm_portscan` answers "will this PyTorch repository run on ROCm?" without an
AMD GPU. It reads dependencies, build flags, Dockerfiles, CUDA sources and
Python (via the AST, never by importing it).

```bash
python -m rocm_portscan path/to/repo                    # terminal
python -m rocm_portscan path/to/repo --format markdown  # a postable report
python -m rocm_portscan path/to/repo --min-tier info --exclude 'third_party'
```

Findings come in three tiers. **BLOCKER** means it breaks, and every blocker
rule cites a proof: a vendor doc, or a check in `rocm_portscan/proofs.py` that
CI runs on every push. **REVIEW** means worth checking. **INFO** means it is
portable, with a note on why. Exit code 1 means blockers were found, so it
works as a CI gate.

Every finding comes from static analysis and nothing here has run on AMD
hardware yet. The rule list and the reasoning behind each tier are in
`docs/superpowers/specs/2026-09-21-rocm-portability-scanner-design.md`.
````

Also add `test` to the list of `make` targets if the README lists them.

- [ ] **Step 7: Lint and commit**

```bash
.venv-test/bin/ruff check scripts/ rocm_portscan/ tests/ --extend-exclude tests/fixtures
git add rocm_portscan/__main__.py tests/test_cli.py README.md
git commit -m "Add the rocm_portscan CLI and document it"
```

- [ ] **Step 8: Push and confirm CI is green**

```bash
git push
gh run watch --exit-status $(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')
```

Expected: `lint`, `test` (3.10, 3.12), `portscan` (3.10, 3.12) and `compose` all pass. If `portscan` fails on 3.10, look first for a 3.11+ feature (see Global Constraints).

---

### Task 9: Dry run against NeMo (local only, nothing published)

`NVIDIA/NeMo` now redirects to `NVIDIA-NeMo/Speech`, the renamed monorepo. That is the target. The org's other repos (Automodel, RL, ...) are a later decision for the owner.

**Files:**
- Create (gitignored, not committed): `results/nemo-speech-report.md`, `results/nemo-speech.json`

- [ ] **Step 1: Clone outside the repo**

```bash
git clone --depth 1 https://github.com/NVIDIA-NeMo/Speech.git ~/src/NeMo-Speech
git -C ~/src/NeMo-Speech rev-parse HEAD
```

Record the commit hash; the report has to name what it scanned.

- [ ] **Step 2: Time the scan (success criterion 4: under a minute)**

```bash
mkdir -p results
time .venv-test/bin/python -m rocm_portscan ~/src/NeMo-Speech --format json > results/nemo-speech.json; echo "exit=$?"
.venv-test/bin/python -m rocm_portscan ~/src/NeMo-Speech --format markdown > results/nemo-speech-report.md
```

If the scan takes over 60 s, profile with `python -m cProfile -s cumtime -m rocm_portscan ~/src/NeMo-Speech | head -30` before optimising anything.

- [ ] **Step 3: Check every finding by hand**

For each BLOCKER: open the file and line and confirm the pin is real, not inside a comment or an NVIDIA-only optional extra. For each REVIEW rule: open at least three hits and check that they are what the rule says they are. Every false positive becomes a `portable/` fixture plus a rule fix, in its own commit, before this step is repeated.

- [ ] **Step 4: Stop and hand over**

Do not publish, post or open issues. Give the owner the rule counts, the commit hash, the scan time, and the false positives found and fixed. Publishing claims about NeMo is the owner's decision (standing rule 4 in `~/Desktop/clear.md`).
