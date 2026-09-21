# ROCm portability scanner — design

**Date:** 2026-09-21
**Status:** approved, not yet implemented
**Repo:** rocm-devops-starter

## Problem

There is no way to find out whether a PyTorch codebase will run on ROCm short
of provisioning an AMD GPU and finding out. AMD ships HIPIFY, but HIPIFY
translates C++/CUDA source; it says nothing about a Python project's
dependencies, its build flags, its container, or its numeric assumptions.

The question "can I move this to AMD at all?" is asked far more often than it
is answered, and it is asked by people who do not yet have an AMD GPU to answer
it with. That is the gap.

## What this is

A static analyser that scans a repository and reports what will not survive a
move from CUDA to ROCm, graded by how confidently we can claim it.

It runs on any machine. It needs no GPU of any vendor.

## Non-goals

- Not a translator. It does not rewrite code; HIPIFY already does that for C++.
- Not a performance predictor. It says nothing about whether the port is fast.
- Not a runtime tool. It never imports or executes the code it scans.
- Not a general Python linter. Only portability from CUDA to ROCm.

## The central constraint: claims must be earned

The output will be published, including findings about codebases the author
does not own. A false "this breaks on ROCm" is worse than silence: it is
checkable, and being wrong in public costs more than the finding was worth.

So findings are tiered, and the top tier is gated on evidence.

| Tier | Means | Requirement |
|---|---|---|
| `BLOCKER` | This does not work on ROCm | **Must** name a proof |
| `REVIEW` | Known-risky, a human must judge | No proof required |
| `INFO` | Portable, but worth knowing | Often documents a non-problem |

A `proof` is one of exactly two things:

- `doc:<url>` — a citation to vendor documentation stating the limitation.
- `selftest:<fn>` — the name of a function in `proofs.py` that **demonstrates
  the failure on the machine running the test suite**, with no GPU required.

A proof establishes that *the platform behaves this way*. It does **not**
establish that a given codebase hits that behaviour. Tier is therefore decided
by a separate question: **does matching this pattern mean the breakage is
certain?** If a pattern is merely correlated with the risk, the rule is
`REVIEW` even when its proof is airtight. Rules of any tier may carry a proof;
only `BLOCKER` requires one.

### The enforcement, which is the point

Two tests in CI:

1. Every rule with tier `BLOCKER` names a non-empty `proof`. A rule without one
   cannot be a blocker; the test fails the build.
2. Every `selftest:` proof is **executed**, and must still demonstrate the
   failure it claims. A proof that stops reproducing fails the build, forcing
   the rule to be demoted or deleted.

This is the differentiator. Pattern matching is commodity. A scanner in which
every blocking claim is backed by a check the reader can run themselves is not.

### The false-positive guard

`tests/fixtures/portable/` holds code that looks unportable and is not. A test
asserts **zero BLOCKERs** fire on that directory.

The motivating example: `init_process_group("nccl")`. PyTorch on ROCm keeps the
literal string `"nccl"` and routes it to RCCL. A naive scanner flags it as a
blocker and is wrong. Here it is an `INFO` rule whose body explains why it is
fine — so the tool teaches rather than just alarms.

## Architecture

```
rocm_portscan/
  rules.py        rule definitions (pure data, no logic)
  proofs.py       runnable proofs backing BLOCKER rules
  collectors.py   AST walker (.py) + text scanner (all other files)
  scan.py         orchestration, file walking, ignore handling
  report.py       terminal / markdown / json renderers
  __main__.py     CLI entry point
```

Hybrid analysis, split by artifact type, because portability blockers live in
both halves:

- **AST** (`ast` stdlib) for Python: imports, attribute chains, call sites,
  string literals in known-significant positions. Precise; never fires on a
  comment or docstring.
- **Text** for everything else: `requirements*.txt`, `pyproject.toml`,
  `setup.py` build flags, `Dockerfile*`, `*.cu`, `*.cuh`, shell scripts. These
  carry real blockers (`apex` pinned in requirements, `-gencode
  arch=compute_90` in setup.py, `FROM nvidia/cuda`) that no AST will ever see.

Data flow: `scan.py` walks the tree, dispatches each file to the right
collector, collectors emit `Finding(rule, path, line, snippet)`, `report.py`
renders. Rules are data; collectors hold the matching logic; renderers hold
no logic about severity.

## Initial ruleset

**BLOCKER**

| id | what | proof |
|---|---|---|
| ROCM001 | `apex` dependency | doc |
| ROCM003 | `.cu`/`.cuh` sources or `CUDAExtension` | doc |
| ROCM004 | `-gencode arch=compute_XX` / `sm_XX` nvcc flags | doc |
| ROCM007 | `pynvml` / `nvidia-ml-py` | doc |
| ROCM008 | `cupy-cuda*` wheel pin | doc |

Each is a hard dependency or a toolchain flag with no ROCm equivalent: matching
the pattern means the build or import fails, full stop.

**REVIEW**

| id | what | proof |
|---|---|---|
| ROCM002 | `transformer_engine` dependency | — |
| ROCM005 | hardcoded `float8_e4m3fn` / `float8_e5m2` | selftest |
| ROCM006 | `-inf` mask fill co-located with fp8 dtypes | selftest |
| ROCM100 | `get_device_capability()` comparisons | — |
| ROCM101 | `flash_attn` | — |
| ROCM102 | `bitsandbytes` | — |
| ROCM103 | `FROM nvidia/cuda` | — |
| ROCM104 | `torch.cuda.nvtx` | — |
| ROCM105 | Triton kernels with CUDA-specific intrinsics | — |
| ROCM106 | `--gpus all` / nvidia-docker | — |
| ROCM107 | `xformers` | — |

Two of these were `BLOCKER` in the first draft and were demoted on review:

- **ROCM005/ROCM006 (fp8).** The proof is solid — `e4m3fnuz` caps at 240 rather
  than 448 and has no infinity, so an out-of-range weight and an `-inf` mask
  fill both become NaN, reproducible on CPU, and it is the same root cause that
  stops vLLM's fp8 path starting on gfx942. But *referencing* `float8_e4m3fn`
  is not itself a break: the dtype exists on ROCm. It breaks when fn-quantised
  weights meet fnuz hardware, which static analysis cannot confirm. Certain
  behaviour, uncertain applicability, therefore `REVIEW`.
- **ROCM006 detection is a heuristic.** Proving "an `-inf` fill reaches an fp8
  tensor" needs dataflow analysis this tool does not do. It fires on
  co-occurrence within a file, which is suggestive, not conclusive — another
  reason it cannot be a blocker.
- **ROCM002 (transformer_engine).** Demoted because the author has not
  confirmed the current state of ROCm support and will not assert a blocker
  from memory. Promote it only with a citation.

**INFO**

ROCM200 `"nccl"` backend string (portable, routes to RCCL) · ROCM201 general
`torch.cuda.*` usage (portable, ROCm reuses the namespace) · ROCM202 fp16 /
`.half()` (portable)

## Interface

```
python -m rocm_portscan <path> [--format terminal|markdown|json]
                               [--min-tier blocker|review|info]
                               [--exclude GLOB]
```

Exit codes: `0` no blockers, `1` blockers found, `2` bad invocation. The
non-zero exit on blockers is what lets someone drop it into their own CI.

`--format markdown` produces a report suitable for posting directly; this is
how the NeMo findings get written.

## Testing

TDD throughout. For each rule, written before the rule itself:

1. A fixture under `tests/fixtures/breaks/` that **must** trip it.
2. A counter-fixture under `tests/fixtures/portable/` that **must not**.

Plus the two meta-tests above (blockers have proofs; proofs still reproduce),
and renderer tests asserting JSON validity and markdown structure.

Existing CI gains a `portscan` job running the suite on 3.10 and 3.12.

## Success criteria

1. Runs clean against `tests/fixtures/portable/` with zero blockers.
2. Every BLOCKER rule's proof executes and reproduces in CI.
3. Produces a defensible findings report against a real large codebase
   (NVIDIA NeMo) that a NeMo maintainer would not dispute.
4. Someone who does not own an AMD GPU gets a useful answer in under a minute.

## Risks

- **False positives are the whole risk.** Mitigated by tiering, the portable
  fixture directory, and publishing only blockers.
- **Rule rot.** Mitigated by executable proofs failing CI when they stop
  reproducing.
- **Scanning NeMo is a large job.** If the run is unwieldy, scope the published
  findings to a subset and say which subset.
