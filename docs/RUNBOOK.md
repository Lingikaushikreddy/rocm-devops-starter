# Runbook: validating this on AMD Developer Cloud

Everything here is unverified until it has run on real AMD hardware. This is
the checklist for that, in order. Record what actually happened, including the
failures - the failures are the publishable part.

## 0. Before you start

The AI Developer Program gives $100 of cloud credits on signup. That is the
budget this runbook is written against, so prefer short sessions and shut
instances down rather than leaving them idle.

## 1. Provision and probe

```bash
git clone <this repo> && cd rocm-devops-starter
python scripts/gpu_probe.py
```

Record:

- [ ] `backend` reads `rocm`, not `cpu-only`
- [ ] `hip_version` is populated
- [ ] device count matches what you paid for
- [ ] device name and `gcn_arch` (MI300X should report `gfx942`)
- [ ] which dtypes report `ok` vs `unsupported`

The dtype table is the interesting output. Marketing material and what a kernel
will actually execute are different things, and the gap is worth writing down.

## 2. Smoke test on device

```bash
python scripts/smoke_train.py                      # auto-detects
python scripts/smoke_train.py --dtype bf16
python scripts/smoke_train.py --dtype fp16
```

Record throughput and peak memory for each. If any dtype fails here but the
probe reported it `ok`, that gap is a genuine finding - a dtype that survives a
matmul but dies in autograd is exactly the sort of thing nobody has documented.

## 3. Container path

```bash
make build
make up
```

- [ ] image builds (note the base tag you pinned)
- [ ] container sees the same device count as the host
- [ ] if it sees zero, work through: `/dev/kfd` present? `render` group? user in
      `video`? `PYTORCH_ROCM_ARCH` needed for this card?

Step 3 failing is the most likely outcome and the most useful one. The
`ROCm Developers > AI/HPC Infrastructure > Containers` category on the AMD
forum currently has zero topics, so a working writeup of this has no
competition.

## 4. Update the status table

Change the "Not yet" rows in `README.md` to reflect what passed, with the
hardware and ROCm version you tested against. Do not mark anything green that
you have not personally run.

## 5. Then write it up

Post the result. Suggested homes on devcommunity.amd.com:

- container findings -> `ROCm Developers > AI/HPC Infrastructure > Containers`
- dtype table -> `ROCm Developers > AI Development > Performance Tuning`
- the CUDA-to-ROCm angle -> `Workloads > CPU to GPU Porting`

Link the repo. Include the failures.
