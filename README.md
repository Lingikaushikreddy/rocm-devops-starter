# rocm-devops-starter

A small, honest starting point for running PyTorch workloads on AMD GPUs: a
diagnostic probe, a smoke-test training run, a ROCm container with the device
plumbing already correct, and CI that actually fails when something breaks.

The gap this fills is not "how do I train a model". It is the hour you lose
between provisioning an AMD instance and knowing whether your stack is sound.

## Status, stated plainly

| Part | Tested |
|---|---|
| `gpu_probe.py`, `smoke_train.py` on CPU | Yes, locally and in CI |
| Probe's no-torch and CPU-only-build paths | Yes |
| fp8 round-trip fidelity check | Yes - reproduces the `e4m3fnuz` 448 -> NaN case |
| `make_report.py` refusing to invent results | Yes, asserted in CI |
| Smoke test's non-zero exit on a non-learning stack | Yes |
| ROCm / HIP device paths | **Not yet** - needs an AMD GPU |
| `docker/Dockerfile.rocm` build | **Not yet** - needs a ROCm host |

The ROCm-specific paths are written against documented behaviour but have not
been run on real hardware. `docs/RUNBOOK.md` is the checklist for validating
them on AMD Developer Cloud, and this table gets updated when they pass.

## Quick start

```bash
python scripts/gpu_probe.py          # what am I actually running on?
python scripts/smoke_train.py        # can this box train anything at all?
```

No arguments, no config, no GPU required. On a machine without PyTorch the
probe tells you which wheel index to use instead of raising an ImportError.

In a container, once you are on a ROCm host:

```bash
make build
make up
```

## What each piece is for

**`scripts/gpu_probe.py`** - reports the driver, the PyTorch build type, every
visible device, and which dtypes actually execute. It distinguishes a ROCm
build from a CPU-only build, which is the single most common cause of "my GPU
isn't detected": `pip install torch` gives you a CPU wheel that will never see
an AMD card. Dtype support is measured by running a real matmul, not by reading
a capability flag, because dtypes routinely construct fine and then fail in the
kernel.

**`scripts/smoke_train.py`** - a ten-second training run whose only job is to
prove forward, backward and an optimiser step all work and a loss goes down.
Labels come from a fixed random projection, so the loss *must* fall on a
working stack. Exits non-zero if it does not, which makes it usable as a CI
gate and as a post-provision check.

**`scripts/make_report.py`** - turns a real probe run into a forum post. Its
useful property is what it refuses to do: it will not emit a hardware report
unless a GPU was actually bound on the machine it ran on, so there is no path
from "haven't run it yet" to a post full of numbers. `--template` gives you the
skeleton, clearly marked as unfilled.

**`docker/`** - a ROCm PyTorch image plus the compose settings people miss:
`/dev/kfd` and `/dev/dri`, the `video` and `render` groups, `ipc: host`, and an
`shm_size` large enough for real dataloaders. Without these the container
starts happily and reports zero devices.

**`.github/workflows/ci.yml`** - lints, runs both scripts on CPU across two
Python versions, validates the compose file, and asserts that the smoke test
*fails* when handed a run that cannot learn. A test suite that cannot fail is
not a test suite.

## Why torch is not in requirements.txt

On AMD, PyTorch must come from the ROCm wheel index, and the ROCm base image
already ships a matching build. Putting `torch` in `requirements.txt` would
overwrite that with a CPU-only wheel from PyPI. That is the fastest way to
break a working ROCm image, so the file carries the explanation instead.

## Layout

```
scripts/gpu_probe.py      diagnostic, safe to run anywhere
scripts/smoke_train.py    tiny training run, non-zero exit on failure
docker/Dockerfile.rocm    ROCm PyTorch image, arch-pinnable
docker/docker-compose.yml device plumbing for ROCm
scripts/make_report.py    forum post from a real run; refuses to fake one
docs/RUNBOOK.md           validating on AMD Developer Cloud
docs/AMD-COMMUNITY-MAP.md where the AMD dev community has gaps
```

## Credits

Two checks here exist because of a measured writeup by William_Mclean on the
AMD developer forum: that `/dev/kfd` can be present on a card whose driver
never bound (so it is not a liveness signal - `rocminfo` resolving a gfx target
is), and that fp8 on gfx942 is `e4m3fnuz`, under which 448 round-trips to NaN.
The second is reproduced in this repo's own test output.

https://devcommunity.amd.com/t/which-numeric-formats-an-mi300x-actually-executes-measured-on-a-developer-cloud-droplet/1043

## License

MIT
