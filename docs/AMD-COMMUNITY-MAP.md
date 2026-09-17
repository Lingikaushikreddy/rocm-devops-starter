# AMD developer community: where the gaps are

Surveyed 2026-09-17 against devcommunity.amd.com, by reading the forum's own
category and topic listings rather than the front page. Topic counts are what
the site reported on that date and will drift.

## The finding

The forum has a detailed category tree and very little in most of it. That is
an opportunity, not a criticism: a good post in an empty category is the only
post in that category.

Categories with **zero** topics, ranked by how cheaply they can be filled by
someone with container and CI experience:

| Topics | Category |
|---|---|
| 0 | ROCm Developers > AI/HPC Infrastructure > **Containers** |
| 0 | ROCm Developers > AI/HPC Infrastructure |
| 0 | ROCm Developers > Getting Started > **Build from source** |
| 0 | ROCm Developers > Getting Started > **Contribution** |
| 0 | Workloads > **Cloud Computing** |
| 0 | Workloads > Database & Analytics |
| 0 | Workloads > HCI & Virtualization |
| 0 | Workloads > Networking Solutions |

Nearly empty, and directly adjacent:

| Topics | Category |
|---|---|
| 1 | Workloads > **CPU to GPU Porting** |
| 1 | ROCm Developers > AI Development > Computer Vision |
| 2 | ROCm Developers > AI/HPC Infrastructure > Cluster Deployment |
| 2 | ROCm Developers > AI/HPC Infrastructure > Distributed Training & Inference |
| 2 | ROCm Developers > AI/HPC Infrastructure > Tools |

For contrast, the busiest technical area is
`ROCm Developers > Getting Started > Tutorials` at 11 topics. This is a small
forum throughout.

Also worth knowing: `AI > AI Developer Program > Notebook Submission` exists as
a submission route and contains one topic, which on inspection is a support
question rather than a notebook. It is effectively unused.

## What the program actually gives you

From the program's own "10 reasons" post:

- $100 in cloud credits on signup
- no hardware needed - AMD Developer Cloud access from day one
- a private Discord for technical discussion
- monthly "Ask the Expert" sessions with open-source partners
- early invites to dev events and workshops
- projects can be featured on official AMD developer channels
- automatic entry to monthly hardware raffles
- a free month of DeepLearning.AI Pro

## Learning resources on the platform

AI Academy carries a **ROCm Certified Associate** track plus courses on HIP
kernel programming, AI agents with MCP, multi-agent systems, and fine-tuning
with Unsloth. Caveat from reading the category: a large share of its topics are
people reporting that lab completion and points are not tracking. Budget for
LMS friction and screenshot your progress.

Key links:

- Community (Discourse): https://devcommunity.amd.com/
- Community hub, both platforms: https://developer.amd.com/playbooks/community/
- AI Developer Program: https://developer.amd.com/ai-developer-program
- Program announcement: https://www.amd.com/en/developer/resources/technical-articles/2025/amd-ai-developer-program.html
- Developer Discord: https://discord.com/invite/amd-dev

## Project ideas, ranked by gap x fit

Ranked for someone with container, CI and ML-infra experience coming from CUDA.

**1. ROCm container + CI starter** - this repo. Hits the empty Containers
category directly, and the device plumbing is the part people get wrong.
Lowest effort, highest coverage of an empty space.

**2. CUDA-to-ROCm port diary** - take something real you already run on NVIDIA
and port it, recording every break. `CPU to GPU Porting` has one topic. The
value is entirely in the failures, which is also what makes it cheap: you do
not need it to succeed to have something worth publishing.

**3. Measured dtype support table for MI300X** - what the hardware advertises
versus what a kernel will execute, including where autograd diverges from a
forward pass. `gpu_probe.py` already emits most of this; running it on real
hardware turns it into a post.

**4. Multi-GPU / distributed smoke harness** - extend the smoke test to
torchrun across devices. `Distributed Training & Inference` has two topics and
`Cluster Deployment` has two.

**5. Notebook submission** - the route exists and is unused. Any competent
notebook is the first real one there.

## Honest caveat

Everything above is a snapshot of one afternoon's reading. Topic counts move,
and an empty category may be empty because it is new rather than because it is
wanted. Worth one question in the forum before investing heavily in any single
one of these.
