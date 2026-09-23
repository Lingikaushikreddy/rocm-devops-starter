#!/usr/bin/env bash
# Reproduce docs/SURVEY-2026-09.md: fetch each repository at the exact commit
# that was scanned, then run rocm_portscan over it. Nothing here needs a GPU.
#
#   scripts/run_survey.sh [workdir]     # default workdir: ./results/survey-src
#
# JSON results land in results/survey/, which is gitignored.
set -euo pipefail

WORKDIR="${1:-results/survey-src}"
OUT="results/survey"
PY="${PYTHON:-python}"
mkdir -p "$WORKDIR" "$OUT"

while read -r repo sha; do
  [ -z "$repo" ] && continue
  dir="$WORKDIR/${repo//\//_}"
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "https://github.com/$repo.git"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q FETCH_HEAD
  # Exit code 1 means blockers were found, which is a result, not an error.
  "$PY" -m rocm_portscan "$dir" --format json > "$OUT/${repo//\//_}.json" || [ $? -eq 1 ]
  echo "scanned $repo@${sha:0:7}"
done <<'EOF'
NVIDIA-NeMo/Speech cf724ac337d1ebc7d0dda1e23fb80916f52927a5
unslothai/unsloth f8bf9a3a8456b88cbc3876f1d30f46ed759d4ecf
axolotl-ai-cloud/axolotl b11a2f7b0ad8bad5a7fe3473f651a915e4ae5817
huggingface/diffusers 0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec
microsoft/DeepSpeed d099bc61bb95bb71abaac3d3c1ba6911f15b2b95
huggingface/peft 116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa
Lightning-AI/litgpt 18c931c15c2398621d35371a12d2b1785e855bb6
pytorch/torchtune bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1
huggingface/pytorch-image-models 0424d45a8d3aec2929e2a4a9c91a6ccdf000c653
facebookresearch/segment-anything-2 2b90b9f5ceec907a1c18123530e92e794ad901a4
openai/whisper 86098128c0b4f24f0e2aa2994de830614b474227
karpathy/nanoGPT 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
EOF
