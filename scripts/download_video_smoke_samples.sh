#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sample_dir="$project_dir/datasets/samples"
mkdir -p "$sample_dir"

curl -L --fail \
  https://raw.githubusercontent.com/seymanurakti/fight-detection-surv-dataset/master/fight/fi001.mp4 \
  -o "$sample_dir/fight-fi001.mp4"
curl -L --fail \
  https://raw.githubusercontent.com/seymanurakti/fight-detection-surv-dataset/master/noFight/nofi001.mp4 \
  -o "$sample_dir/nofight-nofi001.mp4"

echo "Smoke-test videos are ready in $sample_dir"
