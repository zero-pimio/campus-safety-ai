#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON311:-/opt/homebrew/bin/python3.11}"
venv_dir="$project_dir/.venv"
model_dir="$project_dir/models/ppTSM"
model_zip="$project_dir/runtime/downloads/ppTSM_fight.zip"
model_url="https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip"
model_sha256="a58a13c904b9b0fa9e101663826e6a003ecf5e61e3648dd51666c2d6702fbdfd"

if [[ ! -x "$python_bin" ]]; then
  echo "Python 3.11 not found at $python_bin" >&2
  echo "Install it with: brew install python@3.11" >&2
  exit 1
fi

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -e "$project_dir[fight-video,vision,dev]"

if [[ ! -f "$model_dir/model.pdmodel" || ! -f "$model_dir/model.pdiparams" ]]; then
  mkdir -p "$(dirname "$model_zip")" "$project_dir/models"
  curl -L --fail "$model_url" -o "$model_zip"
  actual_sha256="$(openssl dgst -sha256 "$model_zip" | awk '{print $NF}')"
  if [[ "$actual_sha256" != "$model_sha256" ]]; then
    echo "Model checksum mismatch: $actual_sha256" >&2
    exit 1
  fi
  unzip -q -o "$model_zip" -d "$project_dir/models"
fi

"$venv_dir/bin/python" -c 'import paddle; paddle.utils.run_check()'
echo "Video runtime is ready: $venv_dir/bin/python"
echo "Fight model is ready: $model_dir"
