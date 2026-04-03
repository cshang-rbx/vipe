#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
WARMUP_VIDEO="${WARMUP_VIDEO:-${REPO_ROOT}/assets/examples/dog-example.mp4}"
WARMUP_OUTPUT="${WARMUP_OUTPUT:-${REPO_ROOT}/gradio_runs/setup_smoketest}"

log() {
  printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%d %H:%M:%S UTC')" "$*"
}

run_in_repo() {
  (
    cd "${REPO_ROOT}"
    "$@"
  )
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

log "Repository root: ${REPO_ROOT}"
log "Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

log "Upgrading pip tooling"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

log "Installing pinned ViPE dependencies"
"${VENV_PIP}" install -r "${REPO_ROOT}/envs/requirements.txt" --extra-index-url "${TORCH_INDEX_URL}"

log "Installing ViPE editable package"
run_in_repo env MAX_JOBS=8 "${VENV_PYTHON}" -m pip install --no-build-isolation -e .

log "Installing Gradio and restoring the Hugging Face Hub version expected by ViPE"
"${VENV_PIP}" install gradio
"${VENV_PIP}" install huggingface-hub==0.36.0

log "Detecting GPU compute capability for a targeted CUDA rebuild"
TORCH_CUDA_ARCH_LIST="$("${VENV_PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; ViPE requires a CUDA-capable GPU for this setup.")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"

log "Rebuilding ViPE CUDA extensions for compute capability ${TORCH_CUDA_ARCH_LIST}"
run_in_repo env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=8 \
  "${VENV_PYTHON}" -m pip install --no-build-isolation -e . --force-reinstall --no-deps

if [[ ! -f "${WARMUP_VIDEO}" ]]; then
  echo "Warm-up video not found: ${WARMUP_VIDEO}" >&2
  exit 1
fi

log "Running a one-GPU warm-up to download the default model stack"
run_in_repo env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${VENV_PYTHON}" run.py \
  pipeline=default \
  streams=raw_mp4_stream \
  streams.base_path="${WARMUP_VIDEO}" \
  streams.frame_end=12 \
  pipeline.output.path="${WARMUP_OUTPUT}" \
  pipeline.output.save_artifacts=false \
  pipeline.output.save_viz=false

log "Setup completed"
cat <<EOF

Launch the Gradio demo on GPU ${GPU_ID} with:

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} "${VENV_PYTHON}" scripts/gradio_demo.py --share --server-name 0.0.0.0 --server-port 7860

EOF
