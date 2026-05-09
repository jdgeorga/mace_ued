#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "${EXAMPLE_DIR}/../.." && pwd)"

export BAND_PATH="${BAND_PATH:-GXMGRX,MR}"
export UED_GMAX="${UED_GMAX:-4}"
export UED_TEMPERATURE_SWEEP_K="${UED_TEMPERATURE_SWEEP_K:-$(seq -s ' ' 0 50 1500)}"
export UED_TEMPERATURE_TARGET_G="${UED_TEMPERATURE_TARGET_G:-2 2 0;3 1 0}"
export UED_WRITE_TILED_CSV="${UED_WRITE_TILED_CSV:-0}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${EXAMPLE_DIR}/run/Si_mace_relaxed}"

bash "${WORKFLOW_DIR}/run_mlip_phonons.sh" \
  "${INPUT_XYZ:-${EXAMPLE_DIR}/Si.xyz}" \
  "${OUTPUT_PREFIX}"
