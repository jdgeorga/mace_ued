#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "${EXAMPLE_DIR}/../.." && pwd)"

# Primitive FCC/diamond ASE special points omit cubic `M`; use an FCC-compatible path unless overridden.
export BAND_PATH="${BAND_PATH:-GXWGL,LGK}"
export UED_GMAX="${UED_GMAX:-3}"
export UED_TEMPERATURE_SWEEP_K="${UED_TEMPERATURE_SWEEP_K:-$(seq -s ' ' 0 50 1500)}"
# Primitive fractional (h,k,l) for diamond Si: (1,1,1) allowed; (1,1,0) systematically absent.
export UED_TEMPERATURE_TARGET_G="${UED_TEMPERATURE_TARGET_G:-1 1 1;1 1 0}"
export UED_WRITE_TILED_CSV="${UED_WRITE_TILED_CSV:-0}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${EXAMPLE_DIR}/run/Si_mace_relaxed}"

bash "${WORKFLOW_DIR}/run_mlip_phonons.sh" \
  "${INPUT_XYZ:-${EXAMPLE_DIR}/Si.xyz}" \
  "${OUTPUT_PREFIX}"
