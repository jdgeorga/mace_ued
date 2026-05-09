#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/pscratch/sd/j/jdgeorga/twist-anything/phonon_unfolding/scratch/phonon_diff/phonon_2.6_env/bin/python}"
INPUT_FILE="${1:-${INPUT_FILE:-}}"
OUTPUT_PREFIX="${2:-${OUTPUT_PREFIX:-${PWD}/mlip_relaxed}}"

if [[ -z "${INPUT_FILE}" ]]; then
  echo "Usage: $0 INPUT_STRUCTURE [OUTPUT_PREFIX]"
  echo "Or set INPUT_FILE=/path/to/structure and optional OUTPUT_PREFIX=/path/to/output_prefix."
  exit 1
fi

SUPER_X="${SUPER_X:-6}"
SUPER_Y="${SUPER_Y:-6}"
SUPER_Z="${SUPER_Z:-1}"
MESH_X="${MESH_X:-36}"
MESH_Y="${MESH_Y:-36}"
MESH_Z="${MESH_Z:-1}"
QPOINTS_FILE="${QPOINTS_FILE:-}"
BAND_PATH="${BAND_PATH:-}"
FMAX="${FMAX:-1e-5}"
STEPS="${STEPS:-500}"
MAXSTEP="${MAXSTEP:-0.05}"
MACE_MODEL="${MACE_MODEL:-medium}"
MACE_MODEL_PATH="${MACE_MODEL_PATH:-}"
DEVICE="${DEVICE:-cpu}"
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float32}"
MPI_RANKS="${MPI_RANKS:-1}"
UED_TEMPERATURE_K="${UED_TEMPERATURE_K:-100}"
UED_PHWMIN_MEV="${UED_PHWMIN_MEV:-0.2}"
UED_GMAX="${UED_GMAX:-3}"
UED_QZ="${UED_QZ:-0}"
UED_QZ_TOL="${UED_QZ_TOL:-1e-8}"
UED_ELECTRON_SCATTERING_MODEL="${UED_ELECTRON_SCATTERING_MODEL:-${UED_FORM_FACTOR_MODEL:-peng}}"
UED_EIGENVECTOR_GAUGE="${UED_EIGENVECTOR_GAUGE:-phonopy_to_phx}"
UED_TEMPERATURE_SWEEP_K="${UED_TEMPERATURE_SWEEP_K:-$(seq -s ' ' 0 50 1500)}"
UED_TEMPERATURE_TARGET_G="${UED_TEMPERATURE_TARGET_G:-1 0 0;1 1 0}"
UED_WRITE_TILED_CSV="${UED_WRITE_TILED_CSV:-1}"

RELAXED_XYZ="${OUTPUT_PREFIX}.xyz"
PHONOPY_YAML="${OUTPUT_PREFIX}_phonon_displacements.yaml"
FORCES_NPY="${OUTPUT_PREFIX}_forces.npy"
ENERGIES_NPY="${OUTPUT_PREFIX}_energies.npy"
OUTPUT_DIR="$(dirname "${OUTPUT_PREFIX}")"

MACE_ARGS=(--mace-model "${MACE_MODEL}" --device "${DEVICE}" --default-dtype "${DEFAULT_DTYPE}")
if [[ -n "${MACE_MODEL_PATH}" ]]; then
  MACE_ARGS+=(--mace-model-path "${MACE_MODEL_PATH}")
fi
if [[ "${DISPERSION:-0}" == "1" ]]; then
  MACE_ARGS+=(--dispersion)
fi
BAND_ARGS=()
if [[ -n "${BAND_PATH}" ]]; then
  BAND_ARGS+=(--band-path "${BAND_PATH}")
fi
QPOINTS_ARGS=()
if [[ -n "${QPOINTS_FILE}" ]]; then
  QPOINTS_ARGS+=(--qpoints-file "${QPOINTS_FILE}")
fi
UED_TEMPERATURE_SWEEP_ARGS=()
if [[ -n "${UED_TEMPERATURE_SWEEP_K}" ]]; then
  # shellcheck disable=SC2206
  UED_TEMPERATURE_SWEEP_VALUES=(${UED_TEMPERATURE_SWEEP_K})
  UED_TEMPERATURE_SWEEP_ARGS=(--temperature-sweep-k "${UED_TEMPERATURE_SWEEP_VALUES[@]}")
fi
UED_TEMPERATURE_TARGET_G_ARGS=()
if [[ -n "${UED_TEMPERATURE_TARGET_G}" ]]; then
  IFS=';' read -r -a UED_TEMPERATURE_TARGET_G_VALUES <<< "${UED_TEMPERATURE_TARGET_G}"
  for target_g in "${UED_TEMPERATURE_TARGET_G_VALUES[@]}"; do
    # shellcheck disable=SC2206
    target_g_components=(${target_g})
    if [[ "${#target_g_components[@]}" -ne 3 ]]; then
      echo "UED_TEMPERATURE_TARGET_G entries must be three numbers separated by spaces; entries are separated by semicolons." >&2
      exit 1
    fi
    UED_TEMPERATURE_TARGET_G_ARGS+=(--temperature-target-g "${target_g_components[@]}")
  done
fi
UED_TILED_CSV_ARGS=()
if [[ "${UED_WRITE_TILED_CSV}" == "0" ]]; then
  UED_TILED_CSV_ARGS=(--no-tiled-csv)
fi

run_mpi_python() {
  if [[ "${MPI_RANKS}" -gt 1 && "$(command -v srun || true)" != "" ]]; then
    srun -n "${MPI_RANKS}" "${PYTHON}" "$@"
  else
    "${PYTHON}" "$@"
  fi
}

echo "[mlip_phonon_scattering] Relaxing structure"
"${PYTHON}" "${SCRIPT_DIR}/scripts/relax_structure.py" \
  "${INPUT_FILE}" "${OUTPUT_PREFIX}" \
  --fmax "${FMAX}" --steps "${STEPS}" --maxstep "${MAXSTEP}" \
  "${MACE_ARGS[@]}"

echo "[mlip_phonon_scattering] Generating phonopy displacements"
"${PYTHON}" "${SCRIPT_DIR}/scripts/generate_displacements.py" \
  "${RELAXED_XYZ}" \
  --supercell "${SUPER_X}" "${SUPER_Y}" "${SUPER_Z}" \
  --output "${PHONOPY_YAML}"

echo "[mlip_phonon_scattering] Computing MACE forces"
run_mpi_python "${SCRIPT_DIR}/scripts/compute_mace_forces.py" \
  "${RELAXED_XYZ}" "${PHONOPY_YAML}" \
  --forces-output "${FORCES_NPY}" \
  --energies-output "${ENERGIES_NPY}" \
  "${MACE_ARGS[@]}"

echo "[mlip_phonon_scattering] Solving phonons"
"${PYTHON}" "${SCRIPT_DIR}/scripts/solve_phonons.py" \
  --input-xyz "${RELAXED_XYZ}" \
  --forces "${FORCES_NPY}" \
  --phonopy-yaml "${PHONOPY_YAML}" \
  --mesh "${MESH_X}" "${MESH_Y}" "${MESH_Z}" \
  "${QPOINTS_ARGS[@]}" \
  "${BAND_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"

echo "[mlip_phonon_scattering] Extracting Qz=0 UED intensities"
"${PYTHON}" "${SCRIPT_DIR}/scripts/extract_uq_intensities.py" \
  --input-h5 "${OUTPUT_DIR}/eigenvector.h5" \
  --output-dir "${OUTPUT_DIR}/ued_intensity" \
  --temperature-k "${UED_TEMPERATURE_K}" \
  --phwmin-mev "${UED_PHWMIN_MEV}" \
  --gmax "${UED_GMAX}" \
  --qz "${UED_QZ}" \
  --qz-tol "${UED_QZ_TOL}" \
  --eigenvector-gauge "${UED_EIGENVECTOR_GAUGE}" \
  --electron-scattering-model "${UED_ELECTRON_SCATTERING_MODEL}" \
  "${UED_TEMPERATURE_SWEEP_ARGS[@]}" \
  "${UED_TEMPERATURE_TARGET_G_ARGS[@]}" \
  "${UED_TILED_CSV_ARGS[@]}"

echo "[mlip_phonon_scattering] Done"
