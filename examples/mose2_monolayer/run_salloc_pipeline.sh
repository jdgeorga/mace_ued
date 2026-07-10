#!/usr/bin/env bash
# Entry point for an interactive GPU allocation; run from any working directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_PHONON_ROOT="${MLIP_PHONON_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
MODELS="${MODELS:-$MLIP_PHONON_ROOT/models}"
export MLIP_PHONON_ROOT MODELS
MACE_PHONON_QUIET=1 source "$MLIP_PHONON_ROOT/load_mace_phonon_env.sh"

export MPI_RANKS="${MPI_RANKS:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/lifetime_run}"
echo "=== allocation: nodes=${SLURM_NNODES:-?}, MPI_RANKS=$MPI_RANKS ==="
bash "$SCRIPT_DIR/run_linewidth.sh"
