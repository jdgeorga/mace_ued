#!/usr/bin/env bash
# MoSe2/WSe2 bilayer linewidth and lifetime pipeline. Run inside a GPU allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_PHONON_ROOT="${MLIP_PHONON_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
MODELS="${MODELS:-$MLIP_PHONON_ROOT/models}"
export MLIP_PHONON_ROOT MODELS
MACE_PHONON_QUIET=1 source "$MLIP_PHONON_ROOT/load_mace_phonon_env.sh"

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
STUB="$MLIP_PHONON_ROOT/repos/mlip_phonon_scattering/ittnotify_stub/libittnotify.so"
[ -f "$STUB" ] && export LD_PRELOAD="$STUB${LD_PRELOAD:+:$LD_PRELOAD}"

MPI_RANKS="${MPI_RANKS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/run_lifetime}"
PREFIX="MoSe2_WSe2_relaxed"
INPUT="$SCRIPT_DIR/MoSe2_WSe2_bilayer.xyz"
MO="$MODELS/MoSe2.model"
W="$MODELS/WSe2.model"
INTER="$MODELS/MoSe2_WSe2.model"
LAYER="[['Mo','Se','Se'],['W','Se','Se']]"
INTERLAYER_ARGS=(--interlayer --intralayer-models "$MO" "$W" --interlayer-model "$INTER" --layer-symbols "$LAYER")

REL="$OUTPUT_DIR/$PREFIX.xyz"
P2="$OUTPUT_DIR/phonopy_disp.yaml"
P3="$OUTPUT_DIR/phono3py_disp.yaml"
F3="$OUTPUT_DIR/${PREFIX}_forces_3rd.npy"
F23="$OUTPUT_DIR/${PREFIX}_forces_2nd_from_3rd.npy"
FC_CACHE="$OUTPUT_DIR/phonon_cache_m36x36x1"
GAMMA_CACHE="$OUTPUT_DIR/phono3py_cache_36x36x1_T50.0K"
W_OUT="$OUTPUT_DIR/W_scattering_mesh_m36x36x1_T50_GPU.h5"
GAMMA_NPZ="$OUTPUT_DIR/gamma_W_full_T50.npz"

multi_rank() { srun --overlap -n "$MPI_RANKS" --gpus-per-task=1 "$@"; }
single_rank() { srun --overlap -N1 -n1 --gpus-per-node=4 "$@"; }

mkdir -p "$OUTPUT_DIR"
# The GPU scatter engine derives its FC/gamma cache dirs (phonon_cache_*, phono3py_cache_*)
# RELATIVE to the current working directory and os.chdir()s into the gamma cache on every
# rank. Run from $OUTPUT_DIR so those relative caches coincide with the absolute paths steps
# 4/6 use, and pre-create the gamma cache so a rank on another node never races the rank-0
# mkdir (see the CWD==OUTPUT_DIR note in ../../../../SPLIT_MLIP_NOTES.md).
mkdir -p "$GAMMA_CACHE"
cd "$OUTPUT_DIR"
echo "=== MoSe2/WSe2 bilayer: MPI_RANKS=$MPI_RANKS, output=$OUTPUT_DIR (cwd=$PWD) ==="
echo "=== [1] relax ==="
single_rank mlip-linewidth-relax "$INPUT" "$OUTPUT_DIR/$PREFIX" "${INTERLAYER_ARGS[@]}" --fmax 1.3e-5 --steps 1000 --maxstep 0.05 --device cuda
echo "=== [2a] phonopy displacements (6 6 1) ==="
mlip-linewidth-phonopy-yaml "$REL" --supercell 6 6 1 --output "$P2"
echo "=== [2b] phono3py displacements (fc3 3 3 1; phonon 6 6 1) ==="
mlip-linewidth-phono3py-yaml "$REL" --supercell 3 3 1 --phonon-supercell 6 6 1 --output "$P3"
echo "=== [3a] second-order forces ==="
multi_rank mlip-linewidth-forces2 "$REL" "$P2" "${INTERLAYER_ARGS[@]}" --output "$OUTPUT_DIR/${PREFIX}_forces_2nd.npy" --device cuda
echo "=== [3b] third-order forces ==="
multi_rank mlip-linewidth-forces3 "$REL" "$P3" "${INTERLAYER_ARGS[@]}" --supercell-matrix 3 3 1 --output "$F3" --device cuda
echo "=== [3c] second-order forces from phono3py displacements ==="
multi_rank mlip-linewidth-forces2-from3 "$REL" "$P3" "${INTERLAYER_ARGS[@]}" --supercell-matrix 6 6 1 --output "$F23" --device cuda
echo "=== [4] force-constant cache ==="
mlip-linewidth-cache-fc --phono3py-yaml "$P3" --fc2-forces "$F23" --fc3-forces "$F3" --cache-dir "$FC_CACHE" --populate-mesh-cache
echo "=== [5] GPU scattering (mesh 36 36 1) ==="
multi_rank mlip-linewidth-scatter-gpu --phono3py-yaml "$P3" --fc2-forces "$F23" --fc3-forces "$F3" --mesh 36 36 1 --temperature 50 --batch-size 1 --fallback-lang NONE --max-w-gb 500 --force-recompute-gamma-detail --output "$W_OUT"
echo "=== [6] extract gamma ==="
mlip-linewidth-extract-gamma --w-h5 "$W_OUT" --cache-dir "$GAMMA_CACHE" --yaml "$P3" --mesh 36 36 1 --temperature 50 --out "$GAMMA_NPZ"
echo "=== [7] plot linewidth and lifetime ==="
mlip-linewidth-plot --name mose2_wse2_aligned --label "MoSe\$_2\$/WSe\$_2\$ aligned bilayer" --yaml "$P3" --fc2 "$FC_CACHE/fc2.npy" --gamma-w "$GAMMA_NPZ" --mesh 36 36 1 --out-dir "$OUTPUT_DIR/figures" || echo "PLOT_WARN"
echo "=== [8] validate physics ==="
mlip-linewidth-validate --run-dir "$OUTPUT_DIR" --expected-atoms 6 --expected-bands 18 --mesh 36 36 1 --temperature 50 --figure-stem mose2_wse2_aligned --shear-range 0.50 0.65 --breathing-range 0.75 0.95
