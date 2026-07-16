#!/usr/bin/env bash
# DFPT-harmonic + split-MLIP linewidth factorial. Run inside an active GPU allocation.
set -euo pipefail

export MACE_PHONON_QUIET=1
export OMP_NUM_THREADS=1  # matdyn 2D-LOTO rgd_blk race fix

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_ROOT="${MLIP_PHONON_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
MODELS="${MODELS:-$WORK_ROOT/models}"
DFPT_DIR="${DFPT_DIR:-/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix}"
MESH="${MESH:-36 36 1}"
TEMP="${TEMP:-50}"
MESH_TAG="${MESH// /x}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/runs/mose2_wse2_bilayer_dfpt_${MESH_TAG}_T${TEMP}}"
MPI_RANKS="${MPI_RANKS:-4}"
ALLOC_JOBID="${ALLOC_JOBID:?Set ALLOC_JOBID to the active GPU allocation job ID.}"

V1_DIR="${V1_DIR:-$WORK_ROOT/runs/mose2_wse2_bilayer_36x36x1_T50}"
V1_GAMMA_NPZ="${V1_GAMMA_NPZ:-$V1_DIR/gamma_W_full_T50.npz}"
V1_YAML="${V1_YAML:-$V1_DIR/phono3py_disp.yaml}"
V1_FC2="${V1_FC2:-$V1_DIR/phonon_cache_m36x36x1/fc2.npy}"

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
STUB="$REPO_ROOT/ittnotify_stub/libittnotify.so"
[ -f "$STUB" ] && export LD_PRELOAD="$STUB${LD_PRELOAD:+:$LD_PRELOAD}"
source "$WORK_ROOT/load_mace_phonon_env.sh"
module load cray-fftw cray-hdf5-parallel

MO="$MODELS/MoSe2.model"
W="$MODELS/WSe2.model"
INTER="$MODELS/MoSe2_WSe2.model"
LAYER="[['Mo','Se','Se'],['W','Se','Se']]"
INTERLAYER_ARGS=(--interlayer --intralayer-models "$MO" "$W" --interlayer-model "$INTER" --layer-symbols "$LAYER")

read -r -a MESH_ARGS <<< "$MESH"
if [ "${#MESH_ARGS[@]}" -ne 3 ]; then
    echo "MESH must contain exactly three integers, got: $MESH" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR/bin"
MATDYN_BIN="${MATDYN_BIN:-$(
    if [ -x /pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe-speedup/bin/matdyn.x ]; then
        printf '%s' /pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe-speedup/bin/matdyn.x
    else
        command -v matdyn.x
    fi
)}"
MATDYN_SHIM="$OUTPUT_DIR/bin/matdyn.x"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    '# srun shim for the validated speedup matdyn.x; OMP=1 avoids the 2D-LOTO race.' \
    'export OMP_NUM_THREADS=1' \
    "exec srun --jobid=$ALLOC_JOBID --overlap -N1 -n8 -c2 --cpu-bind=cores $MATDYN_BIN \"\$@\"" \
    > "$MATDYN_SHIM"
chmod +x "$MATDYN_SHIM"
export PATH="$OUTPUT_DIR/bin:$PATH"

STRUCT="$OUTPUT_DIR/dfpt_structure.xyz"
FC2_DFPT="$OUTPUT_DIR/dfpt_fc2.npy"
P3="$OUTPUT_DIR/phono3py_disp.yaml"
MODES_LOTO="$OUTPUT_DIR/dfpt_modes_loto.npz"
MODES_NOLOTO="$OUTPUT_DIR/dfpt_modes_noloto.npz"
F3_RES="$OUTPUT_DIR/f3_res.npy"
F2_RES="$OUTPUT_DIR/f2_res.npy"
F3_NORES="$OUTPUT_DIR/disp_forces_3rd_nores.npy"
F2_NORES="$OUTPUT_DIR/disp_forces_2nd_from3_nores.npy"
FC_V2_RES="$OUTPUT_DIR/fc_V2_res"
FC_V2_NORES="$OUTPUT_DIR/fc_V2_nores"
# V3 deliberately has its own MLIP-built FC2 cache: it has no injected modes.
FC_V3_RES="$OUTPUT_DIR/fc_V3_res"
FC_V3_NORES="$OUTPUT_DIR/fc_V3_nores"

multi_rank() { srun --jobid="$ALLOC_JOBID" --overlap -n "$MPI_RANKS" --gpus-per-task=1 "$@"; }
single_rank() { srun --jobid="$ALLOC_JOBID" --overlap -N1 -n1 --gpus-per-node=4 "$@"; }

echo "=== DFPT bilayer linewidth: mesh=$MESH temp=$TEMP MPI_RANKS=$MPI_RANKS output=$OUTPUT_DIR ==="
echo "=== [1] read DFPT structure, raw FC2, and 2D NAC ==="
if [ -f "$STRUCT" ] && [ -f "$FC2_DFPT" ] && [ -f "$OUTPUT_DIR/nac_2d.npz" ]; then
    echo "skip (exists)"
else
    mlip-linewidth-read-dfpt --dfpt-dir "$DFPT_DIR" --layer-symbols "$LAYER" --out-dir "$OUTPUT_DIR"
fi

echo "=== [2] phono3py displacements: FC3 3x3x1, phonon 6x6x1 ==="
if [ -f "$P3" ]; then
    echo "skip (exists)"
else
    mlip-linewidth-phono3py-yaml "$STRUCT" --supercell 3 3 1 --phonon-supercell 6 6 1 --output "$P3"
fi

# matdyn-modes needs phono3py's exact BZ grid, so the YAML is prepared first.
echo "=== [3a] matdyn modes: 2D-LOTO on, fixed gauge -1, floor 0.02 THz ==="
if [ -f "$MODES_LOTO" ]; then
    echo "skip (exists)"
else
    mlip-linewidth-matdyn-modes --phono3py-yaml "$P3" --ifc-xml "$DFPT_DIR/collect/ifc.q2r.xml" \
        --loto-2d on --dfpt-fc2 "$FC2_DFPT" --dfpt-structure "$STRUCT" --mesh "${MESH_ARGS[@]}" \
        --workdir "$OUTPUT_DIR/matdyn_loto" --floor-freq-thz 0.02 --gauge-sign -1 --out "$MODES_LOTO"
fi

echo "=== [3b] matdyn modes: 2D-LOTO off, fixed gauge -1, floor 0.02 THz ==="
if [ -f "$MODES_NOLOTO" ]; then
    echo "skip (exists)"
else
    mlip-linewidth-matdyn-modes --phono3py-yaml "$P3" --ifc-xml "$DFPT_DIR/collect/ifc.q2r.xml" \
        --loto-2d off --dfpt-fc2 "$FC2_DFPT" --dfpt-structure "$STRUCT" --mesh "${MESH_ARGS[@]}" \
        --workdir "$OUTPUT_DIR/matdyn_noloto" --floor-freq-thz 0.02 --gauge-sign -1 --out "$MODES_NOLOTO"
fi

if [ ! -f "$P3" ]; then
    echo "phono3py displacement YAML was not created: $P3" >&2
    exit 3
fi

echo "=== [4a] FC3 forces with reference-force subtraction ==="
if [ -f "$F3_RES" ]; then echo "skip (exists)"; else
    multi_rank mlip-linewidth-forces3 "$STRUCT" "$P3" "${INTERLAYER_ARGS[@]}" --supercell-matrix 3 3 1 --output "$F3_RES" --subtract-reference-forces --device cuda
fi
echo "=== [4b] FC3 forces without reference-force subtraction ==="
if [ -f "$F3_NORES" ]; then echo "skip (exists)"; else
    multi_rank mlip-linewidth-forces3 "$STRUCT" "$P3" "${INTERLAYER_ARGS[@]}" --supercell-matrix 3 3 1 --output "$F3_NORES" --device cuda
fi
echo "=== [4c] FC2-from-FC3 forces with reference-force subtraction ==="
if [ -f "$F2_RES" ]; then echo "skip (exists)"; else
    multi_rank mlip-linewidth-forces2-from3 "$STRUCT" "$P3" "${INTERLAYER_ARGS[@]}" --supercell-matrix 6 6 1 --output "$F2_RES" --subtract-reference-forces --device cuda
fi
echo "=== [4d] FC2-from-FC3 forces without reference-force subtraction ==="
if [ -f "$F2_NORES" ]; then echo "skip (exists)"; else
    multi_rank mlip-linewidth-forces2-from3 "$STRUCT" "$P3" "${INTERLAYER_ARGS[@]}" --supercell-matrix 6 6 1 --output "$F2_NORES" --device cuda
fi

echo "=== [5a] cache DFPT FC2 for V2 residual-on ==="
if [ -f "$FC_V2_RES/fc2.npy" ] && [ -f "$FC_V2_RES/fc3.npy" ]; then echo "skip (exists)"; else
    mlip-linewidth-cache-fc --phono3py-yaml "$P3" --fc2-forces "$F2_RES" --fc3-forces "$F3_RES" --cache-dir "$FC_V2_RES" --fc2-source dfpt --dfpt-fc2 "$FC2_DFPT" --mesh "${MESH_ARGS[@]}" --populate-mesh-cache
fi
echo "=== [5b] cache DFPT FC2 for V2 residual-off ==="
if [ -f "$FC_V2_NORES/fc2.npy" ] && [ -f "$FC_V2_NORES/fc3.npy" ]; then echo "skip (exists)"; else
    mlip-linewidth-cache-fc --phono3py-yaml "$P3" --fc2-forces "$F2_NORES" --fc3-forces "$F3_NORES" --cache-dir "$FC_V2_NORES" --fc2-source dfpt --dfpt-fc2 "$FC2_DFPT" --mesh "${MESH_ARGS[@]}" --populate-mesh-cache
fi

# Variant flags follow RESULTS_dfpt_bilayer_20260715.md and driver_extras.sh:
# V1 is reused; V2 is DFPT injected (loto/noloto x residual on/off); V3 is MLIP at DFPT geometry.
run_variant() {
    local name="$1" fc3="$2" fc2="$3" fc_cache="$4" injected="${5:-}"
    local gamma="$OUTPUT_DIR/gamma_${name}.npz"
    local gamma_cache="$OUTPUT_DIR/gamma_${name}"
    local w_h5="$OUTPUT_DIR/W_${name}.h5"
    if [ -f "$gamma" ]; then
        echo "skip $name (exists)"
        return
    fi
    local scatter=(mlip-linewidth-scatter-gpu --phono3py-yaml "$P3" --fc3-forces "$fc3" --fc2-forces "$fc2" --mesh "${MESH_ARGS[@]}" --temperature "$TEMP" --batch-size 1 --fallback-lang NONE --max-w-gb 500 --force-recompute-gamma-detail --cache-dir "$gamma_cache" --fc-cache-dir "$fc_cache" --output "$w_h5")
    if [ -n "$injected" ]; then
        scatter+=(--injected-modes "$injected")
    fi
    multi_rank "${scatter[@]}"
    mlip-linewidth-extract-gamma --w-h5 "$w_h5" --cache-dir "$gamma_cache" --yaml "$P3" --mesh "${MESH_ARGS[@]}" --temperature "$TEMP" --out "$gamma"
}

echo "=== [6] six DFPT-geometry scatter/extract variants (V1 is reused) ==="
run_variant V2_loto_res "$F3_RES" "$F2_RES" "$FC_V2_RES" "$MODES_LOTO"
run_variant V2_loto_nores "$F3_NORES" "$F2_NORES" "$FC_V2_NORES" "$MODES_LOTO"
run_variant V2_noloto_res "$F3_RES" "$F2_RES" "$FC_V2_RES" "$MODES_NOLOTO"
run_variant V2_noloto_nores "$F3_NORES" "$F2_NORES" "$FC_V2_NORES" "$MODES_NOLOTO"
run_variant V3_res "$F3_RES" "$F2_RES" "$FC_V3_RES"
run_variant V3_nores "$F3_NORES" "$F2_NORES" "$FC_V3_NORES"

echo "=== [7] compare V1 plus all six DFPT-geometry variants ==="
if [ -f "$OUTPUT_DIR/figures_full/comparison_summary.csv" ]; then
    echo "skip (exists)"
else
    mlip-linewidth-compare --mesh "${MESH_ARGS[@]}" --temperature "$TEMP" --out-dir "$OUTPUT_DIR/figures_full" --variants \
        "mlip_strict=$V1_YAML,$V1_FC2,$V1_GAMMA_NPZ,ev" \
        "dfpt_loto_res=$P3,$FC2_DFPT,$OUTPUT_DIR/gamma_V2_loto_res.npz,ry,$MODES_LOTO" \
        "dfpt_loto_nores=$P3,$FC2_DFPT,$OUTPUT_DIR/gamma_V2_loto_nores.npz,ry,$MODES_LOTO" \
        "dfpt_noloto_res=$P3,$FC2_DFPT,$OUTPUT_DIR/gamma_V2_noloto_res.npz,ry,$MODES_NOLOTO" \
        "dfpt_noloto_nores=$P3,$FC2_DFPT,$OUTPUT_DIR/gamma_V2_noloto_nores.npz,ry,$MODES_NOLOTO" \
        "mlip_at_dfpt_res=$P3,$FC_V3_RES/fc2.npy,$OUTPUT_DIR/gamma_V3_res.npz,ev" \
        "mlip_at_dfpt_nores=$P3,$FC_V3_NORES/fc2.npy,$OUTPUT_DIR/gamma_V3_nores.npz,ev"
fi

echo "=== DFPT linewidth factorial complete: $OUTPUT_DIR ==="
