# MoSe2/WSe2 DFPT-harmonic linewidth sweep

This script reproduces the seven-variant comparison: one reused all-MLIP V1
reference and six DFPT-geometry variants (DFPT modes with 2D-LOTO on/off and
residual subtraction on/off, plus MLIP harmonic modes at the DFPT geometry).
Run it from an active GPU allocation:

```bash
export ALLOC_JOBID=<jobid>
bash run_dfpt_linewidth.sh
```

All products go to `OUTPUT_DIR`, which defaults to
`.../runs/mose2_wse2_bilayer_dfpt_${MESH// /x}_T${TEMP}`. The script is
restartable: completed artifacts are skipped.

Overrides (defaults):

- `MLIP_PHONON_ROOT`: inferred as `.../split_mlip_generators`; `MODELS`:
  `$MLIP_PHONON_ROOT/models`; `DFPT_DIR`:
  `.../tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix`.
- `MESH`: `"36 36 1"`; `TEMP`: `50`; `OUTPUT_DIR`:
  `$MLIP_PHONON_ROOT/runs/mose2_wse2_bilayer_dfpt_${MESH// /x}_T${TEMP}`;
  `MPI_RANKS`: `4`; `ALLOC_JOBID`: required, with no default.
- `V1_DIR`: `$MLIP_PHONON_ROOT/runs/mose2_wse2_bilayer_36x36x1_T50`;
  `V1_GAMMA_NPZ`: `$V1_DIR/gamma_W_full_T50.npz`; `V1_YAML`:
  `$V1_DIR/phono3py_disp.yaml`; `V1_FC2`:
  `$V1_DIR/phonon_cache_m36x36x1/fc2.npy`.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`: `1`. An existing `LD_PRELOAD` is retained;
  the local `ittnotify_stub` is prepended when present.

Reproducibility gotchas:

- **matdyn 2D-LOTO requires `OMP_NUM_THREADS=1`.** `PHonon/PH/rigid.f90`'s
  `rgd_blk` 2D-LOTO G-sum OMP loops have a race that corrupts the Γ non-analytic
  term with >1 thread (gate: 8.9 cm⁻¹ @ OMP>1 → 0.000000 cm⁻¹ @ OMP=1).
- Use `q-e-epw-tdbe-speedup/bin/matdyn.x` with `cray-fftw` and
  `cray-hdf5-parallel`; the script writes an allocation-aware shim in
  `$OUTPUT_DIR/bin/matdyn.x`.
- The QE→phono3py eigenvector Bloch gauge sign is **−1**. The script passes
  `--gauge-sign -1`; the overlap residual is diagnostic only for this
  NAC-sensitive heterobilayer.
