# AGENTS.md — using the MACE-interlayer → phonopy setup

Operating guide for an agent (or student) working **inside** this directory
(`/pscratch/sd/j/jdgeorga/reu_tests`) on NERSC Perlmutter. To (re)build the setup
from scratch, read `SETUP_MACE_PHONON.md` instead.

## Golden rules

1. **Always source the env first, every shell:**
   ```bash
   source /pscratch/sd/j/jdgeorga/reu_tests/load_mace_phonon_env.sh
   ```
   It prints an import sanity report. If any line says `IMPORT FAILED`, stop and
   fix the env before running anything (see Troubleshooting).
2. **The harmonic MACE band-structure examples run on a CPU interactive node** and use `--device cpu`. (The anharmonic *linewidth* examples default to `--device cuda` — they need GPUs; see the Anharmonic section below.) The QE workflow requires a GPU allocation — use `sbatch`, never `bash`, for the QE `run.sh` scripts.
3. **These are research repos owned by the user.** You may make local git commits;
   **never `git push`** without explicit approval.
4. **Don't edit files under `mace_phonon_env/`** (the venv) by hand.

## Run the examples

### MACE workflow (CPU, fast)
```bash
source load_mace_phonon_env.sh
bash examples/mose2_monolayer/run.sh      # MoSe2 monolayer phonon band structure (~33 s)
bash examples/mose2_wse2_bilayer/run.sh   # MoSe2/WSe2 bilayer, interlayer coupling (~39 s)
```

### QE workflow (GPU, debug queue)
```bash
source load_mace_phonon_env.sh            # still needed for Python env
sbatch examples/mose2_monolayer_qe/run.sh     # MoSe2 monolayer via DFT (~8 min, 8 nodes)
sbatch examples/mose2_wse2_bilayer_qe/run.sh  # MoSe2/WSe2 bilayer via DFT (~10 min, 8 nodes)
```

Outputs land in each example's `run/`: `*_relaxed.xyz`, `*_forces.npy`,
`eigenvector_band.h5`, and **`band_structure.png`**.

## The pipeline (what `run.sh` does)

Four stages, all scripts in `repos/mlip_phonon_scattering/scripts/`. Stages 2 and 4
are identical between MACE and QE; only stages 1 and 3 differ by calculator.

### MACE variant

| Stage | Script | Output |
|-------|--------|--------|
| 1. Relax | `relax_structure.py IN.xyz OUT/prefix` | `OUT/prefix.xyz` |
| 2. Displacements | `generate_displacements.py OUT/prefix.xyz --supercell A B C --output OUT/disp.yaml` | `OUT/disp.yaml` |
| 3. Forces | `compute_mace_forces.py OUT/prefix.xyz OUT/disp.yaml --forces-output ... --energies-output ...` | `*_forces.npy` |
| 4. Solve | `solve_phonons.py --input-xyz OUT/prefix.xyz --forces ... --phonopy-yaml OUT/disp.yaml --mesh A B C --output-dir OUT` | `band_structure.png`, `eigenvector_band.h5` |

### QE variant

| Stage | Script | Output |
|-------|--------|--------|
| 1. Relax | `relax_structure_qe.py IN.xyz OUT/prefix --qe-config qe_config.yaml` | `OUT/prefix.xyz` |
| 2. Displacements | *(same as MACE)* | `OUT/disp.yaml` |
| 3. Forces | `compute_qe_forces.py OUT/prefix.xyz OUT/disp.yaml --qe-config qe_config.yaml --forces-output ... --energies-output ...` | `*_forces.npy` |
| 4. Solve | *(same as MACE)* | `band_structure.png`, `eigenvector_band.h5` |

The QE scripts launch `srun --exact --overlap pw.x` for each displacement.
`max_concurrent` (default 8) controls how many pw.x jobs run simultaneously;
`nranks_per_disp` (default 4) is GPUs per job. With 32 GPUs: 8 × 4 = 32 total.

## QE config (`qe_config.yaml`)

Each QE example has a `qe_config.yaml` with these key fields:
```yaml
pseudo_dir: /global/homes/j/jdgeorga/espresso/pseudo_new/
pseudo_files:              # ONCV PBE-1.2 pseudopotentials
  Mo: Mo_ONCV_PBE-1.2.upf
  Se: Se_ONCV_PBE-1.2.upf
  # W: W_ONCV_PBE-1.2.upf  (bilayer only)
ecutwfc: 60.0              # Ry — safe for ONCV TMD pseudos
ecutrho: 480.0             # Ry
input_dft: vdw-df-c09      # vdW functional for 2D systems
assume_isolated: 2D        # Coulomb cutoff for monolayer/bilayer
relax_kpoints: [12, 12, 1] # k-mesh for primitive cell relax
scf_kpoints: [1, 1, 1]     # Gamma-only for supercell SCF
nranks_per_disp: 4         # GPUs per pw.x job
max_concurrent: 8          # simultaneous pw.x jobs (32 GPUs / 4 = 8)
qe_module: espresso/7.5-libxc-7.0.0-gpu
```

No `--interlayer` flag: QE treats Mo, Se, W atoms uniformly; the bilayer
interlayer interaction is fully captured by DFT without special model switching.

## Monolayer vs bilayer — the ONLY difference is the calculator flags

**Monolayer (single model)** — relax/forces use:
```bash
--device cpu --default-dtype float64 --mace-model-path models/MoSe2.model
```
Input may be a plain `species:pos` xyz (no `atom_types`/`layer_ids` needed).

**Bilayer (interlayer stack)** — relax/forces add:
```bash
--interlayer \
--intralayer-models models/MoSe2.model models/WSe2.model \
--interlayer-model models/MoSe2_WSe2.model \
--layer-symbols "[['Mo','Se','Se'],['W','Se','Se']]"
```
Input **must** have `atom_types` (0,1,2 for the bottom layer; 3,4,5 for the top)
and `layer_ids` (0/1) arrays. `--intralayer-models` is one model **per layer**, in
layer order; `--interlayer-model` is one model **per adjacent layer pair**.
`--layer-symbols` lists the species of each layer, matching the `atom_types` blocks.

Stages 2 and 4 (displacements, solve) are identical for both — forces are forces.

## Models (`models/`)

Symlinks to the trained TMD potentials:
`MoSe2.model`, `WSe2.model` (intralayer) and `MoSe2_WSe2.model` (interlayer).
Other pairs (MoS2, WS2, MoS2_WS2, MoS2_WSe2) exist in
`repos/mace-interlayer/examples/TMD_trained_models/` — add a symlink to use them.

## Making a new system

- **New monolayer (e.g. WSe₂):** make a 3-atom `species:pos` xyz; point
  `--mace-model-path` at the matching intralayer model. Copy `mose2_monolayer/` as
  a template.
- **New bilayer (e.g. MoS₂/WS₂):** provide an xyz with `atom_types`/`layer_ids`;
  set `--intralayer-models <bottom> <top>`, `--interlayer-model <pair>`, and
  `--layer-symbols` to match. Copy `mose2_wse2_bilayer/` as a template.
- Keep `--default-dtype float64` for tight relaxation (`--fmax 1e-5`).
- Bigger systems: raise `--supercell` (accuracy ↑, cost ↑) and parallelize stage 3
  with `srun -n N python .../compute_mace_forces.py ...` (it auto-detects mpi4py).

## Sanity-checking results

Read `frequencies_thz` from `run/eigenvector_band.h5` (shape `[n_qpoints, n_bands]`):
- `n_bands` should be `3 × n_atoms` (9 monolayer, 18 bilayer).
- The 3 lowest modes go to ~0 at Γ (acoustic). No modes should be strongly
  negative (imaginary → unstable/under-relaxed structure).
- **Bilayer only:** expect small finite interlayer modes at Γ — a doubly-degenerate
  shear mode (~0.5–0.6 THz) and a breathing mode (~0.8–0.9 THz). If these are ~0,
  the interlayer model did **not** engage — check your `--interlayer` flags and that
  the input has `atom_types`/`layer_ids`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `import mace` points at `~/.local/...` or `e3nn` missing | `PYTHONNOUSERSITE=1` not set. Re-source `load_mace_phonon_env.sh` (it sets it). |
| `ModuleNotFoundError: macewrapper` / `n_layer` | The interlayer helpers are now the packaged `mlip_phonon_scattering.interlayer` modules (pip-installed editable) — no PYTHONPATH needed. Reinstall editable (`pip install -e repos/mlip_phonon_scattering` on branch `split_mlip_phonon_lifetime`) and re-source `load_mace_phonon_env.sh`. |
| `RuntimeError: Unknown keyword argument 'compute_edge_forces'` | A model was loaded with stock `MACECalculator`. The repo routes local models through `CompatMACECalculator`; ensure the `mlip_phonon_scattering` package is installed editable on branch `split_mlip_phonon_lifetime` so its `mlip_phonon_scattering.interlayer` helpers are importable (reinstall editable / re-source the loader if not). |
| `KeyError: atom_types` on a bilayer | Input xyz lacks `atom_types`/`layer_ids` arrays. Bilayer inputs require them. |
| Many imaginary (negative) frequencies | Structure under-relaxed (lower `--fmax`, more `--steps`) or supercell too small. |
| `command not found: python` shows 2.7 | You didn't source the load script / the venv isn't active. |
| QE: `mkdir: cannot create .../run: Permission denied` | Used `bash` instead of `sbatch` for a QE run.sh. Always `sbatch` for QE jobs. |
| QE: `srun: error: --overlap not supported` | Perlmutter SLURM version too old; contact NERSC. Current Perlmutter supports `--overlap`. |
| QE: `JOB DONE` missing from output, `RuntimeError` raised | SCF did not converge; check `qe_disps/disp_NNNN/scf.pwo`. Try increasing `max_seconds` or reducing `conv_thr`. Failed displacements can be re-run: the script skips any `.pwo` already containing `JOB DONE`. |
| QE: imaginary interlayer modes in bilayer | Relax did not converge tightly enough. The QE BFGS uses `forc_conv_thr~1e-3 Ry/Bohr` by default; this is sufficient but add `"ions": {"forc_conv_thr": 1e-4}` to `qe_config.yaml` for tighter convergence. |

## Where things live

- Env loader: `load_mace_phonon_env.sh`
- Workflow code: `repos/mlip_phonon_scattering/` (branch `split_mlip_phonon_lifetime`, pip -e)
  - MACE modules: `calculator.py`, `relax.py`, `forces.py`
  - QE modules: `qe_calculator.py`, `qe_relax.py`, `qe_forces.py`
  - Shared: `phonopy_io.py`, `solve.py`
  - Linewidth pipeline (anharmonic): `mlip_phonon_scattering/linewidth/` (stage code,
    `w_scatter_lib`, GPU engine, extract/plot/path) and
    `mlip_phonon_scattering/interlayer/` (`n_layer`, `macewrapper`)
  - Scripts: `scripts/` (harmonic stage scripts for both MACE and QE); the anharmonic
    pipeline runs via the `mlip-linewidth-*` console commands
- MACE + interlayer helpers: `repos/mace-interlayer/` (branch `port/mace-v0.3.16-interlayer`),
  helpers in `examples/interlayer_helpers/`
- trajEXtories (QE driver reference): `repos/trajEXtories/` (local clone)
- Models: `models/`
- MACE examples: `examples/mose2_monolayer/`, `examples/mose2_wse2_bilayer/`
- QE examples: `examples/mose2_monolayer_qe/`, `examples/mose2_wse2_bilayer_qe/`
- Full build/debug notes: `SETUP_MACE_PHONON.md`
- **Anharmonic (linewidth/lifetime) pipeline:** vendored into the
  `mlip_phonon_scattering` package —
  `repos/mlip_phonon_scattering/mlip_phonon_scattering/linewidth/` (stage code,
  `w_scatter_lib`, GPU engine, extract/plot/path) and `.../interlayer/` (`n_layer`,
  `macewrapper`); run via the `mlip-linewidth-*` console commands. Build in
  `SETUP_MACE_PHONO3PY.md`; physics in `THEORY_MACE_PHONO3PY.md`; run guide below.

---

## Anharmonic extension — phonon linewidths & lifetimes (phono3py_einsum GPU + MACE)

Everything above concerns the *harmonic* band-structure workflow (relax →
displacements → fc2 → ω(**q**)). This section is the operating guide for the
**anharmonic** pipeline: 3rd-order force constants → three-phonon scattering →
imaginary self-energy → **linewidth γ(q) and lifetime τ(q)** band-path figures.
Build/recreation instructions are in
[`SETUP_MACE_PHONO3PY.md`](SETUP_MACE_PHONO3PY.md); the physics/math is in
[`THEORY_MACE_PHONO3PY.md`](THEORY_MACE_PHONO3PY.md).

Throughout, **`$WORKDIR`** is *your own* Perlmutter working directory (e.g.
`/pscratch/sd/<a>/<user>/reu_tests`) and **`$ACCOUNT`** is your GPU allocation account
(e.g. `m2651_g`). The PI reference setup that this reproduces lives at
`/pscratch/sd/j/jdgeorga/reu_tests`; do not run in the PI's directory.

### What is validated

Both cases are GPU-validated on **4 nodes** at **`srun --overlap -n 4 --gpus-per-task=1`**
(`MPI_RANKS=4`; mesh 36×36×1, T = 50 K). See the rank-geometry note under "Running the
bilayer pipeline" — `srun -n 16` **hangs**.

- **Bilayer — reproduces the golden reference.** The **MoSe₂/WSe₂ aligned bilayer**
  runs end-to-end and **reproduces the golden** to γ_max 0.020164 vs golden 0.020166 THz
  (max|Δ| 6.7×10⁻⁴, mean|Δ| 7.2×10⁻⁷ THz; γ shape (1296, 18), fc2 clean at min ≈
  −5×10⁻⁸ THz, interlayer shear 0.575 THz doubly-degenerate pair + breathing 0.853 THz,
  4 figures). All commands below default to this case.
- **Monolayer — GPU-validated (self-contained physics checks pass).** The **MoSe₂
  monolayer** path (`examples/mose2_monolayer/`) runs the **same packaged pipeline** as
  the bilayer — the `mlip_phonon_scattering.linewidth` stage code and GPU engine
  (`gpu_scattering_W_phonons_bilayer_comm_mesh`) — driven by the same `mlip-linewidth-*`
  console commands with single-model flags. It is now GPU-validated: γ shape (1296, 9),
  γ_max 0.01741 THz, fc2 clean (min ≈ +3.2×10⁻⁸ THz, 9 bands), Γ-acoustic γ = 0, τ > 0,
  4 figures. There is **no golden reference** for the monolayer, so it is
  *physics-validated, not golden-reproduced*; converge mesh/supercells yourself for
  production numbers. See the monolayer subsection at the end of this section.

### Golden rules (in addition to the harmonic ones above)

1. **This pipeline needs GPUs.** Run it *inside* a GPU allocation (`salloc`/`sbatch`),
   never on a login node. The engine uses CuPy + the `phono3py_einsum` fork's
   `lang="GPU"` path.
2. **Source the env first, every shell** — the same `load_mace_phonon_env.sh`. Its
   sanity report now also imports `phono3py`, `cupy`, and
   `mlip_phonon_scattering.linewidth.w_scatter_lib`; if any of those says `IMPORT
   FAILED`, stop and fix the env (see `SETUP_MACE_PHONO3PY.md`).
3. **Do not pipe the env activation** (`... | tee`). A pipe runs in a subshell and
   silently loses the activated venv/modules. Source it directly.
4. **`python` on a login node is 2.7.** Always work through the venv's Python 3.12
   (sourcing the load script handles this).

### Running the bilayer pipeline

The run is driven by two scripts in `$WORKDIR/examples/mose2_wse2_bilayer/`:

- **`run_linewidth.sh`** — the 7-step pipeline (relax → yamls → forces → FC cache →
  GPU W-engine → gamma extract → plot), each step an `mlip-linewidth-*` console command,
  writing into an isolated `OUTPUT_DIR` (default `run_lifetime/`). MPI steps launch
  `srun --overlap -n $MPI_RANKS --gpus-per-task=1`; single-GPU steps use
  `srun --overlap -N1 -n1 --gpus-per-node=4`.
- **`run_salloc_pipeline.sh`** — a thin driver you run inside an allocation: it sources
  the env, does a CuPy smoke test on a compute-node GPU (also via `srun --overlap`, like
  every other srun step in the pipeline), calls `run_linewidth.sh`, then runs
  `mlip-linewidth-validate` on the `OUTPUT_DIR`.

> **Canonical, portable examples.** The version-controlled runnable examples ship with the
> branch checkout at `repos/mlip_phonon_scattering/examples/{mose2_wse2_bilayer,mose2_monolayer}/`
> (script-relative paths, no hard-coded PI paths). The `$WORKDIR/examples/...` copies
> referenced here are the PI's local run dirs; prefer the committed
> `repos/mlip_phonon_scattering/examples/` versions when reproducing.

> **⚠ Rank geometry — use `-n 4 --overlap`, NOT `-n 16`.** On this Perlmutter stack the
> pipeline **hangs at `srun -n 16 --gpus-per-task=1`** — a 16-rank mpi4py/PMI wireup
> deadlock, reproduced on *both* CPU-forces and GPU-forces (it is a rank-count issue, not a
> device or torch/CUDA issue; a 1-rank relax is fine). Run everything at **`srun --overlap
> -n 4 --gpus-per-task=1`** (`MPI_RANKS=4`). At `-n 4` the full bilayer (forces + 36×36×1
> scatter) completes in ~8 min. `--overlap` is required so successive `srun` steps in one
> allocation don't deadlock on GPU-slice accounting.

```bash
# 1. grab a GPU allocation (4 nodes x 4 GPUs)
salloc -N 4 -C gpu --gpus-per-node=4 -A $ACCOUNT -q interactive -t 60

# 2. from the head node of the allocation (MPI_RANKS defaults to 4):
cd $WORKDIR/examples/mose2_wse2_bilayer
bash run_salloc_pipeline.sh
```

`run_salloc_pipeline.sh` defaults `MPI_RANKS=4`. To run the steps yourself instead, source
the env and call `run_linewidth.sh` directly:

```bash
source $WORKDIR/load_mace_phonon_env.sh
cd $WORKDIR/examples/mose2_wse2_bilayer
MPI_RANKS=4 bash run_linewidth.sh
```

> `run_salloc_pipeline.sh` finishes by calling **`mlip-linewidth-validate`** on the
> `OUTPUT_DIR` — a self-contained physics/acceptance check (no external reference file).
> The optional `REFERENCE_GAMMA` env var can point it at a golden `.npz` if you have one,
> but none is required (see "Physics sanity checks" below).

`run_linewidth.sh` accepts these environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `OUTPUT_DIR` | `run_lifetime/` (under the example) | Isolated run directory — all outputs land here, never over the example dir |
| `MPI_RANKS` | `4` | MPI ranks for force + scattering stages (use 4; 16 hangs — see rank-geometry note above) |
| `DEVICE` | `cuda` | MACE device |
| `MESH_X`/`MESH_Y`/`MESH_Z` | `36`/`36`/`1` | q-mesh |
| `TEMP` | `50` | Temperature (K) |
| `PREFIX` | `MoSe2_WSe2_relaxed` | Prefix for relaxed structure + force files |
| `FORCE` | `no` | Set `yes` to rerun stages even when outputs exist |

### The 7 steps and their outputs

All steps are `mlip-linewidth-*` console commands provided by the
`mlip_phonon_scattering.linewidth` package (editable install). Case parameters (top of
`run_linewidth.sh`): mesh **36×36×1**, **T = 50 K**, models
`MO=MoSe2.model  W=WSe2.model  INTER=MoSe2_WSe2.model`,
`LAYER="[['Mo','Se','Se'],['W','Se','Se']]"`. Outputs land in `OUTPUT_DIR`
(default `run_lifetime/`).

| # | Step | Command (`mlip-linewidth-*`) | Key flags | Output |
|---|------|------------------------------|-----------|--------|
| 1 | Relax | `mlip-linewidth-relax IN.xyz OUT/PREFIX --interlayer --intralayer-models MO W --interlayer-model INTER --layer-symbols LAYER` | `--fmax 1.3e-5 --steps 1000 --maxstep 0.05 --device cuda` | `PREFIX.xyz` |
| 2a | phonopy yaml (fc2 SC) | `mlip-linewidth-phonopy-yaml PREFIX.xyz` | `--supercell 6 6 1 --output phonopy_disp.yaml` | `phonopy_disp.yaml` |
| 2b | phono3py yaml | `mlip-linewidth-phono3py-yaml PREFIX.xyz` | `--supercell 3 3 1 --phonon-supercell 6 6 1 --output phono3py_disp.yaml` | `phono3py_disp.yaml` |
| 3a | 2nd-order forces (MPI) | `mlip-linewidth-forces2 PREFIX.xyz phonopy_disp.yaml --interlayer --intralayer-models MO W --interlayer-model INTER --layer-symbols LAYER` | `--output *_forces_2nd.npy --device cuda` | `*_forces_2nd.npy` |
| 3b | 3rd-order forces (MPI) | `mlip-linewidth-forces3 PREFIX.xyz phono3py_disp.yaml --interlayer --intralayer-models MO W --interlayer-model INTER --layer-symbols LAYER` | `--supercell-matrix 3 3 1 --output *_forces_3rd.npy --device cuda` | `*_forces_3rd.npy` |
| 3c | 2nd-from-3rd forces (MPI) | `mlip-linewidth-forces2-from3 PREFIX.xyz phono3py_disp.yaml --interlayer --intralayer-models MO W --interlayer-model INTER --layer-symbols LAYER` | `--supercell-matrix 6 6 1 --output *_forces_2nd_from_3rd.npy --device cuda` | `*_forces_2nd_from_3rd.npy` |
| 4 | Symmetrized FC cache | `mlip-linewidth-cache-fc --phono3py-yaml phono3py_disp.yaml --fc2-forces F23 --fc3-forces F3` | `--cache-dir phonon_cache_m36x36x1 --populate-mesh-cache` | `phonon_cache_m36x36x1/{fc3,fc2}.npy` |
| 5 | GPU W-engine (MPI) | `mlip-linewidth-scatter-gpu --phono3py-yaml phono3py_disp.yaml --fc2-forces *_forces_2nd_from_3rd.npy --fc3-forces *_forces_3rd.npy` | `--mesh 36 36 1 --temperature 50 --batch-size 1 --fallback-lang NONE --max-w-gb 500 --force-recompute-gamma-detail` | `W_scattering_mesh_m36x36x1_T50_GPU.h5` + `phono3py_cache_36x36x1_T50.0K/gamma_detail-*.hdf5` |
| 6 | Extract exact γ | `mlip-linewidth-extract-gamma` | `--w-h5 W....h5 --cache-dir phono3py_cache_36x36x1_T50.0K --yaml phono3py_disp.yaml --mesh 36 36 1 --temperature 50 --out gamma_W_full_T50.npz` | `gamma_W_full_T50.npz` |
| 7 | Plot | `mlip-linewidth-plot` | `--name mose2_wse2_aligned --label "MoSe\$_2\$/WSe\$_2\$ aligned bilayer" --yaml phono3py_disp.yaml --fc2 phonon_cache_m36x36x1/fc2.npy --gamma-w gamma_W_full_T50.npz --mesh 36 36 1 --out-dir figures` | `figures/mose2_wse2_aligned_linewidth_lifetime_band_path{,_meshpoints}.{png,pdf}` |

**The same commands serve the monolayer** — swap the bilayer `--interlayer` /
`--intralayer-models` / `--interlayer-model` / `--layer-symbols` flags for a single
`--mace-model-path models/MoSe2.model --default-dtype float64` (and use the monolayer
4×4×1 / 8×8×1 supercells). See "Monolayer lifetime workflow" below.

**Why two supercells (step 2b):** the 3rd-order IFCs (fc3) need a displaced atom *per
pair* of atoms → the displacement count explodes, so fc3 uses a **small 3×3×1**
supercell. The harmonic fc2 that feeds the scattering matrix elements is built on the
**larger 6×6×1** phonon supercell, from a dedicated 2nd-from-3rd force set (step 3c).
MACE makes the many fc3 force calls cheap.

**Restartability:** each step is guarded by `need()` — skipped if its output exists
(set `FORCE=yes` to force a rerun). Step 5 uses `--force-recompute-gamma-detail`, so a
clean run recomputes the per-grid-point `gamma_detail-*.hdf5` regardless of cache.

### Two figures you get

- **`..._linewidth_lifetime_band_path.png/pdf`** — the smooth figure. Mesh γ is mapped
  onto the Γ–K–M–Γ path by **joint (q, ω) inverse-distance interpolation** (robust
  across band crossings). Top panel = linewidth γ (THz); bottom = lifetime τ (ps).
- **`..._meshpoints.png/pdf`** — the exact figure. Only mesh modes that land *exactly*
  on the Γ–K–M–Γ segments are plotted, with their raw γ (no interpolation). Sparser,
  but a ground-truth cross-check of the smooth figure at K and M.

### Physics sanity checks (self-contained — no external reference)

Run these after a pipeline completes. They are the acceptance criteria; if any fails,
the run is not trustworthy.

1. **Harmonic base is clean — no imaginary fc2 modes at the relaxed geometry.** The
   scattering matrix elements are built on the fc2 phonons, so a bad harmonic solution
   poisons everything downstream.
   ```bash
   python - <<'PY'
   import numpy as np, phono3py as p3
   from phonopy import Phonopy
   ph3 = p3.load("phono3py_disp.yaml", log_level=0)
   ph = Phonopy(ph3.unitcell, supercell_matrix=ph3.phonon_supercell_matrix,
                primitive_matrix=ph3.primitive_matrix)
   ph.force_constants = np.load("phonon_cache_m36x36x1/fc2.npy")
   ph.symmetrize_force_constants()
   ph.run_mesh([36,36,1], is_gamma_center=True)
   f = ph.get_mesh_dict()["frequencies"]
   print("min freq (THz):", f.min())        # tiny negatives near Gamma are numerical
   assert f.min() > -0.05, "imaginary fc2 modes -> relax harder / bigger supercell"
   print("fc2 OK")
   PY
   ```
2. **γ is non-negative and finite.** The γ extractor (`mlip-linewidth-extract-gamma`)
   already *enforces* this (raises on non-finite γ or γ < −1e-9). Confirm from the npz:
   ```bash
   python - <<'PY'
   import numpy as np
   z = np.load("gamma_W_full_T50.npz")
   g = z["gamma_W_full"]                      # (num_grid, nband), THz
   assert np.all(np.isfinite(g)) and g.min() >= -1e-9
   print("gamma OK  min/max THz:", g.min(), g.max(), " shape:", g.shape)  # ~(1296, 18), max ~0.0202
   PY
   ```
3. **Lifetimes are positive.** τ = 1/(4π·γ) is > 0 wherever γ > 0 (and +∞ for the γ = 0
   acoustic modes near Γ, which is physical — no decay channel). Any finite τ ≤ 0 is a bug.
4. **Both figures render** and show the expected features: acoustic branches with γ → 0
   near Γ, larger γ (broader modes) at higher ω / near K and M, the interlayer shear
   (~0.57 THz) and breathing (~0.85 THz) modes present, and the `_meshpoints` figure
   agreeing with the smooth figure where mesh points land on the path.

See [`THEORY_MACE_PHONO3PY.md`](THEORY_MACE_PHONO3PY.md) §7 for what the magnitudes and
trends *mean*.

### The τ / γ convention (know this before you interpret a figure)

- **γ (linewidth)** is the **imaginary phonon self-energy = HWHM**, reported in **THz**
  (ordinary frequency). It is the *exact* W_scatter reduction of the per-triplet
  `gamma_detail`, i.e. a true Brillouin-zone sum over the 36×36×1 mesh, **not** an
  interpolation.
- **τ (lifetime)** in ps is **τ = 1 / (4π·γ) = 1 / (2·2π·γ)**
  (`mlip_phonon_scattering.linewidth.linewidth_path.lifetime_from_gamma`): the factor
  **2** turns HWHM into FWHM = 2γ,
  and **2π** converts ordinary-frequency THz to angular frequency so 1/ω comes out in
  ps. γ ≤ 0 → τ = +∞. The engine's stored `W_tau` uses the same factor.
- **Project rule (do not violate):** interpolate **γ**, never τ. γ is smooth, additive,
  and non-negative; τ = 1/γ diverges for soft-partner modes and is meaningless to
  interpolate. This is recorded as `feedback-linewidth-definition`.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `IMPORT FAILED: cupy` / `phono3py` in the sanity report | The GPU env isn't built. See `SETUP_MACE_PHONO3PY.md`. |
| `import phono3py` resolves outside `repos/phono3py_einsum` | Reinstall the fork editable; verify branch `port/phono3py-4.3.3` and check `phono3py.__file__`. |
| `RuntimeError: CuPy is required` / no CUDA device | You launched on a login node or without GPUs. Run inside `salloc`/`sbatch -N.. --gpus-per-node=4`. |
| Env "loads" but Python is 2.7 / imports vanish | You *piped* the source command (subshell). Source it directly, un-piped. |
| `Detected N all-zero/invalid gamma_detail files` | The GPU backend emitted empty detail. The pipeline runs with `--fallback-lang NONE` (GPU-only, fail loud). To repair on CPU, rerun step 5 with `--fallback-lang C`. |
| `Estimated W_rate memory (.. GB) exceeds --max-w-gb` | The dense W matrix is too big. Raise `--max-w-gb`, or add `--skip-assemble` (the figures need only `gamma_detail`). |
| `frequency_points_at_bands` error under `lang=GPU` | The GPU path requires `frequency_points_at_bands=True`; the driver sets it. Don't override it. |
| Non-finite / negative γ raised by `extract_gamma_example.py` | A genuine upstream failure (under-relaxed structure, bad fc3, or all-zero `gamma_detail`). Re-check step 1 (fc2 clean?) and step 5 logs. |
| `ModuleNotFoundError: mlip_phonon_scattering...` / `mlip-linewidth-* not found` | The package isn't installed editable on this branch. On `split_mlip_phonon_lifetime` run `pip install -e repos/mlip_phonon_scattering`, then re-source `load_mace_phonon_env.sh`. |
| `undefined symbol: __itt_*` from torch | Set `LD_PRELOAD` to the ittnotify stub; `run_linewidth.sh` does this automatically if `repos/mlip_phonon_scattering/ittnotify_stub/libittnotify.so` exists. |

### Monolayer lifetime workflow (GPU-validated — self-contained physics checks)

> **Status.** The monolayer path is **GPU-validated: all four physics checks above
> pass** (γ shape (1296, 9), γ_max 0.01741 THz, fc2 clean at min ≈ +3.2×10⁻⁸ THz over 9
> bands, Γ-acoustic γ = 0, τ > 0, 4 figures). It shares the validated bilayer code path.
> There is **no golden reference** for the monolayer, so it is *physics-validated, not
> golden-reproduced* — the checks are the acceptance criteria, and mesh/supercell
> convergence for production numbers is **your** responsibility.

A monolayer (e.g. MoSe₂, 3 atoms) runs the **same 7-step pipeline** as the bilayer
through the same `mlip-linewidth-*` commands — it just drops the interlayer machinery
and uses a single model. It ships as `examples/mose2_monolayer/` with its own
`run_linewidth.sh` + `run_salloc_pipeline.sh`, writing into an isolated `OUTPUT_DIR`
(default `lifetime_run/`). Same rank geometry as the bilayer: `srun --overlap -n 4
--gpus-per-task=1` (`MPI_RANKS=4`); `srun -n 16` hangs.

```bash
salloc -N 4 --gpus-per-node=4 -A $ACCOUNT -q interactive -t 60
cd $WORKDIR/examples/mose2_monolayer
bash run_salloc_pipeline.sh
```

What differs from the bilayer:

- **Single model, no layers.** Relax/forces use `--mace-model-path models/MoSe2.model
  --default-dtype float64` (no `--interlayer` / `--intralayer-models` /
  `--interlayer-model` / `--layer-symbols`). The input xyz is a plain `species:pos` cell
  with no `atom_types`/`layer_ids`.
- **Supercells/mesh.** fc3 uses a **4×4×1** supercell and fc2/phonon an **8×8×1**
  supercell (`mlip-linewidth-phono3py-yaml --supercell 4 4 1 --phonon-supercell 8 8 1`);
  mesh 36×36×1, T = 50 K.
- **Branch count.** Expect **9** branches (3-atom cell), not 18. The final
  `mlip-linewidth-validate` runs with `--expected-atoms 3 --expected-bands 9`.
- **No interlayer modes.** There is no shear/breathing pair to check; the interlayer
  physics checks do not apply.

The four physics checks above are the only acceptance criteria (there is no monolayer
golden reference in this project) — and the validated run passes all four (γ_max 0.01741
THz, fc2 clean, 9 bands, 4 figures). Because there is no reference to diff against,
still inspect the figures and reconfirm γ ≥ 0 / finite and τ > 0 whenever you change the
mesh, supercells, temperature, or model.
