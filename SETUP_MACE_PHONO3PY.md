# Setting up the anharmonic pipeline — phono3py_einsum (GPU) + MACE (recreation guide)

This document explains how to build, **from scratch in your own NERSC Perlmutter
working directory**, the environment that computes **phonon linewidths and
lifetimes** (three-phonon scattering) for the MoSe₂/WSe₂ bilayer using the
`phono3py_einsum` GPU fork on top of MACE forces. It is the anharmonic companion to
[`SETUP_MACE_PHONON.md`](SETUP_MACE_PHONON.md) (the harmonic MACE/QE env) — that
document is a **prerequisite**; this one adds the GPU scattering stack on top of it.
For how to *run* the finished pipeline see [`AGENTS.md`](AGENTS.md); for the physics
see [`THEORY_MACE_PHONO3PY.md`](THEORY_MACE_PHONO3PY.md).

> **Notation.** `$WORKDIR` is *your* working directory (e.g.
> `/pscratch/sd/<a>/<user>/reu_tests`); `$ACCOUNT` is your GPU allocation account
> (e.g. `m2651_g`). The PI reference setup that this reproduces lives at
> `/pscratch/sd/j/jdgeorga/reu_tests` — reference it, do not run in it.
>
> **Status:** the **MoSe₂/WSe₂ aligned bilayer** linewidth pipeline is validated
> end-to-end on Perlmutter (**4 nodes, `srun --overlap -n 4 --gpus-per-task=1`**,
> mesh 36×36×1, T = 50 K): it **reproduces the golden reference** — γ_max 0.020164 vs
> golden 0.020166 THz, max|Δ| 6.7×10⁻⁴, mean|Δ| 7.2×10⁻⁷ THz. The **MoSe₂ monolayer**
> linewidth path (the NEW single-model path) is now **GPU-validated (self-contained
> physics checks pass)**; there is **no golden reference for it**, so it is
> *physics-validated, not golden-reproduced* — γ_max 0.01741 THz, 9 bands, fc2 clean,
> 4 figures. Mesh/supercell convergence for production numbers remains **your**
> responsibility. Run at **`srun --overlap -n 4 --gpus-per-task=1`** (`MPI_RANKS=4`);
> **`srun -n 16` HANGS** (16-rank mpi4py/PMI wireup deadlock, device-independent). See
> [`AGENTS.md`](AGENTS.md).
>
> **Physics invariant:** γ is the HWHM imaginary self-energy (THz); τ = 1/(4πγ);
> interpolate γ, never τ. See [`THEORY_MACE_PHONO3PY.md`](THEORY_MACE_PHONO3PY.md).

---

## 0. What this provides

On top of the harmonic MACE env, this adds:

- The **`phono3py_einsum` fork** — phono3py 4.3.4.dev with a `lang="GPU"` imaginary
  self-energy path (CuPy `einsum` evaluation of the fc3 interaction), paired with
  phonopy 4.3.1 and cupy 13.6.
- The **linewidth/lifetime pipeline**, vendored into the **`mlip_phonon_scattering`
  package** (branch `split_mlip_phonon_lifetime`, pip-installed editable):
  `mlip_phonon_scattering/linewidth/` holds the stage code (relax → displacements →
  2nd/3rd forces → symmetrized FC cache → GPU W-engine), the exact `gamma_detail` → γ
  reduction library `w_scatter_lib`, and the band-path γ extraction + linewidth/lifetime
  plotting; `mlip_phonon_scattering/interlayer/` holds the bilayer `n_layer`/`macewrapper`
  helpers. The whole pipeline is driven by the installed `mlip-linewidth-*` console
  commands. There is **no separate `phonon_linewidth` repo** anymore.
- Env wiring in `load_mace_phonon_env.sh` so all of the above is importable.

The end product is the 7-step pipeline driven by
`examples/mose2_wse2_bilayer/run_linewidth.sh` (and its allocation wrapper
`run_salloc_pipeline.sh`), producing the linewidth/lifetime band-path figures in an
isolated `OUTPUT_DIR`. A parallel `examples/mose2_monolayer/` ships the same pipeline
for the single-model monolayer case.

The repositories you use (under `$WORKDIR/repos/`):

| Directory | Repository | Branch |
|---|---|---|
| `mace-interlayer` | `https://github.com/jornada-group/mace-interlayer.git` | `port/mace-v0.3.16-interlayer` |
| `mlip_phonon_scattering` | `https://github.com/jdgeorga/mlip_phonon_scattering.git` | `split_mlip_phonon_lifetime` |
| `phono3py_einsum` | `https://github.com/jdgeorga/phono3py_einsum.git` | `port/phono3py-4.3.3` |

> **What is vendored vs. external.** The linewidth pipeline itself — the interlayer
> helpers (`mlip_phonon_scattering/interlayer/{n_layer,macewrapper}`), `_phonon_unfolding`,
> `w_scatter_lib`, the GPU engine, and the extract/plot/path modules (all under
> `mlip_phonon_scattering/linewidth/`) — is **vendored into the `mlip_phonon_scattering`
> package**; a single `pip install -e repos/mlip_phonon_scattering` provides it all. Only
> two dependencies stay **external**: `mace-interlayer` (the mace-torch fork + trained
> models) and `phono3py_einsum` (the `lang="GPU"` phono3py fork). Trained models are still
> referenced by path, not committed.

> **`trajEXtories` is NOT required** for the linewidth pipeline — there are no
> `trajEX`/`PWTask` imports anywhere under `mlip_phonon_scattering/linewidth/`. It belongs
> only to the QE harmonic path. Skip it unless you also want the QE workflow.

## Final layout (additions in **bold**)

```
$WORKDIR/
  load_mace_phonon_env.sh          # source first, every session (extended for GPU pipeline)
  mace_phonon_env/                 # venv (--system-site-packages on pytorch/2.11.0)
  repos/
    mlip_phonon_scattering/        # branch: split_mlip_phonon_lifetime  (pip -e; pipeline vendored)
      **mlip_phonon_scattering/interlayer/  # n_layer, macewrapper (bilayer helpers)**
      **mlip_phonon_scattering/linewidth/   # stage code, w_scatter_lib, GPU engine, extract/plot/path**
    mace-interlayer/               # branch: port/mace-v0.3.16-interlayer
    **phono3py_einsum/             # branch: port/phono3py-4.3.3  (GPU fork; editable)**
  models/                          # symlinks -> mace-interlayer TMD_trained_models/*.model
  examples/
    mose2_wse2_bilayer/            # VALIDATED
      MoSe2_WSe2_bilayer.xyz
      **run_linewidth.sh           # 7-step linewidth pipeline -> OUTPUT_DIR (run_lifetime/)**
      **run_salloc_pipeline.sh     # env + cupy smoke test + run_linewidth.sh + validate**
    **mose2_monolayer/             # GPU-VALIDATED (physics checks; single-model)**
      **MoSe2_monolayer.xyz**
      **run_linewidth.sh           # same 7 steps, single MACE model -> OUTPUT_DIR (lifetime_run/)**
      **run_salloc_pipeline.sh**
  SETUP_MACE_PHONON.md  SETUP_MACE_PHONO3PY.md  AGENTS.md
  THEORY_MACE_PHONON.md  THEORY_MACE_PHONO3PY.md
```

Large force arrays, HDF5 caches, and figures are written into an isolated `OUTPUT_DIR`
under each example (`run_lifetime/` for the bilayer, `lifetime_run/` for the monolayer)
— never over the example directory itself.

## Verified component versions

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.12.13 | `module pytorch/2.11.0` |
| torch | 2.11.0+cu129 | module (system site-packages) |
| mace-torch | 0.3.16 | editable, `repos/mace-interlayer` |
| **mlip_phonon_scattering** | **`split_mlip_phonon_lifetime`** | editable, `repos/mlip_phonon_scattering` (linewidth pipeline vendored) |
| e3nn | 0.4.4 | venv |
| ase | 3.29.0 | venv |
| **phonopy** | **4.3.1** | venv (`>=4.3.0,<4.4.0`) |
| **phono3py** | **4.3.4.dev (fork)** | editable, `repos/phono3py_einsum` (`port/phono3py-4.3.3`) |
| **phonors** | **0.3.0** | pulled by phonopy 4.3.x |
| **cupy** | **13.6.0** | `cupy-cuda12x==13.6.0` |

The critical compatibility set is:

```
phono3py_einsum branch  port/phono3py-4.3.3
phonopy                 >=4.3.0,<4.4.0
cupy-cuda12x            ==13.6.0
nanobind                <2.10   (build-time only)
```

Do not upgrade one member of this set without rebuilding and revalidating the GPU
imaginary-self-energy path. Note the harmonic setup pins phonopy **4.2.2**; the
anharmonic GPU stack requires phonopy **4.3.1** to match the fork. If you are
extending an existing harmonic `mace_phonon_env`, the `pip install
"phonopy>=4.3.0,<4.4.0"` in step 3 upgrades it (the harmonic examples continue to
work on 4.3.1).

---

## 1. Prerequisite: the harmonic env

Build the harmonic environment first, exactly as in
[`SETUP_MACE_PHONON.md`](SETUP_MACE_PHONON.md) §§1–3: the `pytorch/2.11.0` module,
a `--system-site-packages` venv (`mace_phonon_env`), `PYTHONNOUSERSITE=1`, then
`pip install -e repos/mace-interlayer` (mace-torch 0.3.16, e3nn 0.4.4, ase,
matscipy) → `pip install phonopy h5py` → `pip install -e repos/mlip_phonon_scattering`.

```bash
export WORKDIR="$PSCRATCH/reu_tests"       # <-- your own directory
export ACCOUNT="<your_gpu_account>"        # e.g. m2651_g
cd "$WORKDIR"
module load pytorch/2.11.0
source mace_phonon_env/bin/activate
export PYTHONNOUSERSITE=1                    # MANDATORY — see Gotcha 1
```

### Gotcha 1 (carried over) — `PYTHONNOUSERSITE=1` is mandatory
The `--system-site-packages` venv would otherwise see the per-user
`~/.local/perlmutter/pytorch2.11.0/...` `mace` install, which **shadows**
`repos/mace-interlayer` and makes pip think `e3nn` is already satisfied.
`PYTHONNOUSERSITE=1` hides the user site; torch still comes from the module's
*system* site-packages. `load_mace_phonon_env.sh` sets this automatically.

## 2. Clone / checkout the pipeline repos into `$WORKDIR/repos/`

The linewidth pipeline is now **vendored into `mlip_phonon_scattering`** on branch
`split_mlip_phonon_lifetime`; there is no longer a separate `phonon_linewidth` clone.
You only add the `phono3py_einsum` fork here and move `mlip_phonon_scattering` onto the
lifetime branch:

```bash
cd "$WORKDIR/repos"

# GPU/einsum phono3py fork (lang="GPU" imaginary self-energy)
git clone https://github.com/jdgeorga/phono3py_einsum.git
git -C phono3py_einsum checkout port/phono3py-4.3.3     # pairs with phonopy 4.3.1, cupy 13.6
git -C phono3py_einsum remote add upstream https://github.com/phonopy/phono3py.git  # optional

# the linewidth pipeline now lives on this branch of mlip_phonon_scattering
git -C mlip_phonon_scattering fetch origin
git -C mlip_phonon_scattering checkout split_mlip_phonon_lifetime

cd "$WORKDIR"
```

You should already have `mace-interlayer` (`port/mace-v0.3.16-interlayer`) and
`mlip_phonon_scattering` from the harmonic setup. If not, clone them per
`SETUP_MACE_PHONON.md` §2. Confirm the branches:

```bash
git -C repos/mace-interlayer      branch --show-current   # port/mace-v0.3.16-interlayer
git -C repos/mlip_phonon_scattering branch --show-current # split_mlip_phonon_lifetime
git -C repos/phono3py_einsum      branch --show-current   # port/phono3py-4.3.3
```

## 3. Build `phono3py_einsum` (the new part)

The fork builds a C/nanobind extension via `scikit_build_core`. Do it in this order,
with the venv active and `PYTHONNOUSERSITE=1` set:

```bash
# phonopy 4.3.1 (also pulls phonors 0.3.0)
PYTHONNOUSERSITE=1 python -m pip install "phonopy>=4.3.0,<4.4.0"

# CuPy for CUDA 12.x (matches the module's cu129 torch)
PYTHONNOUSERSITE=1 python -m pip install cupy-cuda12x==13.6.0

# build backend deps for scikit-build-core (nanobind must be < 2.10)
PYTHONNOUSERSITE=1 python -m pip install "nanobind<2.10" scikit-build-core setuptools_scm

# build the fork editable, no build isolation, GNU toolchain
CC=gcc CXX=g++ PYTHONNOUSERSITE=1 \
  python -m pip install -e repos/phono3py_einsum --no-build-isolation
```

- The fork's `pyproject.toml` uses `build-backend = scikit_build_core` and needs
  `scikit-build-core`, `nanobind<2.10`, `numpy` at build time — hence the explicit
  install and `--no-build-isolation` (so the build sees the venv's numpy/nanobind).
- It builds **without LAPACKE/BLAS**; the C extension still compiles. Result:
  `phono3py 4.3.4.dev` resolving to `repos/phono3py_einsum`.
- `CC=gcc CXX=g++` forces the GNU toolchain (the Cray `CC`/`cc` wrappers confuse
  scikit-build's compiler detection).

### Gotcha 2 — build order matters
Install `mace-interlayer` **before** anything that could pull `mace-torch` from
PyPI (carried over from the harmonic setup), and install the `phonopy` upgrade
**before** building `phono3py_einsum` so the fork links against 4.3.1, not the
harmonic 4.2.2. If you build the fork against the wrong phonopy you'll see version
mismatches at import.

### Gotcha 3 — a successful `import phono3py` is not enough
The W-engine calls `run_imag_self_energy(..., lang="GPU")`; that path is supplied
*only* by the fork. Always check the version **and** the source path, not just that
the import succeeds:

```bash
python -c "import phono3py; print(phono3py.__version__, phono3py.__file__)"
# -> 4.3.4.dev...  /.../$WORKDIR/repos/phono3py_einsum/phono3py/__init__.py
```

### Gotcha 4 — CuPy is CUDA-12 and needs a real GPU to exercise
`cupy-cuda12x==13.6.0` matches the module's cu129 torch. `import cupy` works on a
login node, but any *kernel* (e.g. the smoke test) must run on a compute-node GPU —
`run_salloc_pipeline.sh` does `srun -N1 -n1 --gpus-per-node=4 python -c "import
cupy ..."` for exactly this reason.

## 4. Install the pipeline package editable (no PYTHONPATH juggling)

The linewidth pipeline is now part of the `mlip_phonon_scattering` package, so it is
installed **editable** — there are **no** `PYTHONPATH` entries for `w_scatter_lib` or the
stage scripts anymore. If you did the harmonic `pip install -e repos/mlip_phonon_scattering`
in §1 while already on `split_mlip_phonon_lifetime`, the vendored `linewidth/` and
`interlayer/` subpackages plus the `mlip-linewidth-*` console commands are already
importable / on `PATH`. If you switched branches after installing, reinstall to refresh
the console entry points:

```bash
PYTHONNOUSERSITE=1 python -m pip install -e repos/mlip_phonon_scattering
```

This provides:

- `mlip_phonon_scattering.linewidth` — stage code (`relax`, `displacements`, `forces3`,
  `fc_cache`), the GPU engine (`gpu_scattering_W_phonons_bilayer_comm_mesh`), the exact γ
  reduction (`w_scatter_lib`), and the band-path extract/plot/path modules.
- `mlip_phonon_scattering.interlayer` — the `n_layer` / `macewrapper` bilayer helpers.
- the `mlip-linewidth-*` console commands (see §8 and `AGENTS.md`).

`load_mace_phonon_env.sh` (step 5) still exports
`PHONO3PY_EINSUM_PATH=$WORKDIR/repos/phono3py_einsum`; the GPU engine uses that env var
as a belt-and-suspenders `sys.path` fallback so it finds the fork even if the editable
install is missing.

## 5. Extend the load script

`load_mace_phonon_env.sh` in the PI reference is already extended for the GPU
pipeline. Because the linewidth pipeline and the interlayer helpers are now **packaged**
(editable install), the load script no longer prepends any `phonon_linewidth` or
`interlayer_helpers` directories to `PYTHONPATH`. If you are recreating it from the
harmonic version, the only GPU-pipeline addition is the fork env var:

```bash
# Forked phono3py (lang="GPU" imaginary self-energy).
export PHONO3PY_EINSUM_PATH="${MACE_PHONON_ROOT}/repos/phono3py_einsum"
```

and extend the sanity-report import list to exercise the packaged GPU stack via its
namespaces:

```python
mods = ["torch", "e3nn", "ase", "phonopy", "phono3py", "cupy", "mace",
        "mlip_phonon_scattering", "mlip_phonon_scattering.interlayer.n_layer",
        "mlip_phonon_scattering.interlayer.macewrapper",
        "mlip_phonon_scattering.linewidth.w_scatter_lib"]
```

`MACE_PHONON_ROOT` is auto-resolved to the script's own directory, so the load
script is portable to `$WORKDIR` with no edits beyond the above.

## 6. Models

`models/` holds **relative** symlinks into
`repos/mace-interlayer/examples/TMD_trained_models/`. The bilayer pipeline uses all
three (two intralayer + one interlayer):

```bash
mkdir -p "$WORKDIR/models" && cd "$WORKDIR/models"
ln -s ../repos/mace-interlayer/examples/TMD_trained_models/MoSe2-models_rmax2_stagetwo_compiled.model      MoSe2.model
ln -s ../repos/mace-interlayer/examples/TMD_trained_models/WSe2-models_rmax2_stagetwo_compiled.model       WSe2.model
ln -s ../repos/mace-interlayer/examples/TMD_trained_models/MoSe2_WSe2-models_rmax2_stagetwo_compiled.model MoSe2_WSe2.model
cd "$WORKDIR"
readlink -f models/*.model    # verify the targets resolve
```

## 7. Example inputs

Two examples ship with the pipeline as **canonical, version-controlled** directories that
come with the branch checkout —
`repos/mlip_phonon_scattering/examples/{mose2_wse2_bilayer,mose2_monolayer}/` (script-relative
paths, fully portable; no hard-coded PI paths). Run or copy those; any `$WORKDIR/examples/...`
copies are throwaway local run dirs, not the handoff artifact. Each has its own
`run_linewidth.sh` and `run_salloc_pipeline.sh`:

- **`examples/mose2_wse2_bilayer/`** (VALIDATED) — `MoSe2_WSe2_bilayer.xyz`, the 6-atom
  bilayer with `atom_types` (0–2 = layer 0 Mo,Se,Se; 3–5 = layer 1 W,Se,Se) and
  `layer_ids` (0/1) arrays. Uses all three models via the `--interlayer` flags. Same
  bilayer xyz as the harmonic example.
- **`examples/mose2_monolayer/`** (GPU-VALIDATED — self-contained physics checks) —
  `MoSe2_monolayer.xyz`, a plain 3-atom `species:pos` cell (no `atom_types`/`layer_ids`).
  Uses a single model via `--mace-model-path models/MoSe2.model --default-dtype float64`.

Both drivers write into an isolated `OUTPUT_DIR` (`run_lifetime/` for the bilayer,
`lifetime_run/` for the monolayer), never over the example directory. The
`run_salloc_pipeline.sh` scripts finish by calling `mlip-linewidth-validate` (a
self-contained physics/acceptance check); there is no private external reference file to
diff against.

---

## 8. Verification (how to know it works)

### 8a. Env imports (login node)

```bash
source "$WORKDIR/load_mace_phonon_env.sh"
# The sanity report must show NO "IMPORT FAILED", in particular:
#   phonopy    4.3.1
#   phono3py   4.3.4.dev   (from repos/phono3py_einsum)
#   cupy       13.6.0
#   mlip_phonon_scattering.linewidth.w_scatter_lib
python -c "import phono3py, cupy, mlip_phonon_scattering.linewidth.w_scatter_lib as w; print(phono3py.__file__)"
# -> must be under repos/phono3py_einsum/
# also confirm the console commands are on PATH:
command -v mlip-linewidth-scatter-gpu mlip-linewidth-extract-gamma mlip-linewidth-plot
```

> **Dependencies.** The force/scatter stages import `mpi4py` and the plotting stage
> imports `scipy`; both are provided by the `pytorch/2.11.0` module — verify they import
> (`python -c "import mpi4py, scipy"`).

### 8b. CuPy on a real GPU (inside an allocation)

```bash
salloc -N 1 --gpus-per-node=4 -A "$ACCOUNT" -q interactive -t 20
srun -N1 -n1 --gpus-per-node=4 python -c \
  "import cupy as cp; x=cp.arange(1000,dtype=cp.float64); print('device sum', float(x.sum()))"
# -> "device sum 499500.0"  (this is the smoke test run_salloc_pipeline.sh does)
```

### 8c. End-to-end bilayer run + physics checks

```bash
salloc -N 4 --gpus-per-node=4 -A "$ACCOUNT" -q interactive -t 60
cd "$WORKDIR/examples/mose2_wse2_bilayer"
bash run_salloc_pipeline.sh          # env + cupy smoke test + run_linewidth.sh + validate
```

> **⚠ Rank geometry — run at `-n 4 --overlap`, NOT `-n 16`.** The MPI stages inside
> `run_salloc_pipeline.sh` / `run_linewidth.sh` launch at **`srun --overlap -n 4
> --gpus-per-task=1`** (`MPI_RANKS=4`, the default). Do **not** raise this to 16: the
> pipeline **HANGS at `srun -n 16 --gpus-per-task=1`** — a 16-rank mpi4py/PMI wireup
> deadlock that is **device-independent** (it hangs on both CPU-forces and GPU-forces;
> a 1-rank relax is fine). The `--overlap` is required so successive `srun` steps in one
> allocation don't deadlock on GPU-slice accounting. This is the geometry the validated
> results were produced at.

Outputs land in the isolated `OUTPUT_DIR` (default `run_lifetime/`). Expected principal
outputs:

```
run_lifetime/MoSe2_WSe2_relaxed.xyz
run_lifetime/phonopy_disp.yaml   run_lifetime/phono3py_disp.yaml
run_lifetime/MoSe2_WSe2_relaxed_forces_2nd.npy  _forces_3rd.npy  _forces_2nd_from_3rd.npy
run_lifetime/phonon_cache_m36x36x1/{fc2,fc3}.npy
run_lifetime/phono3py_cache_36x36x1_T50.0K/gamma_detail-*.hdf5
run_lifetime/W_scattering_mesh_m36x36x1_T50_GPU.h5
run_lifetime/gamma_W_full_T50.npz
run_lifetime/figures/mose2_wse2_aligned_linewidth_lifetime_band_path{,_meshpoints}.{png,pdf}
```

The final `mlip-linewidth-validate` step enforces the acceptance criteria; the run is
correct when **all four self-contained physics checks pass** (detailed in `AGENTS.md` →
"Physics sanity checks"):

1. **No imaginary fc2 modes** at the relaxed geometry (min mesh frequency > −0.05 THz;
   the validated bilayer run's minimum is ≈ −5×10⁻⁸ THz, i.e. numerical zero).
2. **γ ≥ 0 and finite** — the γ extractor (`mlip-linewidth-extract-gamma`) *enforces*
   this (raises on non-finite γ or γ < −1e-9); confirm from `gamma_W_full_T50.npz`
   (shape (1296, 18); γ ranges 0 to γ_max 0.020164 THz — **reproduces the golden**
   0.020166 THz to max|Δ| 6.7×10⁻⁴, mean|Δ| 7.2×10⁻⁷ THz).
3. **τ > 0** wherever γ > 0 (τ = +∞ for the γ = 0 acoustic modes near Γ — physical).
4. **Both band-path figures render** with the expected features.

The validated bilayer additionally shows a doubly-degenerate interlayer shear pair
near 0.575 THz and a breathing mode near 0.853 THz at Γ (γ_max 0.020164 THz vs golden
0.020166). The physics checks are self-contained; separately, this run also
**reproduces the golden reference** end-to-end (max|Δ| 6.7×10⁻⁴, mean|Δ| 7.2×10⁻⁷ THz;
fc2 min ≈ −5×10⁻⁸ THz).

### 8d. Monolayer run (GPU-validated — self-contained physics checks)

The monolayer pipeline shares the validated bilayer code path with single-model flags
(`--mace-model-path models/MoSe2.model --default-dtype float64`, no `--interlayer`). It
uses an fc3 **4×4×1** / phonon **8×8×1** supercell pair, mesh 36×36×1, T = 50 K, and
expects **9** branches (3-atom cell).

```bash
salloc -N 4 --gpus-per-node=4 -A "$ACCOUNT" -q interactive -t 60
cd "$WORKDIR/examples/mose2_monolayer"
bash run_salloc_pipeline.sh          # writes into lifetime_run/ (srun --overlap -n 4 --gpus-per-task=1)
```

This path is now **GPU-validated: all four self-contained physics checks pass** (run
with `--expected-atoms 3 --expected-bands 9`) — γ shape (1296, 9), γ_max 0.01741 THz,
fc2 clean (min ≈ +3.2×10⁻⁸ THz, 9 bands), Γ-acoustic γ = 0, τ > 0, and 4 figures. Note
there is **no golden reference** for the monolayer, so it is **physics-validated, not
golden-reproduced**; the reported γ_max is from these validated case parameters, and
mesh/supercell convergence for production numbers is still **your** responsibility. Run
at `srun --overlap -n 4 --gpus-per-task=1` (`-n 16` hangs — see §8c). See the monolayer
subsection in `AGENTS.md`.

## 9. Notes for a future agent

- **Always run inside a GPU allocation.** The engine hard-requires CuPy + a visible
  CUDA device.
- **Never pipe the env activation** (`source ... | tee`): the pipe is a subshell and
  the activated venv/modules are lost.
- **Login-node `python` is 2.7.** Always go through the venv's 3.12.
- **The repos are the user's own.** Commit locally; **do not `git push`** without
  asking.
- **Scaling:** `--mesh` and the supercells (steps 2a/2b) set accuracy vs. cost. The
  dense W matrix scales as (N_ir·n_band)² — use `--max-w-gb` as a guardrail, or
  `--skip-assemble` when you only need `gamma_detail` for the figures. fc3 forces
  dominate wall time; they are MPI-parallel over displacements, launched at the only
  validated geometry `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`). Do **not**
  scale ranks blindly — `srun -n 16` **hangs** (16-rank mpi4py/PMI deadlock; see §8c).
- **Gamma-detail caches are temperature-tagged** (`phono3py_cache_..._T50.0K/`). Do
  not mix caches from different temperatures; a new T needs a fresh cache + output.
- **Monolayer is GPU-validated (physics checks)** — `examples/mose2_monolayer/` runs
  the same packaged pipeline with single-model flags (fc3 4×4×1 / phonon 8×8×1) and
  passes all four self-contained physics checks (γ_max 0.01741 THz, 9 bands, fc2 clean,
  4 figures). There is **no golden reference**, so it is physics-validated, not
  golden-reproduced; converge mesh/supercells yourself for production numbers. See the
  monolayer subsection in `AGENTS.md` and §8d.
- **Rank geometry is fixed at `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`).**
  `srun -n 16 --gpus-per-task=1` **hangs** (16-rank mpi4py/PMI wireup deadlock,
  device-independent — see §8c). `--overlap` lets successive `srun` steps share the
  allocation without deadlocking on GPU-slice accounting.
