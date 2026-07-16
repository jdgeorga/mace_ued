# Setting up the MACE-interlayer → phonopy workflow (recreation guide)

This document explains how to build this working directory from scratch on
**NERSC Perlmutter**. It is written so that a coding **agent** (or a person) can
reproduce the whole environment and verify it end-to-end. For how to *use* the
finished setup, see [`AGENTS.md`](AGENTS.md) and the per-example `README.md` files.

> **Status:** verified working on 2026-06-24 (MACE) and 2026-06-29 (QE) on Perlmutter.
> MACE examples: ~33 s / ~39 s (CPU). QE examples: ~8 min / ~10 min (8 nodes, 32 GPUs, debug queue).

---

## 0. What this provides

A self-contained directory that computes **2nd-order (harmonic) phonon band
structures** for TMD systems using either **MACE machine-learning interatomic
potentials** or **Quantum ESPRESSO DFT**. Both share the same displacement generation
and phonon solving steps; only relax and force evaluation differ.

**MACE workflow** (fast, CPU):
- **MoSe₂ monolayer** — single MACE model (`examples/mose2_monolayer/`)
- **MoSe₂/WSe₂ bilayer** — stacked *intralayer + interlayer* models via `NLayerCalculator` (`examples/mose2_wse2_bilayer/`)

**QE workflow** (DFT accuracy, GPU, SLURM):
- **MoSe₂ monolayer** — pw.x relax + SCF forces (`examples/mose2_monolayer_qe/`)
- **MoSe₂/WSe₂ bilayer** — pw.x relax + SCF forces (`examples/mose2_wse2_bilayer_qe/`)

Pipeline (both variants): `relax → generate phonopy displacements → compute forces → solve phonons → band_structure.png`.

## Final layout

```
reu_tests/
  load_mace_phonon_env.sh          # source this first, every session
  mace_phonon_env/                 # venv (--system-site-packages on pytorch/2.11.0)
  repos/
    mlip_phonon_scattering/        # branch: split-mlip   (interlayer + QE integration)
    mace-interlayer/               # branch: port/mace-v0.3.16-interlayer
      examples/interlayer_helpers/ # NLayerCalculator / MACEWCalculator helpers
    trajEXtories/                  # local clone — QE driver reference + PWTask utilities
  models/                          # symlinks -> mace-interlayer TMD_trained_models/*.model
  examples/
    mose2_monolayer/               # MACE: run.sh + MoSe2_monolayer.xyz
    mose2_wse2_bilayer/            # MACE: run.sh + MoSe2_WSe2_bilayer.xyz
    mose2_monolayer_qe/            # QE: run.sh + qe_config.yaml + MoSe2_monolayer.xyz
    mose2_wse2_bilayer_qe/         # QE: run.sh + qe_config.yaml + MoSe2_WSe2_bilayer.xyz
  SETUP_MACE_PHONON.md  AGENTS.md
```

## Verified component versions

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.12.13 | `module pytorch/2.11.0` |
| torch | 2.11.0+cu129 | module (system site-packages) |
| mace-torch | 0.3.16 | editable, `repos/mace-interlayer` |
| e3nn | 0.4.4 | installed into venv |
| ase | 3.29.0 | installed into venv |
| phonopy | 4.2.2 | installed into venv |

---

## 1. Environment (the proven recipe)

The base is NERSC's PyTorch module (gives a working, optimized torch); a venv on
top of it holds everything else. **Do not** build a fresh conda torch — the
module torch is the validated one.

```bash
cd /pscratch/sd/j/jdgeorga/reu_tests
module load pytorch/2.11.0                                   # Python 3.12 + torch 2.11
python3 -m venv --system-site-packages mace_phonon_env      # torch inherited from module
source mace_phonon_env/bin/activate
export PYTHONNOUSERSITE=1                                    # SEE GOTCHA 1 — essential
python -m pip install --upgrade pip
```

### Gotcha 1 — `PYTHONNOUSERSITE=1` is mandatory
The user site `~/.local/perlmutter/pytorch2.11.0/...` already contains a separate,
non-editable `mace-torch` install (from an earlier project). With a
`--system-site-packages` venv it is visible and:
- it **shadows** `repos/mace-interlayer`, so `import mace` resolves to the wrong copy;
- it makes `pip` think `e3nn` is "already satisfied", so e3nn never lands in the venv.

Setting `PYTHONNOUSERSITE=1` hides the user site. `torch` still comes from the
module's **system** site-packages (not the user site), so nothing breaks.
`load_mace_phonon_env.sh` exports this automatically.

## 2. Clone the two repos

```bash
mkdir -p repos && cd repos
git clone https://github.com/jdgeorga/mlip_phonon_scattering.git
git -C mlip_phonon_scattering checkout -b split-mlip          # interlayer integration branch

git clone https://github.com/jornada-group/mace-interlayer.git
git -C mace-interlayer checkout port/mace-v0.3.16-interlayer
cd ..
```

## 2b. Clone trajEXtories (QE driver reference)

trajEXtories provides the `minimal_QE_example` pattern that the QE workflow follows,
and its `input_pw.py` (`PWTask`) is importable as a utility. It has no `setup.py`, so
it is added to `PYTHONPATH` rather than pip-installed.

```bash
cd repos
git clone /pscratch/sd/j/jdgeorga/headless_job/trajEXtories trajEXtories
cd ..
```

`load_mace_phonon_env.sh` automatically prepends `repos/trajEXtories` to `PYTHONPATH`.

## 3. Install dependencies into the venv

```bash
# user site hidden so e3nn/ase/etc. install INTO the venv (not resolved from ~/.local)
# IMPORTANT: install mace-interlayer FIRST so `mace` is the fork, not the PyPI build.
PYTHONNOUSERSITE=1 python -m pip install -e repos/mace-interlayer   # mace-torch 0.3.16, e3nn==0.4.4, ase, matscipy, ...
PYTHONNOUSERSITE=1 python -m pip install phonopy h5py
PYTHONNOUSERSITE=1 python -m pip install -e repos/mlip_phonon_scattering   # the workflow package (setup.py)

# trajEXtories dependencies (needed for `from trajEX.calculator.inputs.input_pw import PWTask`)
PYTHONNOUSERSITE=1 python -m pip install "pydantic-yaml>=1.3.0" natsort
```

`mlip_phonon_scattering` ships a `setup.py` and installs editable as above (it
also puts the stage scripts — `relax_structure.py`, etc. — on PATH). It
deliberately does **not** list `mace-torch` in `install_requires`, so pip won't
pull the upstream PyPI build over the interlayer fork — hence install
`mace-interlayer` first. The load script (step 5) also adds the repo root to
`PYTHONPATH`, so the package works even without the editable install; the two are
redundant and harmless together. The interlayer helpers
(`interlayer_helpers`) are always provided via `PYTHONPATH`.

Sanity check:
```bash
PYTHONNOUSERSITE=1 python -c "import torch,e3nn,ase,phonopy,mace; \
print(torch.__version__, e3nn.__version__, phonopy.__version__); print(mace.__file__)"
# mace.__file__ MUST point at repos/mace-interlayer/mace/__init__.py
```

## 4. The interlayer integration (already committed on `split-mlip`)

Bilayer phonons need a stacked calculator (separate intralayer models per layer +
one interlayer model), which stock `mlip_phonon_scattering` did not support. The
`split-mlip` branch adds it **without** disturbing the monolayer path. Changes:

- **`mlip_phonon_scattering/calculator.py`**
  - `MACECalculatorConfig` gains `interlayer`, `intralayer_models`,
    `interlayer_models`, `layer_symbols`.
  - `build_nlayer_calculator(config, atoms)` builds intralayer `MACEWCalculator`s
    (one per layer) + interlayer `MACEWCalculator`s (one per adjacent pair) by
    slicing atoms on contiguous `atom_types` ranges, and wraps them in an
    `NLayerCalculator`. Imported lazily from `interlayer_helpers`.
  - `build_calculator(config, atoms=None)` dispatches: interlayer → NLayer, else
    the single-model path.
  - **Local single-model path uses `CompatMACECalculator`** — SEE GOTCHA 2.
  - `add_interlayer_arguments()` adds `--interlayer`, `--intralayer-models`,
    `--interlayer-model`, `--layer-symbols`.
- **`relax.py` / `forces.py`** call `build_calculator(config, atoms)`. `forces.py`
  builds the calculator once from the **first displaced supercell** (correct for
  both paths; the NLayer calc must bind to a structure of the right size — SEE GOTCHA 3).
- **`scripts/relax_structure.py` / `scripts/compute_mace_forces.py`** call
  `add_interlayer_arguments(parser)`.

The helpers themselves live in `repos/mace-interlayer/examples/interlayer_helpers/`
(`n_layer.py`, `macewrapper.py`, copied from the original non-git
`mace-interlayer-example` working dir). They are on `PYTHONPATH`. (`macewrapper.py`
has an optional `ittnotify` preload that no-ops on torch 2.11 — no stub binary is
needed; see the helper folder's README if you hit `undefined symbol: __itt_*` on a
different PyTorch build.)

### Gotcha 2 — old compiled models need `CompatMACECalculator`
The trained TMD models (`*_rmax2_stagetwo_compiled.model`) are older TorchScript
modules whose `forward` does **not** accept the newer kwargs
(`compute_edge_forces`, `compute_atomic_stresses`) that mace-torch 0.3.16's stock
`MACECalculator` passes — you get `RuntimeError: Unknown keyword argument
'compute_edge_forces'`. `macewrapper.CompatMACECalculator` catches this and retries
without those kwargs. The single-model (monolayer) path therefore uses
`CompatMACECalculator` too, falling back to stock `MACECalculator` only if the
helper can't be imported.

### Gotcha 3 — `MACEWCalculator` binds to its construction atoms
`MACEWCalculator` stores the `atom_types`/`layer_ids` of the atoms it is built
with and reuses them at `calculate()` time. So one `NLayerCalculator` is valid
only for structures of identical topology/size. All phonopy displaced supercells
share topology, so the calc is built **once** from the first supercell and reused.

### Gotcha 4 (verified OK, no fix needed) — supercell array alignment
`phonopy_io.copy_repeated_arrays` overlays `atom_types`/`layer_ids` onto phonopy
supercells with `np.repeat(values, repeats)`. This was verified to match phonopy's
diagonal-supercell atom ordering exactly (`np.repeat(numbers) == supercell.numbers`,
atom_types consistent with atomic numbers across all 54 atoms). No change required,
but re-check this if you ever use a **non-diagonal** supercell matrix.

## 4b. QE workflow modules (already on `split-mlip`)

The QE integration lives in `repos/mlip_phonon_scattering/mlip_phonon_scattering/`:

- **`qe_calculator.py`** — `QECalculatorConfig` dataclass; `write_qe_input()` (via ASE
  `write_espresso_in`); `run_pw_subprocess()` (launches `srun --exact --overlap pw.x`);
  `parse_qe_output()` (reads forces/energy from pw.x output via ASE); YAML config loader.
- **`qe_relax.py`** — `relax_structure_qe()`: writes pw.x relax input, runs it, extracts
  final geometry.
- **`qe_forces.py`** — `compute_displacement_forces_qe()`: writes all SCF inputs upfront,
  then runs up to `max_concurrent` pw.x jobs in parallel using
  `concurrent.futures.ProcessPoolExecutor`. Each worker calls `srun --exact --overlap`
  to claim its 4-GPU slice within the SLURM allocation. Restart-safe: skips any
  displacement whose output already contains `JOB DONE`.

The corresponding CLI scripts are `scripts/relax_structure_qe.py` and
`scripts/compute_qe_forces.py`, installed into the venv's `bin/` via `setup.py`.

**No libqe wrapper required.** The QE integration uses the standard NERSC module
`espresso/7.5-libxc-7.0.0-gpu` (`pw.x` binary) via subprocess. trajEXtories is
referenced for its `minimal_QE_example` pattern, not imported as a library.

### QE config (`qe_config.yaml`)

Each QE example directory has a `qe_config.yaml`. Key fields:

```yaml
pseudo_dir: /global/homes/j/jdgeorga/espresso/pseudo_new/  # ONCV PBE-1.2 pseudos
pseudo_files: {Mo: Mo_ONCV_PBE-1.2.upf, Se: Se_ONCV_PBE-1.2.upf}  # add W for bilayer
ecutwfc: 60.0       # Ry — safe for these ONCV pseudos
ecutrho: 480.0      # Ry
input_dft: vdw-df-c09
assume_isolated: 2D
relax_kpoints: [12, 12, 1]  # primitive cell
scf_kpoints: [1, 1, 1]      # Gamma-only for 4×4×1 or 3×3×1 supercell
nranks_per_disp: 4           # 4 GPUs per pw.x job
max_concurrent: 8            # 8 simultaneous jobs → 32 GPUs total
```

### SLURM configuration (debug queue)

Both QE `run.sh` scripts request:
```
#SBATCH --constraint=gpu  --qos=debug  --nodes=8  --ntasks=32
#SBATCH --ntasks-per-node=4  --gpus-per-node=4  --cpus-per-task=32  --time=00:30:00
```

The Python orchestrator runs directly on the first allocated node (not via `srun`);
it then launches nested `srun --exact --overlap` calls for each pw.x job.

## 5. Load script

`load_mace_phonon_env.sh` (already written) does, when sourced: `module load
pytorch/2.11.0`; `export PYTHONNOUSERSITE=1`; activate the venv; prepend
`PYTHONPATH` with `repos/mlip_phonon_scattering` and
`repos/mace-interlayer/examples/interlayer_helpers`; print a sanity import report.

## 6. Models

`models/` holds relative symlinks into `repos/mace-interlayer/examples/TMD_trained_models/`:
`MoSe2.model`, `WSe2.model`, `MoSe2_WSe2.model`.

## 7. Example inputs

- `examples/mose2_wse2_bilayer/MoSe2_WSe2_bilayer.xyz` — 6-atom relaxed bilayer with
  the required `atom_types` (0–2 = layer 0 Mo,Se,Se; 3–5 = layer 1 W,Se,Se) and
  `layer_ids` (0/1) arrays. Copied from the original example assets.
- `examples/mose2_monolayer/MoSe2_monolayer.xyz` — the 3 layer-0 atoms of the
  bilayer as a plain `species:pos` structure (the single-model path needs no
  `atom_types`/`layer_ids`).

---

## 8. Verification (how to know it works)

### MACE workflow

```bash
source load_mace_phonon_env.sh                       # sanity report: all imports OK
bash examples/mose2_monolayer/run.sh                 # ~33 s on CPU
bash examples/mose2_wse2_bilayer/run.sh              # ~39 s on CPU
```

### QE workflow

```bash
source load_mace_phonon_env.sh                       # still needed for Python env
sbatch examples/mose2_monolayer_qe/run.sh            # ~8 min on debug queue (8 nodes)
sbatch examples/mose2_wse2_bilayer_qe/run.sh         # ~10 min on debug queue (8 nodes)
```

Each produces `run/band_structure.png` plus `eigenvector_band.h5`. Physics checks
(read `frequencies_thz` from the band h5):

| System | Workflow | Acoustic @ Γ | A₁g | E₂g | Interlayer shear | Interlayer breathing |
|--------|----------|-------------|-----|-----|-----------------|---------------------|
| MoSe₂ monolayer | MACE | ~0 | 7.2 THz | 8.6 THz | — | — |
| MoSe₂ monolayer | QE (verified) | ~0 | **7.21 THz** | **8.43 THz** | — | — |
| MoSe₂/WSe₂ bilayer | MACE | ~0 | — | — | ~0.57 THz | ~0.85 THz |
| MoSe₂/WSe₂ bilayer | QE (verified) | ~0 | — | — | **0.58 THz** | **1.31 THz** |

The QE bilayer breathing mode (1.31 THz) is stiffer than MACE (0.85 THz) — a known
difference between the vdw-df-c09 functional and the trained interlayer potential.
The finite interlayer modes confirm that bilayer coupling is captured by DFT.

No imaginary modes in either workflow is the key sanity indicator.

## 9. Notes for a future agent

- **MACE**: run on a CPU interactive node (`--device cpu`). GPU works but is unnecessary.
- **QE**: always `sbatch` the `run.sh`; never `bash`. The script uses `srun --exact --overlap` internally and must be inside a SLURM allocation.
- The repos are the user's own. Commit locally; **do not `git push`** without asking.
- `mace-interlayer-example` (the original source of the helpers) is **not** a git
  repo. The helpers are vendored into `repos/mace-interlayer/examples/interlayer_helpers/`
  so this setup is self-contained; they can later be committed upstream.
- To scale up MACE accuracy: increase `--supercell` and `--mesh` in the example `run.sh`;
  parallelize with `srun -n <N> python ... compute_mace_forces.py` (auto-uses mpi4py).
- To scale up QE accuracy: increase `ecutwfc` (60→80 Ry), tighten `conv_thr` (1e-10),
  or use denser `relax_kpoints`. For larger supercells, increase `nranks_per_disp` and
  adjust `max_concurrent` accordingly (`total_GPUs / nranks_per_disp`).
- The QE displacement outputs in `run/qe_disps/` are kept after the job completes.
  Re-running `compute_qe_forces.py` will skip any displacement with `JOB DONE` in its
  output — safe to restart after a partial run.
