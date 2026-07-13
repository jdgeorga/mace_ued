# MLIP Phonon Scattering

Self-contained workflow for MACE-based phonons and Qz=0 UED intensities:

1. Relax a structure with ASE and MACE.
2. Generate phonopy displacements.
3. Compute MACE energies and forces.
4. Solve harmonic phonons and export `eigenvector.h5`.
5. Extract tiled zero- and one-phonon UED intensities.

The current UED default converts phonopy HDF5 eigenvectors to the PH.x-style
phase gauge with `--eigenvector-gauge phonopy_to_phx`.

## Layout

- `mlip_phonon_scattering/`: Python package.
- `scripts/`: command-line entry points for each workflow stage.
- `run_mlip_phonons.sh`: end-to-end driver.
- `examples/`: clean MoS2 and Si handoff examples.
- `docs/`: implementation notes.
- `tests/`: focused unit tests.

The stage scripts default to the MACE foundation model. Quantum ESPRESSO
variants (`scripts/relax_structure_qe.py`, `scripts/compute_qe_forces.py`,
backed by the `qe_*` modules) provide DFT relaxation and forces. The MACE relax
and force stages also accept `--interlayer` with one or more `--interlayer-model`
paths to build a stacked `NLayerCalculator` for bilayer/interlayer phonons (this
requires the `mace-interlayer` fork; see Installation).

See `docs/ued_temperature_outputs.md` for details on the temperature-dependent
UED Bragg figures, and `docs/phonopy_eigenvector_gauge.md` for the eigenvector
gauge convention used by the UED default.

## Installation

Install the package editable from the repository root:

```bash
pip install -e .
```

This installs the Python dependencies (numpy, h5py, matplotlib, ase, phonopy)
and the stage scripts as command-line tools. It intentionally does **not**
install MACE: install `mace-torch` for single foundation-model runs, or install
the `mace-interlayer` fork *before* this package for bilayer/interlayer runs.

## Environment

On NERSC, the existing project environment is:

```bash
source /pscratch/sd/j/jdgeorga/twist-anything/phonon_unfolding/scratch/phonon_diff/phonon_2.6_env/bin/activate
```

That environment is only a convenience. On another machine, use Python 3.10 or
newer, create a virtual environment, and `pip install -e .` (see Installation)
to pull in the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Then install MACE separately. For a CPU-only setup, install a CPU PyTorch build
before `mace-torch`. For a CUDA setup, install the PyTorch build that matches
your local CUDA driver, then install `mace-torch`.

The driver defaults to the NERSC project interpreter. If you are using your own
environment, either activate it and call the stage scripts directly with
`python`, or set `PYTHON` when running the common driver:

```bash
PYTHON="$(which python)" bash run_mlip_phonons.sh /path/to/input_structure.xyz /path/to/output_prefix
```

The repository root must be on `PYTHONPATH` when running scripts from outside
this directory (an editable `pip install -e .` handles this automatically):

```bash
export PYTHONPATH=/path/to/mlip_phonon_scattering:${PYTHONPATH:-}
```

If importing `torch` fails with an `iJIT_NotifyEvent` symbol error on NERSC,
preload the local ITT stub before running MACE stages:

```bash
source env/preload_ittnotify_stub.sh
```

The helper resolves the default stub relative to this repository as
`ittnotify_stub/libittnotify.so`, so the checkout is portable. If the stub
lives somewhere else, set `MLIP_ITTNOTIFY_STUB=/path/to/libittnotify.so`
before sourcing the helper.

## Quick Start

From this directory:

```bash
bash examples/MoS2/run_mos2.sh
bash examples/Si/run_si.sh
```

Both examples inherit these defaults from `run_mlip_phonons.sh`:

- supercell: `6 6 1`
- phonon mesh: `36 36 1`
- relaxation threshold: `FMAX=1e-5`
- relaxation steps: `STEPS=500`
- UED temperature: `100 K`
- UED temperature sweep: `0:50:1500 K`
- UED temperature Bragg targets: `G = (1,0,0)` and `G = (1,1,0)`; Si (primitive cell) overrides to `G = (1,1,1)` and `G = (1,1,0)`
- UED phonon cutoff: `0.2 meV`
- UED reciprocal tiling: radial `|Q| <= GMAX * |G_primitive|min`, with `GMAX=3`
- tiled UED CSV output: disabled by the MoS2 and Si example wrappers to reduce runtime and memory
- electron scattering model: `peng`
- eigenvector gauge: `phonopy_to_phx`

Set environment variables to override defaults, for example:

```bash
DEVICE=cuda OUTPUT_PREFIX=/path/to/run/MoS2_mace_relaxed bash examples/MoS2/run_mos2.sh
```

## Common Driver

```bash
bash run_mlip_phonons.sh /path/to/input_structure.xyz /path/to/output_prefix
```

Important overrides:

```bash
MACE_MODEL=medium
MACE_MODEL_PATH=
DEVICE=cpu
DEFAULT_DTYPE=float32
DISPERSION=0
SUPER_X=6
SUPER_Y=6
SUPER_Z=1
MESH_X=36
MESH_Y=36
MESH_Z=1
BAND_PATH=
QPOINTS_FILE=
FMAX=1e-5
STEPS=500
MAXSTEP=0.05
MPI_RANKS=1
UED_TEMPERATURE_K=100
UED_TEMPERATURE_SWEEP_K="$(seq -s ' ' 0 50 1500)"
UED_TEMPERATURE_TARGET_G="1 0 0;1 1 0"
UED_PHWMIN_MEV=0.2
UED_GMAX=3
UED_QZ=0
UED_QZ_TOL=1e-8
UED_WRITE_TILED_CSV=1
UED_ELECTRON_SCATTERING_MODEL=peng
UED_EIGENVECTOR_GAUGE=phonopy_to_phx
```

## Stage Commands

Relax:

```bash
python scripts/relax_structure.py input.xyz runs/example/relaxed \
  --mace-model medium \
  --device cpu \
  --fmax 1e-5 \
  --steps 500
```

Generate displacements:

```bash
python scripts/generate_displacements.py runs/example/relaxed.xyz \
  --supercell 6 6 1 \
  --output runs/example/relaxed_phonon_displacements.yaml
```

Compute forces:

```bash
python scripts/compute_mace_forces.py \
  runs/example/relaxed.xyz \
  runs/example/relaxed_phonon_displacements.yaml \
  --forces-output runs/example/relaxed_forces.npy \
  --energies-output runs/example/relaxed_energies.npy \
  --mace-model medium
```

Solve phonons:

```bash
python scripts/solve_phonons.py \
  --input-xyz runs/example/relaxed.xyz \
  --forces runs/example/relaxed_forces.npy \
  --phonopy-yaml runs/example/relaxed_phonon_displacements.yaml \
  --mesh 36 36 1 \
  --output-dir runs/example
```

Extract Qz=0 UED intensities:

```bash
python scripts/extract_uq_intensities.py \
  --input-h5 runs/example/eigenvector.h5 \
  --output-dir runs/example/ued_intensity \
  --qz 0 \
  --temperature-sweep-k $(seq -s ' ' 0 50 1500) \
  --temperature-target-g 1 0 0 \
  --temperature-target-g 1 1 0 \
  --no-tiled-csv \
  --eigenvector-gauge phonopy_to_phx \
  --electron-scattering-model peng
```

## Outputs

For an output prefix like `runs/example/relaxed`, the driver writes:

- `runs/example/relaxed.xyz`
- `runs/example/relaxed.traj`
- `runs/example/relaxed.traj.xyz`
- `runs/example/relaxed_phonon_displacements.yaml`
- `runs/example/relaxed_forces.npy`
- `runs/example/relaxed_energies.npy`
- `runs/example/band_structure.png`
- `runs/example/eigenvector.h5`
- `runs/example/eigenvector_band.h5`
- `runs/example/ued_intensity/tiled_intensities_qz0.csv`, only when `UED_WRITE_TILED_CSV=1`
- `runs/example/ued_intensity/tiled_intensities_qz0_long.csv`, only when `UED_WRITE_TILED_CSV=1`
- `runs/example/ued_intensity/temperature_dependent_bragg.csv`
- `runs/example/ued_intensity/dw_factor_vs_temperature.png`
- `runs/example/ued_intensity/zero_phonon_intensity_vs_temperature.png`
- `runs/example/ued_intensity/*.png`

The temperature-dependent outputs use the same phonon mesh and Debye-Waller
calculation as the Qz=0 UED maps. `temperature_dependent_bragg.csv` stores the
per-atom Debye-Waller factors and zero-phonon intensities at each requested
temperature target. The common defaults are `G = (1,0,0)` and `G = (1,1,0)`;
the Si example overrides these to primitive-cell `G = (1,1,1)` and `G = (1,1,0)`.
`dw_factor_vs_temperature.png` plots species-averaged Debye-Waller factors for
the requested Bragg vectors, and
`zero_phonon_intensity_vs_temperature.png` plots the corresponding elastic
zero-phonon intensities as two subplots.

The MoS2 and Si example wrappers set `UED_WRITE_TILED_CSV=0` because the tiled
wide and long CSVs are much larger than the figures and dominate UED extraction
time and memory. Set `UED_WRITE_TILED_CSV=1` if those tabular Q-point exports are
needed.

## Verification

```bash
python -m pytest -q tests
python -m compileall -q mlip_phonon_scattering scripts tests
```
