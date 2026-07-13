# MLIP Phonon Scattering

Self-contained workflow for MACE-based phonons and Qz=0 UED intensities:

1. Relax a structure with ASE and MACE.
2. Generate phonopy displacements.
3. Compute MACE energies and forces.
4. Solve harmonic phonons and export `eigenvector.h5`.
5. Extract tiled zero- and one-phonon UED intensities.

The current UED default converts phonopy HDF5 eigenvectors to the PH.x-style
phase gauge with `--eigenvector-gauge phonopy_to_phx`.

The MACE force engine (steps 1 and 3) supports two model configurations: a single
foundation or fine-tuned model applied to the whole structure, or a **split MLIP**
that describes a bilayer with per-layer fine-tuned intralayer models plus a
fine-tuned interlayer model, stacked by an `NLayerCalculator`. See
[Split MLIP architecture](#split-mlip-architecture) and
[Force generation](#force-generation).

## Setup

Building and activating the environment is documented separately from this
README:

- [`SETUP_MACE_PHONON.md`](SETUP_MACE_PHONON.md) — recreation guide for the
  environment on NERSC Perlmutter: MACE and the `mace-interlayer` fork, the
  Quantum ESPRESSO variant, model weights, and end-to-end verification.
- [`THEORY_MACE_PHONON.md`](THEORY_MACE_PHONON.md) — what each pipeline stage
  does physically and mathematically, from the MLIP force engine to the
  harmonic phonon band structure.

All commands below assume that environment is active.

## Split MLIP architecture

For a van der Waals bilayer, a single foundation model rarely captures both the
stiff intralayer bonding and the soft interlayer coupling with the accuracy phonon
calculations need. This branch instead composes several fine-tuned MACE models:

- **Per-layer intralayer models.** Each layer gets its own fine-tuned MACE model
  (e.g. `MoSe2.model`, `WSe2.model`) that describes only the in-plane bonding of
  that layer.
- **Adjacent-pair interlayer models.** Each adjacent layer pair gets a fine-tuned
  interlayer MACE model (e.g. `MoSe2_WSe2.model`) that describes the coupling
  between the two layers.
- **`NLayerCalculator` stacking.** `mlip_phonon_scattering.calculator.build_nlayer_calculator()`
  builds one `MACEWCalculator` per layer and one per adjacent layer pair, then wraps
  them in an `NLayerCalculator` whose total energy and forces are the sum of the
  per-layer intralayer contributions and the interlayer-pair contributions.

Layers are identified by the `atom_types` array carried on the input structure.
Each layer occupies a contiguous range of `atom_types` values sized by its
`--layer-symbols` entry: the first `len(symbols[0])` types form layer 0, the next
`len(symbols[1])` types form layer 1, and so on. Intralayer calculators are sliced
from the atoms in a single layer's range; each interlayer calculator is sliced from
the union of an adjacent pair's ranges. Inputs therefore must carry `atom_types`
(and `layer_ids`); a foundation-model input does not need them.

`build_calculator()` dispatches on the `--interlayer` flag: without it, the single
foundation/fine-tuned path (`build_mace_foundation_calculator()`) is used; with it,
the split path (`build_nlayer_calculator()`) is used and requires the `atoms` to
build against. The split path validates that there is exactly one intralayer model
per layer and one interlayer model per adjacent pair (one fewer than the number of
layers).

### Requirements

- **`mace-interlayer` fork.** The split path builds `MACEWCalculator` /
  `NLayerCalculator` from the interlayer helper modules `macewrapper` and `n_layer`,
  which are **not** vendored in this repository on this branch. They are imported
  from `PYTHONPATH`, expected at `repos/mace-interlayer/examples/interlayer_helpers`
  (see [Setup](#setup)). The `mace` package itself must be the
  forked `mace-interlayer` build (which provides `is_interlayer_calc` / `layer_ids`
  support), installed instead of upstream `mace-torch` — this is why `mace` is
  deliberately excluded from `install_requires` (see `setup.py` and Installation).
- **Model weights.** The fine-tuned model files are not stored in the repository.
  Examples expect them under `$MODELS` (default `$MLIP_PHONON_ROOT/models`),
  e.g. `MoSe2.model`, `WSe2.model`, and `MoSe2_WSe2.model`.

### Invoking a split run

The split path is driven through the stage scripts directly (the `run_mlip_phonons.sh`
driver only passes single-model flags). Both `scripts/relax_structure.py` and
`scripts/compute_mace_forces.py` accept the interlayer flags; the relax step for a
MoSe2/WSe2 bilayer looks like:

```bash
python scripts/relax_structure.py bilayer.xyz runs/bilayer/relaxed \
  --interlayer \
  --intralayer-models "$MODELS/MoSe2.model" "$MODELS/WSe2.model" \
  --interlayer-model "$MODELS/MoSe2_WSe2.model" \
  --layer-symbols "[['Mo','Se','Se'],['W','Se','Se']]" \
  --device cpu \
  --fmax 1e-5 \
  --steps 500
```

The matching force command is shown under
[Force generation → Split intralayer + interlayer models](#split-intralayer--interlayer-models).

## Layout

- `mlip_phonon_scattering/`: Python package.
- `scripts/`: command-line entry points for each workflow stage.
- `run_mlip_phonons.sh`: end-to-end driver (single-model MACE path).
- `examples/`: clean MoS2 and Si handoff examples.
- `docs/`: implementation notes.
- `tests/`: focused unit tests.

The stage scripts default to the MACE foundation model. Quantum ESPRESSO variants
(`scripts/relax_structure_qe.py`, `scripts/compute_qe_forces.py`, backed by the
`qe_*` modules) provide DFT relaxation and forces as an alternative to the MACE
relax and force stages. The split-MLIP interlayer path (`--interlayer`, see
[Split MLIP architecture](#split-mlip-architecture)) requires the `mace-interlayer`
fork; see Installation.

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
the `mace-interlayer` fork *before* this package for split intralayer/interlayer
runs (upstream `mace-torch` lacks the interlayer support the split path needs).

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

The driver runs the single-model MACE path (foundation or one local model). Split
intralayer/interlayer runs use the stage scripts directly (see
[Force generation](#force-generation)). Important overrides:

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

Forces for the phonopy displacements are their own stage; see
[Force generation](#force-generation).

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

## Force generation

Forces for the phonopy displacement supercells are computed with a MACE calculator
by `scripts/compute_mace_forces.py`, which writes the `forces.npy` and
`energies.npy` arrays that `solve_phonons.py` consumes. Two model configurations are
supported: a single foundation/fine-tuned model, or a split intralayer + interlayer
model stack.

### Foundation or single fine-tuned model

By default the force engine uses the MACE `mace_mp` foundation model selected by
`--mace-model` (a size such as `small`/`medium`/`large` or a named model). Pass
`--mace-model-path` to load a single local fine-tuned model instead; local compiled
TorchScript models are loaded through the `CompatMACECalculator` shim (which retries
without the newer forward kwargs that older models do not accept, falling back to the
stock `MACECalculator`).

```bash
python scripts/compute_mace_forces.py \
  runs/example/relaxed.xyz \
  runs/example/relaxed_phonon_displacements.yaml \
  --forces-output runs/example/relaxed_forces.npy \
  --energies-output runs/example/relaxed_energies.npy \
  --mace-model medium \
  --device cpu
```

Swap `--mace-model medium` for `--mace-model-path /path/to/model.model` to use a
single local fine-tuned model.

### Split intralayer + interlayer models

Add `--interlayer` to build the stacked `NLayerCalculator` from per-layer intralayer
models and adjacent-pair interlayer models (see
[Split MLIP architecture](#split-mlip-architecture)). The flags are:

- `--intralayer-models`: one intralayer model path per layer (space-separated).
- `--interlayer-model`: one interlayer model path per adjacent layer pair
  (repeat the flag once per pair).
- `--layer-symbols`: the per-layer chemical symbols as a Python literal, e.g.
  `"[['Mo','Se','Se'],['W','Se','Se']]"`. This defines each layer's `atom_types`
  range, so the input structure must carry the matching `atom_types` (and
  `layer_ids`) arrays.

This path requires the `mace-interlayer` fork and its `macewrapper` / `n_layer`
helpers on `PYTHONPATH`, and the model files (default under `$MODELS`). For a
MoSe2/WSe2 bilayer:

```bash
python scripts/compute_mace_forces.py \
  runs/bilayer/relaxed.xyz \
  runs/bilayer/relaxed_phonon_displacements.yaml \
  --forces-output runs/bilayer/relaxed_forces.npy \
  --energies-output runs/bilayer/relaxed_energies.npy \
  --interlayer \
  --intralayer-models "$MODELS/MoSe2.model" "$MODELS/WSe2.model" \
  --interlayer-model "$MODELS/MoSe2_WSe2.model" \
  --layer-symbols "[['Mo','Se','Se'],['W','Se','Se']]" \
  --device cpu
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
