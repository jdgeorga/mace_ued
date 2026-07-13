# MLIP Phonon Scattering

Self-contained workflow for MACE-based phonons and Qz=0 UED intensities:

1. Relax a structure with ASE and MACE.
2. Generate phonopy displacements.
3. Compute MACE energies and forces.
4. Solve harmonic phonons and export `eigenvector.h5`.
5. Extract tiled zero- and one-phonon UED intensities.

The current UED default converts phonopy HDF5 eigenvectors to the PH.x-style
phase gauge with `--eigenvector-gauge phonopy_to_phx`.

This branch additionally provides a phonon-linewidth / lifetime workflow built on
a GPU-enabled phono3py fork; see [Phonon Lifetimes (phono3py)](#phonon-lifetimes-phono3py).

Both pipelines can drive their MACE forces from either a single foundation /
fine-tuned model or a stacked split-MLIP calculator; see
[Split MLIP architecture](#split-mlip-architecture) and
[Force generation](#force-generation).

## Setup

Building and activating the environment is documented separately from this
README:

- [`SETUP_MACE_PHONON.md`](SETUP_MACE_PHONON.md) — recreation guide for the
  harmonic environment on NERSC Perlmutter: MACE and the `mace-interlayer`
  fork, the Quantum ESPRESSO variant, model weights, and end-to-end
  verification. Build this first.
- [`SETUP_MACE_PHONO3PY.md`](SETUP_MACE_PHONO3PY.md) — builds the anharmonic
  lifetime stack (the `phono3py_einsum` GPU fork) on top of the harmonic
  environment, with its own end-to-end verification.
- [`THEORY_MACE_PHONON.md`](THEORY_MACE_PHONON.md) — what each harmonic
  pipeline stage does physically and mathematically, from the MLIP force
  engine to the phonon band structure.
- [`THEORY_MACE_PHONO3PY.md`](THEORY_MACE_PHONO3PY.md) — the physics behind
  the lifetime pipeline (three-phonon scattering, linewidths, lifetimes) and
  its invariants.

All commands below assume that environment is active.

## Split MLIP architecture

Layered van der Waals materials (e.g. a MoSe2/WSe2 bilayer) are described more
accurately by dedicated models than by a single foundation model. This package
supports a **split MLIP**: one intralayer MACE model fine-tuned per layer
chemistry, plus one interlayer MACE model per adjacent layer pair. The intralayer
models capture the in-plane bonding of each monolayer, and the interlayer models
capture the weaker cross-layer coupling that sets the shear/breathing modes.

The stacking is assembled by `build_nlayer_calculator()` in
`mlip_phonon_scattering/calculator.py`:

- The input structure must carry an `atom_types` array. Each layer is a
  contiguous range of `atom_types` values, sized by its `--layer-symbols` entry.
- For each layer, the atoms in its range are sliced out and wrapped in a
  per-layer `MACEWCalculator` loaded from that layer's intralayer model.
- For each adjacent layer pair, the union of the two layers' atoms is sliced out
  and wrapped in a `MACEWCalculator` loaded from the pair's interlayer model
  (`is_interlayer_calc=True`).
- All per-layer and per-pair calculators are combined into an `NLayerCalculator`,
  which sums their contributions into total energies and forces.

`MACECalculatorConfig` exposes this through `interlayer`, `intralayer_models`,
`interlayer_models`, and `layer_symbols`; `config_from_args()` builds it from the
`--interlayer / --intralayer-models / --interlayer-model / --layer-symbols` CLI
flags shared by every stage that computes forces.

**On this branch the interlayer helpers are vendored** at
`mlip_phonon_scattering/interlayer/` (`macewrapper.py`, `n_layer.py`), copied
2026-07-09 from `jornada-group/mace-interlayer` (branch
`port/mace-v0.3.16-interlayer`; see `interlayer/UPSTREAM.md`). What remains
external is:

- the **mace-interlayer fork** of `mace-torch` — it must be installed instead of
  upstream `mace-torch`, which is why `mace` is deliberately excluded from
  `install_requires` (see the note in `setup.py` and
  [`SETUP_MACE_PHONO3PY.md`](SETUP_MACE_PHONO3PY.md)); and
- the **trained model weights**, which are not committed. Examples resolve
  `$MODELS` (default `$MLIP_PHONON_ROOT/models`) and expect `MoSe2.model`,
  `WSe2.model`, and `MoSe2_WSe2.model`.

A split-model run is invoked the same way in both pipelines: pass `--interlayer`
together with `--intralayer-models`, `--interlayer-model` (repeatable, one per
pair), and `--layer-symbols` to the relax and force stages. See
[Force generation](#force-generation) for the harmonic/UED commands and
[Phonon Lifetimes (phono3py)](#phonon-lifetimes-phono3py) for the lifetime
commands. The MoSe2/WSe2 bilayer example
(`examples/mose2_wse2_bilayer/`) is the flagship split-model use.

Older compiled TorchScript models (both the single fine-tuned models and the
split intralayer/interlayer models) do not accept mace-torch 0.3.16's newer
`forward` kwargs. The single-model path loads them through the
`CompatMACECalculator` shim, and the split path uses the vendored
`MACEWCalculator`, both of which retry without those kwargs.

## Layout

- `mlip_phonon_scattering/`: Python package, including `linewidth/` (phono3py
  lifetime stages) and `interlayer/` (vendored NLayerCalculator support for
  split-MLIP bilayers).
- `scripts/`: command-line entry points for the phonon and UED workflow stages.
- `run_mlip_phonons.sh`: end-to-end phonon + UED driver.
- `examples/`: MoS2 and Si UED examples, plus MoSe2 monolayer and MoSe2/WSe2
  bilayer lifetime examples.
- `docs/`: implementation notes.
- `tests/`: focused unit tests.

See `docs/ued_temperature_outputs.md` for details on the temperature-dependent
UED Bragg figures.

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

The `run_mlip_phonons.sh` driver runs the single foundation / fine-tuned model
path. To drive the harmonic pipeline with a split MLIP, call the relax and force
stages directly with the interlayer flags shown in
[Force generation](#force-generation).

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

Compute forces: see [Force generation](#force-generation).

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

Forces for the phonopy displacements are computed with a MACE calculator via
`scripts/compute_mace_forces.py`. Two model configurations are supported and
share the same CLI flags: a single foundation / fine-tuned model, or a split
intralayer + interlayer stack. The relax stage (`scripts/relax_structure.py`)
accepts the same flags, so a structure is relaxed and its forces computed with
the same model configuration.

### Foundation or single fine-tuned model

By default the calculator uses MACE's `mace_mp` foundation model; select its
size/name with `--mace-model` (e.g. `medium`, `large`). To use one local
fine-tuned model instead, pass `--mace-model-path` (it overrides `--mace-model`);
older compiled TorchScript models are loaded through the `CompatMACECalculator`
shim automatically.

```bash
python scripts/compute_mace_forces.py \
  runs/example/relaxed.xyz \
  runs/example/relaxed_phonon_displacements.yaml \
  --forces-output runs/example/relaxed_forces.npy \
  --energies-output runs/example/relaxed_energies.npy \
  --mace-model medium
```

Single local fine-tuned model:

```bash
python scripts/compute_mace_forces.py \
  runs/example/relaxed.xyz \
  runs/example/relaxed_phonon_displacements.yaml \
  --forces-output runs/example/relaxed_forces.npy \
  --energies-output runs/example/relaxed_energies.npy \
  --mace-model-path "$MODELS/MoSe2.model" \
  --default-dtype float64
```

### Split intralayer + interlayer models

Pass `--interlayer` with one intralayer model per layer, one interlayer model per
adjacent layer pair (`--interlayer-model` is repeatable), and `--layer-symbols`
giving each layer's chemical symbols as a nested Python list. This requires the
`mace-interlayer` fork to be installed (see
[Split MLIP architecture](#split-mlip-architecture)), and the input structure
must carry the `atom_types` array used to slice each layer.

```bash
python scripts/compute_mace_forces.py \
  runs/example/relaxed.xyz \
  runs/example/relaxed_phonon_displacements.yaml \
  --forces-output runs/example/relaxed_forces.npy \
  --energies-output runs/example/relaxed_energies.npy \
  --interlayer \
  --intralayer-models "$MODELS/MoSe2.model" "$MODELS/WSe2.model" \
  --interlayer-model "$MODELS/MoSe2_WSe2.model" \
  --layer-symbols "[['Mo','Se','Se'],['W','Se','Se']]" \
  --device cuda \
  --default-dtype float64
```

The phono3py lifetime pipeline computes its 2nd- and 3rd-order forces with the
same two model configurations. The `mlip-linewidth-forces2`,
`mlip-linewidth-forces3`, and `mlip-linewidth-forces2-from3` console scripts take
the identical `--mace-model-path` (single) or
`--interlayer / --intralayer-models / --interlayer-model / --layer-symbols`
(split) flags; the monolayer example uses the single path and the bilayer example
uses the split path. See
[Phonon Lifetimes (phono3py)](#phonon-lifetimes-phono3py) and the example
`run_linewidth.sh` scripts for full invocations.

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
the Si example (primitive cell) overrides these to `G = (1,1,1)` (allowed
reflection) and `G = (1,1,0)` (systematic absence).
`dw_factor_vs_temperature.png` plots species-averaged Debye-Waller factors for
the requested Bragg vectors, and
`zero_phonon_intensity_vs_temperature.png` plots the corresponding elastic
zero-phonon intensities as two subplots.

The MoS2 and Si example wrappers set `UED_WRITE_TILED_CSV=0` because the tiled
wide and long CSVs are much larger than the figures and dominate UED extraction
time and memory. Set `UED_WRITE_TILED_CSV=1` if those tabular Q-point exports are
needed.

## Phonon Lifetimes (phono3py)

This branch adds a phonon-linewidth / lifetime workflow built on a GPU-enabled
phono3py fork (`phono3py_einsum`, which provides the `lang="GPU"` path). The
stages live in `mlip_phonon_scattering/linewidth/` and are installed as
`mlip-linewidth-*` console scripts by `pip install -e .` (relax, phonopy /
phono3py displacements, 2nd- and 3rd-order forces, force-constant cache, GPU
scattering, gamma extraction, plotting, and validation).

Install `phono3py_einsum` as a package, or point `PHONO3PY_EINSUM_PATH` at a
local checkout; both are covered in [Setup](#setup).

The relax and force stages take the same two model configurations as the
harmonic pipeline (see [Force generation](#force-generation)): a single
fine-tuned model via `--mace-model-path`, or a split stack via
`--interlayer --intralayer-models ... --interlayer-model ... --layer-symbols ...`.

Two self-contained GPU examples exercise the pipeline end to end:

- `examples/mose2_monolayer/`: single-model MoSe2 monolayer (single fine-tuned
  `MoSe2.model`), physics-validated by its own checks (no golden reference).
- `examples/mose2_wse2_bilayer/`: MoSe2/WSe2 aligned bilayer using the split
  stack (intralayer `MoSe2.model` + `WSe2.model` plus interlayer
  `MoSe2_WSe2.model`), validated against a golden reference. This is the flagship
  split-model run.

Both run inside a Perlmutter GPU allocation via `bash run_salloc_pipeline.sh`.
They must run at `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`): a
16-rank launch deadlocks in mpi4py/PMI wireup, and `--overlap` keeps successive
`srun` steps within one allocation from deadlocking on GPU-slice accounting. See
each example's README for validated numbers.

## Verification

```bash
python -m pytest -q tests
python -m compileall -q mlip_phonon_scattering scripts tests
```
</content>
</invoke>
