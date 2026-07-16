# Examples

The MoS2 and Si examples run the MACE phonon + Qz=0 UED pipeline; the MoSe2
monolayer and MoSe2/WSe2 bilayer examples run the phono3py linewidth / lifetime
pipeline (see [Linewidth and lifetime examples](#linewidth-and-lifetime-examples)
below).

Defaults inherited from `../run_mlip_phonons.sh`:

- supercell: `6 6 1`
- phonon mesh: `36 36 1`
- relaxation threshold: `FMAX=1e-5`
- relaxation steps: `STEPS=500`
- UED temperature sweep: `0:50:1500 K`
- tiled UED CSV output: disabled by default to keep examples fast and memory-light
- UED eigenvector gauge: `phonopy_to_phx`
- electron scattering model: `peng`

Run from the repository root:

```bash
bash examples/MoS2/run_mos2.sh
bash examples/Si/run_si.sh
```

Set `OUTPUT_PREFIX=/path/to/run/name` to write outputs somewhere else.

Each example writes temperature-dependent UED diagnostics under
`run/ued_intensity/`:

- `temperature_dependent_bragg.csv`: per-atom Debye-Waller factors and
  zero-phonon intensities from 0 K to 1500 K in 50 K steps. MoS2 uses
  `G = (1,0,0)` and `G = (1,1,0)`; Si uses `G = (1,1,1)` and `G = (1,1,0)`.
- `dw_factor_vs_temperature.png`: species-averaged Debye-Waller factors
  versus temperature for both Bragg vectors.
- `zero_phonon_intensity_vs_temperature.png`: two-subplot elastic
  zero-phonon intensity plot for the selected Bragg vectors.

The examples set `UED_WRITE_TILED_CSV=0`, so they do not write
`tiled_intensities_qz0.csv` or `tiled_intensities_qz0_long.csv` unless you
override with `UED_WRITE_TILED_CSV=1`.

## Linewidth and lifetime examples

These two examples run the phono3py linewidth / lifetime pipeline and contrast
the two supported MACE model configurations (see the top-level README's
[Split MLIP architecture](../README.md#split-mlip-architecture) and
[Force generation](../README.md#force-generation) sections). Both resolve trained
weights from `$MODELS` (default `$MLIP_PHONON_ROOT/models`), which are not
committed to the repo.

[`mose2_wse2_bilayer`](mose2_wse2_bilayer/) is the flagship **split-MLIP** run:
the MoSe2/WSe2 aligned bilayer stacked with `--interlayer` from the intralayer
`MoSe2.model` and `WSe2.model` plus the interlayer `MoSe2_WSe2.model`
(`--layer-symbols "[['Mo','Se','Se'],['W','Se','Se']]"`). Its self-contained
validation checks the expected bilayer shear/breathing modes in addition to the
general physics checks, against a golden reference.

[`mose2_monolayer`](mose2_monolayer/) is the **single fine-tuned model** MoSe2
GPU pipeline (one `MoSe2.model` via `--mace-model-path`). It uses the 4x4x1 fc3
and 8x8x1 phonon supercells, and is physics-validated through its self-contained
checks rather than against a golden reference.

Both pipelines HANG at `srun -n 16 --gpus-per-task=1` (16-rank mpi4py/PMI wireup deadlock, device-independent). Must run at `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`); `--overlap` is required so successive `srun` steps within one allocation don't deadlock on GPU-slice accounting.
