# Examples

This folder contains the clean handoff examples. Older MoS2 and Si examples,
including generated run outputs, were moved to `../example_archive/`.

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
  `G = (1,0,0)` and `G = (1,1,0)`; Si uses `G = (2,2,0)` and `G = (3,1,0)`.
- `dw_factor_vs_temperature.png`: species-averaged Debye-Waller factors
  versus temperature for both Bragg vectors.
- `zero_phonon_intensity_vs_temperature.png`: two-subplot elastic
  zero-phonon intensity plot for the selected Bragg vectors.

The examples set `UED_WRITE_TILED_CSV=0`, so they do not write
`tiled_intensities_qz0.csv` or `tiled_intensities_qz0_long.csv` unless you
override with `UED_WRITE_TILED_CSV=1`.

## Linewidth and lifetime examples

[`mose2_wse2_bilayer`](mose2_wse2_bilayer/) is the MoSe2/WSe2 aligned-bilayer GPU pipeline. It uses the intralayer MoSe2 and WSe2 models plus the MoSe2/WSe2 interlayer model, and its self-contained validation checks the expected bilayer modes in addition to the general physics checks.

[`mose2_monolayer`](mose2_monolayer/) is the single-model MoSe2 GPU pipeline. It uses the 4x4x1 fc3 and 8x8x1 phonon supercells, and is physics-validated through its self-contained checks rather than against a golden reference.

Both pipelines HANG at `srun -n 16 --gpus-per-task=1` (16-rank mpi4py/PMI wireup deadlock, device-independent). Must run at `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`); `--overlap` is required so successive `srun` steps within one allocation don't deadlock on GPU-slice accounting.
