# Examples

This folder contains the clean MoS2 and Si handoff examples. Both use the single
MACE foundation-model force path; for bilayer split intralayer/interlayer (MACE
`NLayerCalculator`) runs see the "Split MLIP architecture" and "Force generation"
sections of the top-level `../README.md`.

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
  `G = (1,0,0)` and `G = (1,1,0)`; Si uses primitive-cell `G = (1,1,1)` and
  `G = (1,1,0)`.
- `dw_factor_vs_temperature.png`: species-averaged Debye-Waller factors
  versus temperature for both Bragg vectors.
- `zero_phonon_intensity_vs_temperature.png`: two-subplot elastic
  zero-phonon intensity plot for the selected Bragg vectors.

The examples set `UED_WRITE_TILED_CSV=0`, so they do not write
`tiled_intensities_qz0.csv` or `tiled_intensities_qz0_long.csv` unless you
override with `UED_WRITE_TILED_CSV=1`.
