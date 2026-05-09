# UED Temperature-Dependent Outputs

The UED extraction stage writes temperature-dependent Bragg diagnostics in
addition to the Qz=0 tiled intensity maps. The default sweep is 0 K to 1500 K in
50 K steps and can be overridden with either:

```bash
UED_TEMPERATURE_SWEEP_K="$(seq -s ' ' 0 50 1500)"
UED_TEMPERATURE_TARGET_G="1 0 0;1 1 0"
```

for `run_mlip_phonons.sh` and the example wrappers, or:

```bash
python scripts/extract_uq_intensities.py \
  --input-h5 runs/example/eigenvector.h5 \
  --output-dir runs/example/ued_intensity \
  --temperature-sweep-k $(seq -s ' ' 0 50 1500) \
  --temperature-target-g 1 0 0 \
  --temperature-target-g 1 1 0 \
  --no-tiled-csv
```

for the standalone extraction step.

The calculation reuses the phonon mesh, eigenvector gauge, phonon-energy cutoff,
and electron scattering-factor model used for the main UED intensity extraction.
For each temperature, it recomputes the Bose occupations, mode amplitudes,
Debye-Waller tensor, Debye-Waller factors, and elastic zero-phonon intensities at
the requested reciprocal-lattice vectors. The common defaults are:

- `G = (1,0,0)`
- `G = (1,1,0)`

The Si example overrides the targets to:

- `G = (2,2,0)`
- `G = (3,1,0)`

The output files are:

- `temperature_dependent_bragg.csv`: one row per temperature. It contains the
  Cartesian reciprocal coordinates, zero-phonon intensity, and per-atom
  Debye-Waller factor for both Bragg vectors.
- `dw_factor_vs_temperature.png`: species-averaged Debye-Waller factors versus
  temperature for the selected Bragg vectors.
- `zero_phonon_intensity_vs_temperature.png`: a two-subplot figure with the
  elastic zero-phonon intensity versus temperature for the selected Bragg
  vectors.

Pass `--temperature-sweep-k` with no values to skip these temperature-dependent
outputs when running `scripts/extract_uq_intensities.py` directly.

The tiled Q-point CSVs, `tiled_intensities_qz0.csv` and
`tiled_intensities_qz0_long.csv`, are independent of these temperature
diagnostics and can be skipped with `--no-tiled-csv`. The MoS2 and Si example
wrappers set `UED_WRITE_TILED_CSV=0` by default because those CSVs are large and
dominate extraction runtime and memory.
