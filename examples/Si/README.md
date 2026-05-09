# Si Example

Clean Si workflow using `Si.xyz` by default.

```bash
bash examples/Si/run_si.sh
```

Outputs are written under `examples/Si/run/` unless `OUTPUT_PREFIX` is set.
This example uses `UED_GMAX=4` by default for one additional reciprocal-space
G shell relative to the common driver default.

The UED analysis step also writes `ued_intensity/temperature_dependent_bragg.csv`,
`ued_intensity/dw_factor_vs_temperature.png`, and
`ued_intensity/zero_phonon_intensity_vs_temperature.png`. These use the default
0 K to 1500 K sweep in 50 K steps at `G = (2,2,0)` and `G = (3,1,0)`.
The example skips the large tiled intensity CSV files by default; set
`UED_WRITE_TILED_CSV=1` to write them.
