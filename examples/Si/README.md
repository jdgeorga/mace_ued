# Si Example

Si workflow using `Si.xyz` by default.

```bash
bash examples/Si/run_si.sh
```

Outputs are written under `examples/Si/run/` unless `OUTPUT_PREFIX` is set.
This example uses `UED_GMAX=3`, interpreted as a radial reciprocal-space cutoff.

The UED analysis step also writes `ued_intensity/temperature_dependent_bragg.csv`,
`ued_intensity/dw_factor_vs_temperature.png`, and
`ued_intensity/zero_phonon_intensity_vs_temperature.png`. These use the default
0 K to 1500 K sweep in 50 K steps at primitive reciprocal-space `G = (1,1,1)`
(allowed reflection) and `G = (1,1,0)` (systematic absence).
The example skips the large tiled intensity CSV files by default; set
`UED_WRITE_TILED_CSV=1` to write them.
