# MoS2 Example

Clean MoS2 workflow using `MoS2.xyz` by default.

```bash
bash examples/MoS2/run_mos2.sh
```

Outputs are written under `examples/MoS2/run/` unless `OUTPUT_PREFIX` is set.

The UED analysis step also writes `ued_intensity/temperature_dependent_bragg.csv`,
`ued_intensity/dw_factor_vs_temperature.png`, and
`ued_intensity/zero_phonon_intensity_vs_temperature.png`. These use the default
0 K to 1500 K sweep in 50 K steps at `G = (1,0,0)` and `G = (1,1,0)`.
The example skips the large tiled intensity CSV files by default; set
`UED_WRITE_TILED_CSV=1` to write them.
