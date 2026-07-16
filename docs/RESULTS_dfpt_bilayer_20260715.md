# DFPT-harmonic + split-MLIP-anharmonic phonon linewidths — MoSe₂/WSe₂ bilayer

**Date:** 2026-07-15 · **System:** MoSe₂/WSe₂ 6-atom aligned bilayer (split-MLIP MACE) ·
**Mesh:** 36×36×1 · **T:** 50 K · **DFPT source:**
`tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix` (6×6×1, loto_2d, SOC, DFT-D3)

## What this is
Phonon linewidths γ(q,ν)/lifetimes τ from a **hybrid**: harmonic phonons (freqs+eigenvectors)
from the completed **DFPT** calc (2D LO-TO), anharmonic FC3 from **split-MLIP (MACE)** forces on
phono3py displacements at the DFPT geometry. DFPT modes are injected into phono3py via
`set_phonon_data`; the MLIP real-space FC3 is contracted against the DFPT eigenvectors.

## Variant matrix
| Variant | Geometry | Harmonic FC2 (freqs+eigvecs) | FC3 |
|---|---|---|---|
| V1 `mlip_strict` | MLIP-relaxed | MLIP | MLIP |
| V2 `dfpt_harm` | DFPT | **DFPT** (loto/noloto) | MLIP @ DFPT geom |
| V3 `mlip_at_dfpt` | DFPT | MLIP | MLIP @ DFPT geom |
(each DFPT-geometry variant × residual-force-subtraction {on, off}; V2 × loto/noloto.)

## Headline result — M/K top-optical (the fair metric), residual-on
| Variant | K: γ(THz) / τ(ps) | M: γ / τ |
|---|---|---|
| V1 mlip_strict | 0.00762 / 10.4 | 0.00842 / 9.5 |
| V2 dfpt_harm (loto) | 0.03129 / 2.5 | 0.02269 / 3.5 |
| V3 mlip_at_dfpt | 0.01066 / 7.5 | 0.00779 / 10.2 |

**Ablation (M/K):**
- **FC2-source effect** (V2/V3, identical geometry + MLIP FC3): DFPT eigenvectors give
  **γ ~2.9× larger** (K 2.94×, M 2.91×) → ~3× shorter M/K optical lifetimes than MLIP.
  Same direction as the ued-project stage6/9 finding that a DFPT eigenbasis inflates ph-ph γ
  (there ~10–30×; here ~3× at M/K).
- **Geometry effect** (V1/V3): small (γ ratio 0.7–1.1×) — DFPT-vs-MLIP geometry barely moves M/K γ.
- **Conclusion:** the *harmonic eigenvector source* (DFPT vs MLIP), not the geometry, is the
  dominant lever on the M/K optical linewidths.

## Finding: DFPT near-Γ flexural (ZA) instability (absent in MLIP)
The DFPT harmonic has genuine imaginary near-Γ flexural modes (loto min −0.67 THz over 38/3999 q;
noloto −0.98 over 165), **OMP-independent (confirmed real, not an artifact)**, that the MLIP
harmonic lacks (MLIP fc2 min ≈ −5e-8 THz). These were **floored to +0.02 THz** so the scatter runs.
Consequence: V2 γ_max (0.33 THz) is dominated by near-Γ hotspots from the floored soft modes — so
**γ_max is not a fair comparison; use M/K top-optical**. Near-Γ γ for V2 is flagged unreliable.

## Reproducibility notes (important)
- **matdyn 2D-LOTO requires `OMP_NUM_THREADS=1`.** `PHonon/PH/rigid.f90`'s `rgd_blk` 2D-LOTO
  G-sum OMP loops have a race that corrupts the Γ non-analytic term with >1 thread (gate: 8.9 cm⁻¹
  @ OMP>1 → 0.000000 cm⁻¹ @ OMP=1). The reference `finish_collect.sbatch` used OMP=1.
- **Binary:** `q-e-epw-tdbe-speedup/bin/matdyn.x` + cray modules (`cray-fftw cray-hdf5-parallel`);
  dispersion gate vs `collect/phband.freq` = bit-exact at OMP=1.
- **Gauge:** QE→phono3py eigenvector Bloch gauge sign = **−1** (validated vs
  `docs/phonopy_eigenvector_gauge.md`); the overlap-residual metric is unreliable for this
  NAC-sensitive heterobilayer (documented) but the discrete sign is correct.
- **DFPT fc2 (dfpt_read):** raw Ry/bohr² (no baked ASR — QE applies asr='crystal' at matdyn time);
  bohr→Å structure conversion applied (a=3.295 Å); atom-order mapped via PH_Q2R helpers.

## Files
- Figures: `figures_core/dispersion_linewidth.{png,pdf}` (γ), `dispersion_lifetime.{png,pdf}` (τ),
  `comparison_summary.csv` (M/K γ,τ + pairwise ratios).
- γ: `gamma_V2_loto_res.npz`, `gamma_V3_res.npz` (+ extras: V2_loto_nores, V2_noloto_{res,nores}, V3_nores).
- Inputs: `dfpt_structure.xyz`, `dfpt_fc2.npy`, `nac_2d.npz`, `phono3py_disp.yaml`,
  `dfpt_modes_{loto,noloto}_floored.npz`; forces `f3_res/f2_res` + `disp_forces_*_nores`.
- Prep/gate logs: `prep.log`, `gate*/`, `clean/`. Driver: `driver_prep.sh`, `driver_extras.sh`, `bin/matdyn.x` (OMP=1 shim).

## Full factorial (5 of 7 variants) — `figures_full/`
M/K top-optical γ(THz)/τ(ps): mlip_strict K 0.0076/10.4 · dfpt_loto_res K 0.0313/2.54 ·
dfpt_loto_nores K 0.0313/2.54 · dfpt_noloto_res K 0.598/0.13 · mlip_at_dfpt_res K 0.0107/7.5.

**Ablation conclusions (all axes):**
1. **FC2-source (DFPT eigvecs vs MLIP): ~2.9× at M/K** — the dominant, physical effect.
2. **Geometry (DFPT vs MLIP-relaxed): small** (γ ratio 0.7–1.1×).
3. **Residual-force subtraction: negligible** — dfpt_loto res-on vs res-off agree to 4 sig figs
   (K 0.031290 vs 0.031283); the constant F(0) cancels in the FC3 finite differences. So for this
   un-re-relaxed DFPT geometry, subtracting residual forces does not change M/K linewidths.
4. **2D-LO-TO essential (magnitude is a regularization artifact)** — noloto γ is ~15–19× the loto
   value at M/K, BUT this is largely because dropping loto_2d destabilizes the near-Γ manifold
   (227 floored modes vs 57), and those floored soft modes act as *decay partners* for the M/K
   optical modes. So the number demonstrates **2D-LOTO is required for a stable harmonic**, but the
   "19×" is not a clean measure of the LOTO term's direct effect on M/K linewidths — bound/reframe it.

## Pre-merge review (Sol + Opus, 2026-07-16) + magnitude caveats
No P0. Verified the V1/V2/V3 isolation is genuinely clean (same fc3/geometry/masses; differ only in
the phonons) → the **qualitative headline is sound**: DFPT eigenvectors dominate M/K optical γ, and
residual-force subtraction is genuinely negligible. Review fixes applied (commit 5224c38): NAC Born
`.T` orientation, compare uses injected modes for DFPT band-lines, `--fc2-source dfpt` fallback
guard, matdyn abs-paths, `MATDYN_BIN` override, atom-order assert. 40 tests pass.

**Two magnitude caveats to resolve before publishing the EXACT 2.9× (direction is robust):**
- **Gauge sign −1 not validated in-situ** — the eigenvector-overlap gate is non-functional for this
  NAC-sensitive heterobilayer (mean overlap ~0.05–0.09). Sign −1 rests on the documented
  phonopy↔QE convention + `select_gauge` independently landing on −1. Recommend an overlap check
  restricted to **high-optical modes** (not 2D-LOTO-sensitive) for −1 vs +1.
- **0.02 THz floor leaks into M/K γ via decay channels** — V2-loto's 57 floored near-Γ modes are
  decay partners for M/K optical modes; their contribution to the 2.9× is unquantified. Recommend a
  **refloor/exclude re-run** of V2-loto to bound M/K stability.

**Full 7/7 factorial complete** (`figures_full7/` = 7-panel γ + τ + CSV). The two final res-off
variants confirm the residual-negligible result end-to-end: V3-nores γ_max 0.0136 = V3-res 0.0136;
V2-noloto-nores γ_max 34.46 = V2-noloto-res 34.46 (residual-force subtraction changes nothing at
any variant). Reproducible via committed `examples/mose2_wse2_bilayer_dfpt/run_dfpt_linewidth.sh`.

## Status
Core (V1|V2|V3) + 5/7 factorial complete, both figure sets (`figures_core/`, `figures_full/`) + CSVs.
Code: `mlip_phonon_scattering` branch `split_mlip_phonon_lifetime` (commits `e66c149`→`e4882ea`).
Phase-2 (fork-native `DynamicalMatrixQELoto2D`) not started — injection route delivered the science.
