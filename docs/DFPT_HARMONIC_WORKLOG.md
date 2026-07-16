# DFPT-harmonic + split-MLIP phonon-linewidth feature — work log, changes & fixes

**Dates:** 2026-07-15 → 2026-07-16 · **Repo:** `mlip_phonon_scattering` branch `split_mlip_phonon_lifetime`
· **Fork:** `phono3py_einsum` (`port/phono3py-4.3.3`) · **System:** MoSe₂/WSe₂ 6-atom bilayer, 36×36×1, 50 K.

## 1. Objective
Add a mode that computes phonon linewidths γ(q,ν)/lifetimes τ from a **hybrid**: harmonic phonons
(frequencies + eigenvectors) from a **completed QE DFPT** calculation (with 2D LO-TO), and anharmonic
FC3 from **split-MLIP (MACE)** forces on phono3py displacements at the DFPT geometry — then compare
against a strict all-MLIP pipeline, with dispersion figures colored by γ and by τ. Scientific
question: *does using DFPT phonons (eigenvectors) give more reasonable M/K linewidths than all-MLIP?*

## 2. Approach & key design decisions
- **Harmonic route = injection** (not a fork rewrite): run QE `matdyn` (loto_2d) on phono3py's exact
  mesh and inject freqs+eigenvectors via `Phono3py.set_phonon_data`; phono3py contracts the MLIP
  real-space FC3 against the injected DFPT eigenvectors. A fork-native `DynamicalMatrixQELoto2D`
  (Phase 2) was scoped as a follow-on.
- **Ablation matrix** (7 variants): V1 `mlip_strict` (all-MLIP) · V2 `dfpt_harm` (DFPT harmonic,
  loto/noloto × residual on/off) · V3 `mlip_at_dfpt` (MLIP fc2 at DFPT geometry, residual on/off).
  Isolates: FC2-source (DFPT vs MLIP eigvecs), geometry, 2D-LOTO, residual-force subtraction.
- Process: brainstorm → spec → plan → **Understand** → **Implement** (Terra/gpt-5.6) → **Review**
  (Sol/gpt-5.6 + Opus) → **Fix** → compute → final review. Model routing per `ued/.claude/CLAUDE.md`.

## 3. What was built (modules/CLIs, `mlip_phonon_scattering/linewidth/`)
- `dfpt_read.py` (`mlip-linewidth-read-dfpt`) — read a DFPT calc: parse `ifc.q2r.xml` → fc2 + NAC
  (ε, Born, α, c, area), structure (no relax), D3-trap + Γ-acoustic gates.
- `matdyn_modes.py` (`mlip-linewidth-matdyn-modes`) — full-BZ q-list, run `matdyn` (loto on/off),
  parse + gauge-correct modes, near-Γ floor → injected-mode npz.
- `gauge.py` — atom-dependent Bloch-gauge transform + overlap/subspace validation.
- `compare_dispersion.py` (`mlip-linewidth-compare`) — ω(q) colored by γ and by τ, multi-variant
  panels + `comparison_summary.csv`.
- Modified `forces3.py` (`--subtract-reference-forces`), `fc_cache.py` (`--fc2-source dfpt`),
  `gpu_scattering_…comm_mesh.py` (`--injected-modes` → `set_phonon_data`).
- `examples/mose2_wse2_bilayer_dfpt/run_dfpt_linewidth.sh` — reproducible end-to-end 7-variant sweep.
- Phase-2 (fork, WIP): `phono3py/phonon/dynamical_matrix_loto_2d.py` + solver/factory wiring.

## 4. Results (headline)
M/K top-optical, T=50 K: **DFPT eigenvectors → ~2.9× larger optical γ** than MLIP (τ_K 2.5 ps DFPT
vs 7.5 ps MLIP-at-DFPT vs 10.4 ps all-MLIP). Geometry effect small (0.7–1.1×). Residual-force
subtraction **negligible** (res-on ≡ res-off to 4 sig figs). 2D-LOTO required for a stable harmonic.
Same direction as the earlier ued stage6/9 DFPT-eigenbasis finding. Figures + CSV in
`runs/mose2_wse2_bilayer_dfpt_36x36x1_T50/figures_{core,full7}/`; details in `RESULTS_dfpt_bilayer_20260715.md`.

---

## 5. Changes & fixes made (the debugging record)

### 5a. Understand-phase API corrections (caught before writing code; C1–C11)
Two agents verified the plan's assumed APIs against the real code and produced 11 corrections:
- **C1** phonopy `PH_Q2R` is a *text* parser and crashes on the XML `ifc.q2r.xml` → parse the XML directly.
- **C2** `set_phonon_data` needs eigenvectors `(Ngrid, nband, nband)` **modes-as-columns**, over the
  **full BZ grid** — not the stored `(Ngrid, nb, nat, 3)`; reshape/transpose before injection.
- **C3** injection is safe if `init_phph_interaction(nac_q_direction=None)` (the `phonon_done` guard
  stops `run_phonon_solver` from overwriting injected modes).
- **C4** `matdyn.modes` eigenvectors are **already** mass-weighted — do NOT divide by √M.
- **C5** the full BZ grid ≫ `prod(mesh)` (3999 vs 1296 at 36×36×1) due to the anisotropic slab.
- **C6** QE fc2 is **Ry/bohr²** → THz factor **108.97**, not the default 15.63.
- **C7** residual F(0) needs no MPI broadcast (rank-0-only save). **C8** the DFPT-fc2 edit lives in the
  shared `set_phono3py_forces_with_mesh_fc_cache` helper, not `fc_cache.py`. **C9** entry points must be
  fully-qualified. **C10** no `need()` guard exists in examples. **C11** figure-reuse needs `out_dirs`+`label`.

### 5b. Foundation code review (Sol/gpt-5.6, R1–R7) — 5 P1 bugs unit tests missed
- **R1** fc2 index convention (s/s1 + block transpose). **R2** removed a spurious ASR hack (store raw
  IFCs — QE applies `asr='crystal'` at matdyn time, not baked). **R3** bohr→Å cell conversion (phonopy
  `read_pwscf` returns bohr; without this the cell was 1.889× too big → would corrupt all FC3 forces).
  **R4** keep fc2 raw Ry/bohr². **R5** gauge returns the complete reproducible transform. **R6** fixed a
  self-comparison in the matdyn test. **R7** per-stage reference-force filename.

### 5c. Compute-phase discoveries (the hard ones)
- **matdyn 2D-LOTO OpenMP race (biggest catch).** The dispersion gate failed at 8.9–20.7 cm⁻¹; root
  cause = an OpenMP race in QE `PHonon/PH/rigid.f90 rgd_blk` corrupting the Γ non-analytic term with
  >1 thread. With **`OMP_NUM_THREADS=1`** the gate is **bit-exact** (0.000000 cm⁻¹) vs `phband.freq`.
- **Wrong matdyn binary.** The reference `phband.freq` was made by `q-e-epw-tdbe-speedup/bin/matdyn.x`
  (+ cray modules), not the trunk build; using the right binary was needed for reproduction.
- **Gauge sign = −1** (QE→phonopy), validated against `docs/phonopy_eigenvector_gauge.md`; the
  overlap-residual metric is unreliable for this NAC-sensitive heterobilayer, so the sign is forced.
- **DFPT near-Γ 2D-flexural instability** (real, OMP-independent: loto min −0.67 THz over 38/3999 q)
  absent in the MLIP harmonic — floored to +0.02 THz so the scatter runs (M/K unaffected).
- Infra gotchas: Perlmutter `/tmp` is node-local (use `$PSCRATCH` for srun); `MACE_PHONON_QUIET=1`
  skips the slow ~3–4 min import-sanity report.

### 5d. fc2 double-transpose bug — the mid-q dispersion error (commit `aab60c9`)
The Phase-2 dispersion gate stalled at 15.7 cm⁻¹; a localize diagnostic proved it was the **short-range
fc2**, not the 2D kernel. Root cause: QE `io_dyn_mat.f90 write_ifc` serializes each 3×3 IFC block
**column-major**, but NumPy `reshape` reads **row-major**, so the parsed block is already transposed;
the code's explicit `.T` **double-transposed** it. **Fix: remove the `.T`** (`dfpt_read.py:299`) →
**15.7 → 4.3 cm⁻¹**, and it corrected the figure band-lines. (Only affected the Phase-2 kernel + band
lines; Phase-1 γ was unaffected — injection uses matdyn's eigvecs, not `dfpt_fc2`.)

### 5e. B hardening (commit `5817fa6`)
D3-gate path (`ph.out`→`out.log`→glob); `--floor-freq-thz` flag (floor was a manual step);
`--gauge-sign -1` guard with a `select_gauge` cross-check warning; committed reproducible
`run_dfpt_linewidth.sh` (bakes `OMP_NUM_THREADS=1` + speedup binary + cray env + floor + gauge).

### 5f. Final pre-merge review fixes (Sol + Opus, no P0; commit `5224c38`)
- **NAC Born-tensor `.T`** — the XML column-major decode left ε/Z* transposed; ε is symmetric
  (harmless) but Z* is non-symmetric. Fixed for correctness (did *not* close the Phase-2 residual).
- **compare** now uses the **injected modes** for DFPT band-lines (was re-diagonalizing raw fc2).
- **`--fc2-source dfpt`** now skips the legacy MLIP-cache fallback (could silently use a stale fc2).
- **matdyn abs-paths** (relative `workdir` double-prefix bug); **`MATDYN_BIN`** override for portability;
  **atom-order self-consistency assert** in `dfpt_read`.

### 5g. Process lessons (recorded for future runs)
- **Parallel codex agents in one non-worktree checkout clobber each other** via stray `git checkout`
  (observed corrupting an in-flight agent's edits). Run code agents **sequentially** or worktree-isolated.
- **Agents sometimes finish without firing a completion notification** → poll on-disk state, don't only
  wait for notifications. **codex `git commit` is slow on the large fork** → `--no-verify` + background.

## 6. Phase-2 (fork-native `DynamicalMatrixQELoto2D`) — status
Committed WIP (`phono3py_einsum` `ebf45473`): kernel + C/Rust solver branches + factory; 3/4 tests
pass. Dispersion gate at **4.30 cm⁻¹** (paper gate <0.5 not met). The fc2 fix (15.7→4.3) was the big
step; the remaining residual is a **structural 2D-LOTO reciprocal-kernel** issue (ruled out gauge-sign,
uniform-scale, and the NAC-Born-transpose) — the multi-day re-derivation originally scoped. **Phase-1
injection is the production route**; Phase-2 is a reusability follow-on.

## 7. Magnitude caveats — resolved by guardrail runs (2026-07-16)
The qualitative headline is robust (verified: V1/V2/V3 differ only in the phonons). Two numerical
choices that affect the *exact* magnitudes were firmed by two guardrail runs (gpt-5.6-terra on job
`55982740`; artifacts under `runs/.../guardrails/`):
- **Gauge sign −1 — CONFIRMED at high-optical K/M.** Mean top-4-optical eigenvector overlap: K −1
  **0.996** vs +1 0.691; M −1 **0.994** vs +1 0.591. The production sign −1 is correct at the
  physically-relevant modes, independent of the (non-functional) whole-spectrum overlap gate.
- **Near-Γ floor sensitivity — 2.9× robust at K, floor-sensitive at M.** M/K top-optical γ re-floored
  at 0.005 / 0.020(prod) / 0.050 THz: **K spread ~0.12%** (0.0310/0.0310/0.0310) — robust; **M spread
  ~36%**, driven entirely by the aggressive 0.005 THz case (M γ 0.0300 vs prod 0.0222), while prod
  0.02 and looser 0.05 agree to ~0.4% (0.02218 vs 0.02210). So the K-optical 2.9× is solid; the
  **M-optical magnitude carries a ±~35% floor caveat** and should be bounded, not quoted bare.
Separately, the noloto "19× LOTO-essential" number is largely a floor-regularization artifact (noloto
floors many more near-Γ modes → inflated decay partners), so it demonstrates LOTO is *required* for a
stable harmonic but is not a clean measure of the LOTO term's direct M/K effect.

## 8. Reproducibility checklist
`MACE_PHONON_QUIET=1` + `OMP_NUM_THREADS=1`; `matdyn.x` = `q-e-epw-tdbe-speedup/bin` + `module load
cray-fftw cray-hdf5-parallel`; GPU srun `--overlap -n 4 --gpus-per-task=1` (never -n16); `$PSCRATCH`
work dirs (not `/tmp`); gauge sign −1; near-Γ floor 0.02 THz. One-shot: `run_dfpt_linewidth.sh`.

## 9. Commit history (feature branch, this effort)
spec `e34ff90` → plan `e0d16a2` → Understand corrections `020c87b` → foundation `e66c149` →
CLIs `d0afd3a` → injection `09b6bcb` → figures `40eaa3f` → results `e4882ea`/`7f9feaa` → B hardening
`5817fa6` → 7/7 `99a9a1a` → fc2 fix `aab60c9` → E fixes `5224c38` → caveats `29b4793`. Fork Phase-2
WIP `ebf45473`.

## 10. What's left
Magnitude guardrail runs (in progress); local merge to main (no push); optional: monolayer + 25°
systems; close the Phase-2 4.3 cm⁻¹ residual (fork-native 2D-LOTO).
