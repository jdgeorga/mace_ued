# Design: DFPT-harmonic + split-MLIP-anharmonic phonon linewidths

**Date:** 2026-07-15
**Repo:** `mlip_phonon_scattering` (branch `split_mlip_phonon_lifetime`)
**Status:** design approved by user; spec under review before `writing-plans`.

## 1. Goal & scientific context

Add a mode to the `mlip-linewidth-*` pipeline that computes phonon linewidths γ(q,ν)
and lifetimes τ from a **hybrid** of first-principles harmonic phonons and MLIP
anharmonicity:

- **Harmonic** (frequencies + eigenvectors): taken from a **completed QE DFPT
  calculation** (with its 2D LO-TO electrostatics), *not* from the MLIP.
- **Anharmonic** (third-order force constants, FC3): computed from **split-MLIP
  (MACE) forces** on phono3py displacements, at the DFPT geometry (no re-relaxation).

phono3py then contracts the MLIP real-space FC3 against the **DFPT eigenvectors** to
get the three-phonon matrix elements and γ(q,ν). This directly tests whether using
DFPT-quality phonons (which a prior effort found to be the dominant lever on linewidth
accuracy — see the ued-project stage2/stage6 audits) yields more reasonable linewidths
than a fully-MLIP pipeline, while keeping the cheap, validated MLIP FC3.

**Reference system for the first build:** MoSe₂/WSe₂ 6-atom aligned bilayer
(split-MLIP interlayer model), DFPT at
`/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/`
(6×6×1 q-mesh, loto_2d, SOC, DFT-D3), linewidth mesh 36×36×1, T = 50 K.

## 2. Scope

**In scope:** reading a completed QE DFPT calc (structure + `ifc.q2r.xml` +
DIELECTRIC/ZSTAR); residual-force subtraction; split-MLIP FC3 at the DFPT geometry;
two harmonic implementations of the DFPT 2D-LO-TO phonons (injection, then fork-native);
NAC/2D-LOTO; the variant/ablation matrix; comparison figures (dispersion colored by
γ and by τ, both pipelines); validation gates.

**Out of scope:** re-relaxing the DFPT structure; running new DFPT/SCF; changing the
existing all-MLIP pipeline's default behavior (new behavior is opt-in/flag-gated);
electron-phonon; 4-phonon.

## 3. Variant / ablation matrix

All on the bilayer, 36×36×1, T = 50 K. FC3 is always split-MLIP (3×3×1). The only
differences are **geometry**, **harmonic FC2 source**, and **residual subtraction**.

| Variant | Geometry | Harmonic FC2 (freqs+eigvecs) | FC3 | NAC | Residual |
|---|---|---|---|---|---|
| **V1 `mlip_strict`** (reference; exists) | MLIP-relaxed | MLIP (forces2-from-3, 6×6×1) | MLIP 3×3×1 | none | n/a (relaxed → F≈0) |
| **V2 `dfpt_harm`** (headline new) | DFPT (read, no relax) | **DFPT loto_2d** (injected → then fork-native) | MLIP 3×3×1 @ DFPT geom | 2D-LOTO | toggle {on, off} |
| **V3 `mlip_at_dfpt`** (ablation) | DFPT (read, no relax) | MLIP (forces2-from-3 @ DFPT geom) | MLIP 3×3×1 @ DFPT geom | none | toggle {on, off} |

**Comparisons the matrix isolates:**
- **V2 vs V1** — headline: DFPT-harmonic hybrid vs fully-MLIP.
- **V3 vs V1** — geometry effect (both MLIP FC2; DFPT vs MLIP-relaxed geometry).
- **V2 vs V3** — **FC2-source effect** (identical DFPT geometry + identical MLIP FC3;
  DFPT-loto_2d FC2 vs MLIP FC2) — the clean isolation of "does the DFPT harmonic matter."
- **residual on vs off** within V2/V3 — effect of the reference-force correction.

Because V2 and V3 share geometry, FC3, and residual setting, **the MLIP FC3 is computed
once per (residual setting) and reused across V2/V3**; V2 and V3 differ only in the FC2
handed to phono3py.

Runs to produce: V1, V2{on,off}, V3{on,off} = 5 linewidth runs.

## 4. Architecture — how it plugs into the existing pipeline

Reuse the existing `mlip-linewidth-*` stages unchanged where possible; add new stages
and flags for the DFPT-harmonic path. Existing stages: relax → phonopy/phono3py-yaml →
forces2/forces3/forces2-from-3 → cache-fc → **scatter-gpu** → extract-gamma → plot
(`mlip_phonon_scattering/linewidth/`).

New / modified stages:

| Stage | Command | New? | Behavior |
|---|---|---|---|
| Read DFPT | `mlip-linewidth-read-dfpt` | **new** | Point at the DFPT dir → emit `dfpt_structure.xyz` (with `atom_types`/`layer_ids`), `dfpt_fc2.npy` (via phonopy `PH_Q2R`), `nac_2d.npz` (ε∞, Born, α, cell/area/c, periodic axes). Run the D3-trap + Γ-acoustic gates. |
| Harmonic (Phase 1) | `mlip-linewidth-matdyn-modes` | **new** | Generate a matdyn input over phono3py's exact 36×36×1 BZ grid addresses; run QE `matdyn` (loto_2d); parse freqs + eigenvectors → `dfpt_modes.npz`. |
| Displacements | `mlip-linewidth-phono3py-yaml` | reuse | On `dfpt_structure.xyz`, supercells 3×3×1 (fc3) / 6×6×1 (fc2). |
| Forces (FC3) | `mlip-linewidth-forces3` | **modified** | Add `--subtract-reference-forces` (default on in DFPT mode): compute F_MLIP on the undisplaced supercell once, subtract from every displaced set. |
| Cache FC | `mlip-linewidth-cache-fc` | **modified** | `--fc2-source {mlip,dfpt}`; when `dfpt`, use `dfpt_fc2.npy` as fc2 and **do not** call `symmetrize_fc2`. |
| Scatter | `mlip-linewidth-scatter-gpu` | **modified** | `--injected-modes dfpt_modes.npz` (Phase 1) → `set_phonon_data` after `init_phph_interaction`; or `--nac-2d nac_2d.npz` (Phase 2) → fork-native `DynamicalMatrixQELoto2D`. Both default off (preserve golden). |
| Extract γ | `mlip-linewidth-extract-gamma` | reuse | unchanged. |
| Compare figs | `mlip-linewidth-compare` | **new** | Dispersion colored by γ and by τ, all variants (§8). |

Design-for-isolation: each new module has one purpose and a file interface —
`dfpt_read.py` (DFPT dir → structure/fc2/NAC arrays), `matdyn_modes.py` (grid → QE modes
npz), `gauge.py` (Bloch-gauge transform + overlap validation), `dynamical_matrix_loto_2d.py`
(Phase-2 fork kernel), `compare_dispersion.py` (figures). Each is unit-testable on its
file inputs without the GPU engine.

## 5. Harmonic route (the one true representational gap)

Only ONE DFPT setting is not representable by phono3py's finite-range fc2 + stock NAC:
the **2D LO-TO** (`loto_2d`) long-range electrostatics (QE Sohier–Calandra–Mauri).
phono3py's built-in NAC is 3D Gonze/Wang and would apply the wrong, vacuum-contaminated
q→0 correction (symptom: Γ flexural mode at −12 cm⁻¹). Two phases, per user decision
"Both now":

### Phase 1 — QE matdyn on the mesh + `set_phonon_data` injection (primary, ~0.5–1.5 d)
- The GPU scatter path **does not diagonalize on-device**; it consumes host
  frequency/eigenvector arrays. Inject via `Phono3py.set_phonon_data(frequencies,
  eigenvectors, grid_address)` (`api_phono3py.py:1280`) after `init_phph_interaction`.
- Generate matdyn q-list = phono3py's exact BZ grid addresses (`q = addresses · QDinv^T`,
  incl. BZ-boundary duplicates); run QE `matdyn` with `loto_2d=.true.`, `asr='crystal'`.
- **Gauge (critical):** apply the atom-dependent Bloch phase `U_κ(q)=exp(±2πi q·τ_κ)`
  to the QE eigenvectors before injection. A global phase cancels in |V³|² but an
  atom-dependent mismatch silently corrupts matrix elements. Sign determined empirically
  by the overlap test (§7). matdyn writes √M-rescaled eigenvectors — undo to phono3py's
  mass convention.

### Phase 2 — fork-native `DynamicalMatrixQELoto2D` (validated against Phase 1)
- New fork module `phono3py/phonon/dynamical_matrix_loto_2d.py`: a literal mass-weighted
  port of QE `rgd_blk(...,loto_2d=.true.)` (`PHonon/PH/rigid.f90`). Subclass the plain
  `DynamicalMatrix` (NOT `DynamicalMatrixNAC` — compiled dispatch treats unknown NAC
  subclasses as 3D Wang). Single authoritative kernel `D = D_short_range + D_QE_loto2d`
  used by both `run(q)` and `get_dynamical_matrices(qpoints)`.
- Add an early explicit branch for `DynamicalMatrixQELoto2D` in the C and Rust grid
  solvers (`phono3py/phonon/solver.py`), since they bypass the virtual `run(q)`.
- Inputs (all in `ifc.q2r.xml`): ε∞, post-ZASR Born, α_ewald=0.982, area A, c=Ω/A,
  periodic axes (0,1). New `nac_params` fields: `method="qe_loto_2d"`, `loto_2d_alpha`,
  `loto_2d_periodic_axes`, `loto_2d_normal_axis`, `loto_2d_c`, `loto_2d_fc_is_short_range=True`.
- Full math, file:line refs, and the on-site subtraction / G-sum cutoff are in the
  companion **`docs/loto2d_implementation_plan.md`** (Sol/gpt-5.6-sol, xhigh).

The Phase-1 injected 36×36×1 dataset is the **regression gold standard** for Phase 2.

## 6. Residual-force subtraction (toggle)

phono3py's `produce_fc2/fc3` assume the undisplaced reference has zero force. At the
DFPT geometry (not the MLIP minimum) the MLIP has a nonzero residual force F_MLIP(0),
which contaminates the finite-difference FCs. The correction:

  F_corrected(u) = F_MLIP(u) − F_MLIP(u=0)

computed once on the undisplaced supercell and subtracted from every displaced set,
before `produce_fc3`/`produce_fc2`. `--subtract-reference-forces` default **on** in DFPT
mode; run both on/off (user: "play around with it"). Report |F_MLIP(0)| and the fc3
with/without diff. In V2, residual affects only FC3 (FC2 is DFPT); in V3 it affects both
MLIP FC2 and FC3.

## 7. Correctness pitfalls & validation gates (acceptance criteria)

From the DFPT-vs-phono3py conflict audit + Sol:

1. **D3 trap gate** (`nscf-clobbers-vdw-ph-skips-d3`): confirm this DFPT's ph.out shows
   "Grimme-D3 Hessian read" and Γ acoustic ≈ 0 before trusting fc2.
2. **ASR:** leave phono3py `symmetrize_fc2` **OFF** for the DFPT fc2 (zasr='crystal'
   already applied; double-ASR perturbs values). Reproduce matdyn `asr='crystal'` if
   Phase-2 rebuilds from IFCs.
3. **Atom-order map:** verify phonopy `PH_Q2R` maps QE's Mo/Se/Se/W/Se/Se (two distinct
   Se environments) onto phono3py's supercell order (a silent swap corrupts fc2 with no
   error). Assert positions match < 1e-4 after mapping.
4. **BvK supercell = 6×6×1** carried into phono3py; **units/masses** (QE→THz factor,
   per-atom amu) match QE (a wrong factor uniformly rescales frequencies).
5. **Harmonic cross-validation gate (make-or-break):** phono3py-reconstructed dispersion
   (injected in P1; fork-native in P2) vs QE matdyn `phband.freq` on the exact 152-pt
   Γ–M–K–Γ path: **max |Δω| < 0.5 cm⁻¹, RMS < 0.1 cm⁻¹**; Γ LO/TO degenerate with a
   finite linear LO slope (2D behavior); 3 acoustic modes → 0 at Γ.
6. **Gauge/eigenvector gate:** |⟨e_phono3py|e_QE⟩| > 0.9999 (isolated modes), min
   subspace singular value > 0.9999 (degenerate clusters).
7. **γ ≥ 0 & finite; τ > 0** (existing `validate`); interlayer shear (~0.57) + breathing
   (~0.85 THz) present.

**Documented science caveat (not a blocker):** DFPT fc2 carries DFT-D3 (2-body)
dispersion; the split-MLIP FC3 does not include Grimme-D3 explicitly (whatever MACE
learned). Inherent to mixing DFPT-harmonic + MLIP-anharmonic; state in methods.

Everything else in the DFPT (assume_isolated='2D' short-range, DFT-D3 in fc2, SOC,
ecut/conv) is fully baked into `ifc.q2r.xml` and needs no fork work. (Corrected earlier
assumptions: XC is plain PBE — no vdW-DF kernel; fixed occupations — no smearing.)

## 8. Comparison figures (deliverable)

`mlip-linewidth-compare` reuses `linewidth_path.compute_band_structure` + the γ-on-path
joint-(q,ω) interpolation. **Project rule: interpolate γ, never τ** (τ=1/γ diverges).
- **Fig 1 — linewidth:** ω(q) along Γ–M–K–Γ as a colormapped scatter/line, colored by γ
  (THz), panels for the compared variants side-by-side with a **shared colorbar**.
- **Fig 2 — lifetime:** same layout, colored by τ = 1/(4πγ) (ps).
- **Primary comparison panels:** V1 `mlip_strict` | V2 `dfpt_harm` (residual-on) |
  V3 `mlip_at_dfpt` (residual-on). **Supplementary:** residual on/off for V2 (and V3).
- Plus a CSV/text `Δ` summary: top-optical γ, τ at K and M for each variant + the
  pairwise ratios (V2/V1, V3/V1, V2/V3).

## 9. Test plan

- **Unit:** `PH_Q2R` fc2 atom-order mapping (assert positions/species); residual
  subtraction (F_corrected(0)=0); NAC-array extraction from `ifc.q2r.xml`; gauge helper
  (round-trip + overlap); matdyn-modes parse (shape, masses, gauge).
- **Physics gates:** §7.5 (dispersion vs `phband.freq`), §7.6 (eigenvector overlap),
  §7.7 (γ≥0, τ>0, interlayer modes).
- **Integration:** both P1 and P2 end-to-end on a small mesh (e.g. 12×12×1) → the two
  comparison figures render; P2 reproduces P1 injected freqs/eigvecs within tolerance.
- **Regression:** existing all-MLIP golden (bilayer γ_max 0.020164 THz) unchanged when
  DFPT flags are off.

## 10. Data & paths

- DFPT: `…/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/` — `collect/ifc.q2r.xml` (6×6×1,
  loto_2d), `collect/matdyn.in` + `phband.freq` (reference dispersion), `collect/matdyn.modes`,
  Γ dielectric/Born in `q1/mose2_wse2.dyn1.xml` (use **post-ZASR** values from
  `ifc.q2r.xml`), structure in `1-mf/out_scf/` (a=3.295 Å vc-relaxed).
- MLIP models: `split_mlip_generators/models/{MoSe2,WSe2,MoSe2_WSe2}.model`.
- Reference all-MLIP run: `split_mlip_generators/runs/mose2_wse2_bilayer_36x36x1_T50`.
- Companion loto_2d plan: `docs/loto2d_implementation_plan.md`.

## 11. Execution plan (after spec approval → `writing-plans` → Workflow)

Per project model routing (`ued/.claude/CLAUDE.md`), multi-agent pipeline:
- **Understand/Plan:** Opus (architecture, interfaces) + **Sol** (gpt-5.6-sol) independent plan review.
- **Implement:** **Terra** (gpt-5.6-terra, codex-implement) for bulk modules (dfpt_read,
  matdyn_modes, residual, cache-fc/scatter wiring, gauge, Phase-2 fork kernel + solver
  branch); **Opus** for the comparison figures (taste ≥ 7, user-facing).
- **Test:** Terra runs pytest + the small-mesh integration + the §7 gates.
- **Review:** **Sol** (codex-review) + Opus/Fable, focused on the gauge, ASR, atom-order,
  and NAC correctness.
- **Triage + Fix:** Opus triages findings → Terra fixes → re-review until gates pass.
- GPU stages run under `salloc` (`srun --overlap -n 4 --gpus-per-task=1`; `-n 16` hangs).

## 12. Open items / risks

- Bloch-gauge sign (± q·τ) resolved empirically by the overlap gate (§7.6) — must pass
  before any linewidth is trusted.
- matdyn q-list must cover phono3py's BZ-grid addresses incl. boundary duplicates.
- Phase-2 fork kernel: C/Rust solver branch + QE-vs-phonopy gauge/α/unit conventions are
  the main risk; Phase-1 dataset gates it.
- `set_phonon_data` must mark all grid points solved and not be overwritten by a later
  `run_phonon_solver`.
