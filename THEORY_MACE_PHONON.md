# Theory — the MACE-interlayer → phonopy phonon workflow

This document explains **what each stage of the pipeline is doing physically and
mathematically**. It is the "why" companion to the operational guides: see
[`AGENTS.md`](AGENTS.md) to *run* the workflow and [`SETUP_MACE_PHONON.md`](SETUP_MACE_PHONON.md)
to *build* it. Everything below is implemented in
`repos/mlip_phonon_scattering/mlip_phonon_scattering/` (`calculator.py`,
`relax.py`, `phonopy_io.py`, `forces.py`, `solve.py`) and the interlayer helpers
in `repos/mace-interlayer/examples/interlayer_helpers/` (`n_layer.py`,
`macewrapper.py`).

The goal of the whole pipeline is the **harmonic phonon band structure**
ω<sub>ν</sub>(**q**): the normal-mode vibration frequencies of the crystal as a
function of wavevector **q**. From it follow thermodynamics, lattice stability,
Raman/IR-active modes, interlayer shear/breathing modes, and the inputs to
phonon-scattering (e.g. UED) calculations.

---

## 1. Block diagram

```
                        ┌─────────────────────────────────────────────────┐
                        │  MACE machine-learned interatomic potential(s)   │
                        │  E({R}) , F_i = −∂E/∂R_i   (the "calculator")    │
                        │                                                  │
                        │  monolayer:  one MACE model                      │
                        │  bilayer:    NLayerCalculator =                  │
                        │     Σ_L E_intra(L) + Σ_⟨L,L'⟩ E_inter(L,L')     │
                        └───────────────┬──────────────────────────────────┘
                                        │  provides energy & forces to every stage
                                        │
   input.xyz                           ▼
 (species, cell,        ┌───────────────────────────────┐
  atom_types,           │  STAGE 1 — RELAX               │   relax.py
  layer_ids)  ────────► │  FIRE: minimize E({R})         │   FIRE optimizer
                        │  until max|F_i| < fmax         │   ── build_calculator()
                        └───────────────┬───────────────┘
                                        │  R⁰ : equilibrium geometry (forces ≈ 0)
                                        ▼   prefix.xyz
                        ┌───────────────────────────────┐
                        │  STAGE 2 — DISPLACEMENTS       │   phonopy_io.py
                        │  build N×N×1 supercell;        │   Phonopy.generate_
                        │  enumerate symmetry-distinct   │   displacements()
                        │  finite displacements ±d=0.01Å │
                        └───────────────┬───────────────┘
                                        │  disp.yaml  +  {displaced supercells}
                                        ▼
                        ┌───────────────────────────────┐
                        │  STAGE 3 — FORCES              │   forces.py
                        │  F = calculator(R⁰ + Δ)        │   build_calculator()
                        │  for every displaced supercell │   (MPI-parallel optional)
                        └───────────────┬───────────────┘
                                        │  forces.npy  (n_disp, n_atoms_sc, 3)
                                        ▼
                        ┌───────────────────────────────┐
                        │  STAGE 4 — SOLVE               │   solve.py
                        │  finite-diff → force constants │   Phonopy.produce_
                        │  Φ → dynamical matrix D(q)     │   force_constants()
                        │  eig D(q) → ω²_ν(q), e_ν(q)    │   run_band_structure()
                        └───────────────┬───────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                  band_structure.png        eigenvector_band.h5 / eigenvector.h5
                  ω_ν(q) along k-path       ω, eigenvectors, q-grid, metadata
```

The single most important idea: **the MACE potential replaces DFT** as the
source of energies and forces, and **phonopy** turns those forces into phonons
via the finite-displacement (frozen-phonon) method. MACE makes the many force
evaluations cheap enough to run on a CPU node in seconds.

---

## 2. Stage 0 — the potential: MACE and the interlayer decomposition

### 2.1 What a MACE model is

A trained MACE model is an approximation to the Born–Oppenheimer **potential
energy surface** (PES): a smooth scalar function of all nuclear coordinates,

  E = E({**R**₁, …, **R**<sub>N</sub>}),

learned from DFT reference data. MACE is an *equivariant message-passing graph
neural network*: each atom *i* carries features built from its neighbours within
a cutoff r<sub>c</sub>; messages are expanded in spherical harmonics and combined
with higher body-order products (the "ACE" tensor construction), so the predicted
energy is invariant under translation/rotation/permutation and the per-atom
energies depend on the local chemical environment up to a chosen body order.

Total energy is a sum of atomic site energies, E = Σ<sub>i</sub> ε<sub>i</sub>,
and **forces are exact analytic gradients** obtained by autodifferentiation:

  **F**<sub>i</sub> = −∂E/∂**R**<sub>i</sub>.

This is the only physics the calculator exposes to the rest of the pipeline:
given a geometry it returns (E, {**F**<sub>i</sub>}). Stages 1 and 3 are just
two different ways of calling it. `--default-dtype float64` is used so that the
forces are smooth enough for the tight relaxation (fmax = 10⁻⁵ eV/Å) and the
finite-difference second derivatives in Stage 4 are not swamped by float32 noise.

### 2.2 Monolayer: one model

For a monolayer (`examples/mose2_monolayer`) the calculator is a single
`MACECalculator` (wrapped as `CompatMACECalculator`, see Gotcha 2 in
`SETUP_MACE_PHONON.md`) loading `models/MoSe2.model`. One model evaluates the
whole 3-atom cell. No `atom_types`/`layer_ids` are needed.

### 2.3 Bilayer: the NLayerCalculator energy decomposition

A bilayer is held together by two physically distinct interactions:

- **Intralayer** bonding (strong, covalent/ionic within a TMD sheet), and
- **Interlayer** coupling (weak, van der Waals between sheets).

A single foundation model trained on monolayers does not know the interlayer
physics, so `build_nlayer_calculator` (`calculator.py:71`) builds a stacked
calculator that writes the total energy as a sum of layer-resolved terms.
For *n* layers L = 0…n−1:

  E<sub>tot</sub> = Σ<sub>L</sub> E<sub>intra</sub>(L) + Σ<sub>L<L′, adjacent</sub> E<sub>inter</sub>(L, L′)

In `n_layer.py` this is the upper-triangular sum over an n×n block: the diagonal
(i = i) terms are intralayer energies, the off-diagonal (i < j) terms are
interlayer energies, evaluated only for adjacent layers. Concretely for the
MoSe₂/WSe₂ bilayer:

  E<sub>tot</sub> = E<sub>MoSe₂</sub>(layer 0) + E<sub>WSe₂</sub>(layer 1) + E<sub>inter</sub>(0,1)

evaluated by three separate models — `MoSe2.model`, `WSe2.model`, and the pair
model `MoSe2_WSe2.model`. Each per-block calculator is a `MACEWCalculator` built
on the atom subset selected by contiguous `atom_types` ranges:

- intralayer block L → atoms whose `atom_types` lie in that layer's range
  (layer 0 → types 0–2, layer 1 → types 3–5);
- interlayer block (L,L′) → the **union** of both layers' atoms (types 0–5),
  evaluated with the interlayer model.

Forces follow the same block structure and are summed:

  **F**<sub>i</sub> = −∂E<sub>tot</sub>/∂**R**<sub>i</sub>
  = Σ<sub>blocks</sub> (−∂E<sub>block</sub>/∂**R**<sub>i</sub>),

implemented as `layer_forces.sum(axis=(0,1))` in `n_layer.py`. Each block's
calculator writes its forces into the global atom slots for the atoms it owns and
zeros elsewhere, so the sum reconstructs the full-system force on every atom.
The interlayer model contributes forces to atoms in *both* layers — that
cross-layer force is exactly the van der Waals coupling that produces the
interlayer shear and breathing phonons in Stage 4.

`MACEWCalculator.calculate` (`macewrapper.py:256`) remaps the global `atom_types`
to model-local "relative layer types" before calling MACE and restores them
after, so each sub-model sees atoms labelled the way it was trained.

> **Why the calculator is rebuilt per stage and bound to atoms.** `MACEWCalculator`
> records the `atom_types`/`layer_ids` of the atoms it was constructed with and
> reuses them at every `calculate()` (Gotcha 3). So the stacked calculator is
> valid only for one topology/size. Stage 1 builds it from the input cell; Stage 3
> rebuilds it from the *first displaced supercell* (`forces.py:30`), which has the
> supercell's atom count. Both are correct because all displaced supercells share
> topology.

---

## 3. Stage 1 — relaxation (find the equilibrium geometry)

**Physics.** Phonons are vibrations *about a mechanical equilibrium*. The
harmonic theory in Stage 4 expands the energy to second order around a point
where the first derivative (the force) vanishes. If that point is not a true
minimum, the linear term is nonzero and the "harmonic" frequencies are
contaminated — typically appearing as spurious **imaginary modes**. So we must
first drive the structure to

  **F**<sub>i</sub> = −∂E/∂**R**<sub>i</sub> = 0   for all *i*.

**Math / method.** `relax_structure` (`relax.py`) attaches the MACE calculator
and runs ASE's **FIRE** optimizer (Fast Inertial Relaxation Engine). FIRE
integrates a fictitious damped Newtonian dynamics,

  m **v̇** = **F** − γ(t) |**F**| (**v̂** − **F̂**),

mixing velocity toward the force direction and adaptively growing the timestep
while **F**·**v** > 0 (downhill), and zeroing velocity when it goes uphill. It
converges to the local minimum without needing a Hessian. `maxstep = 0.05 Å`
caps any single atomic move for stability. Iteration stops when

  max<sub>i</sub> |**F**<sub>i</sub>| < fmax = 10⁻⁵ eV/Å   (or after `steps`).

That tight `fmax` matters because the residual force sets the floor on how clean
the acoustic-mode ω→0 limit and any soft interlayer modes can be. The relaxed
geometry **R⁰**, the trajectory, and the final energy are written to
`prefix.xyz` / `prefix.traj`. For the bilayer, the cell is kept fixed and only
internal coordinates + layer registry/spacing relax under the combined
intra+inter potential, so the equilibrium interlayer distance is set by the vdW
model.

---

## 4. Stage 2 — finite displacements (set up the frozen-phonon problem)

**Physics — the harmonic approximation.** Expand the crystal energy about the
relaxed geometry **R⁰** in atomic displacements **u** = **R** − **R⁰**:

  E = E₀ + Σ<sub>iα</sub> (∂E/∂u<sub>iα</sub>)·u<sub>iα</sub>
        + ½ Σ<sub>iα,jβ</sub> Φ<sub>iα,jβ</sub> u<sub>iα</sub> u<sub>jβ</sub> + …

The linear term vanishes because Stage 1 made the forces zero. The harmonic
approximation keeps only the quadratic term. Its coefficients are the
**second-order interatomic force constants (IFCs)**

  Φ<sub>iα,jβ</sub> = ∂²E / ∂u<sub>iα</sub>∂u<sub>jβ</sub>
                    = −∂F<sub>iα</sub> / ∂u<sub>jβ</sub>,

where i, j label atoms and α, β ∈ {x, y, z}. The whole point of Stages 2–3 is to
measure Φ; Stage 4 turns Φ into frequencies.

**Why a supercell.** A phonon of wavevector **q** displaces atoms with a phase
e<sup>i**q**·**R**</sup>, so to represent IFCs out to a finite range — equivalently,
to resolve phonons at **q**-points denser than just Γ — the force response must be
computed in a cell large enough to contain that range. `generate_displacements`
(`phonopy_io.py`) builds a diagonal supercell **S** = diag(N, N, 1) (4×4×1
monolayer, 3×3×1 bilayer; the "1" because these are 2D — no periodicity needed out
of plane). Larger N captures longer-range IFCs and gives a denser commensurate
**q**-grid, at the cost of more/larger force evaluations.

**The finite-displacement (frozen-phonon) method.** Rather than differentiate the
forces analytically, phonopy measures Φ by finite difference. It displaces one
symmetry-inequivalent atom at a time by a small amount d = 0.01 Å along a
Cartesian direction, creating a set of displaced supercells. **Crystal symmetry**
(space group, found with `symprec = 1e-5`) is used to enumerate only the
*distinct* displacements; forces on all atoms in each displaced supercell then
generate the full Φ by symmetry. With the default one-sided displacements
(`is_plusminus=False`) phonopy uses +d and reconstructs the rest from symmetry;
a two-sided ±d set would cancel the leading anharmonic (cubic) error at double the
cost. The displacement dataset and supercell metadata are saved to `disp.yaml`.

`copy_repeated_arrays` (`phonopy_io.py:37`) propagates the per-atom `atom_types`
and `layer_ids` from the primitive cell onto each displaced supercell with
`np.repeat`, so the bilayer calculator can still slice atoms by layer in Stage 3.
(For a diagonal supercell this ordering matches phonopy exactly — Gotcha 4.)

---

## 5. Stage 3 — forces on the displaced supercells

This is the numerically heavy stage and the one MACE makes cheap. For each
displaced supercell `compute_displacement_forces` (`forces.py`) evaluates

  **F**(**R⁰** + **Δ**<sub>k</sub>) = MACE forces on supercell *k*,

i.e. it just calls the calculator's `get_forces()` for every displacement *k* and
stacks the results into an array of shape (n_displacements, n_atoms_supercell, 3)
saved as `forces.npy` (energies likewise to `energies.npy`).

Each force column is the numerator of the finite-difference derivative that
becomes a row of Φ:

  Φ<sub>iα,jβ</sub> ≈ −[ F<sub>iα</sub>(u<sub>jβ</sub> = +d) − F<sub>iα</sub>(0) ] / d.

Because the equilibrium forces are ≈ 0 (Stage 1) and the displacement is small,
this difference quotient approximates the second derivative in the harmonic
window. For the bilayer the forces come from the summed intra+inter
NLayerCalculator, so Φ contains genuine interlayer force constants.

The stage is **embarrassingly parallel** over displacements: if launched under
`srun`/`mpirun` with `mpi4py` present, rank 0 scatters the supercells across
ranks, each rank evaluates its chunk, and the forces are gathered and saved in
phonopy's original order. Serial otherwise. (Identical results either way — the
displacements are independent.)

---

## 6. Stage 4 — solve the harmonic phonon problem

`solve_phonons` (`solve.py`) is where the measured forces become physics.

### 6.1 Build and symmetrize the force constants

```
phonon.forces = np.load(forces_path)
phonon.produce_force_constants()      # finite-diff → Φ
phonon.symmetrize_force_constants()   # impose symmetries below
```

`produce_force_constants` solves the (symmetry-reduced) finite-difference
equations relating the recorded forces to Φ. `symmetrize_force_constants`
enforces the exact properties Φ must obey but the numerics only approximate:

- **Permutation symmetry:** Φ<sub>iα,jβ</sub> = Φ<sub>jβ,iα</sub> (mixed partials commute).
- **Acoustic sum rule (translational invariance):**
  Σ<sub>j</sub> Φ<sub>iα,jβ</sub> = 0. Rigidly translating the whole crystal costs
  no energy and exerts no force, so each atom's force constants must sum to zero
  over partners. This is what forces the three acoustic branches to ω → 0 as
  **q** → Γ; without it they would not close the gap to zero exactly.

### 6.2 The dynamical matrix and the eigenvalue problem

For a periodic crystal the harmonic equations of motion decouple per wavevector
**q** into the **dynamical matrix**, a 3N<sub>p</sub>×3N<sub>p</sub> Hermitian
matrix (N<sub>p</sub> = atoms in the primitive cell) built by mass-weighting Φ and
Fourier-transforming over lattice vectors **R**:

  D<sub>iα,jβ</sub>(**q**) = (1 / √(m<sub>i</sub> m<sub>j</sub>))
      Σ<sub>**R**</sub> Φ<sub>iα,jβ</sub>(**R**) e<sup>i**q**·**R**</sup>.

The masses m<sub>i</sub> come from the primitive cell (`phonopy_io` passes ASE
masses through to `PhonopyAtoms`). Solving the eigenvalue problem at each **q**,

  D(**q**) **e**<sub>ν</sub>(**q**) = ω²<sub>ν</sub>(**q**) **e**<sub>ν</sub>(**q**),

gives:

- **eigenvalues** ω²<sub>ν</sub>(**q**) — the squared phonon frequencies. The code
  reports ω in **THz**. A *negative* eigenvalue ω² < 0 is reported as an
  *imaginary* frequency (plotted negative); it signals a structure that is not at
  a true minimum along that mode (under-relaxed, or a real instability).
- **eigenvectors** **e**<sub>ν</sub>(**q**) — the polarization vectors: the
  (complex, mass-weighted) pattern of atomic motion for mode ν. There are
  3N<sub>p</sub> branches ν (9 for the 3-atom monolayer, 18 for the 6-atom
  bilayer).

### 6.3 Band structure: which q-points

A **band structure** is ω<sub>ν</sub>(**q**) sampled along a path through
high-symmetry points of the Brillouin zone. `_ase_band_path_from_structure`
(`solve.py:99`) asks ASE for the standard path and special points of the relaxed
lattice (e.g. Γ–M–K–Γ for a hexagonal TMD); `get_band_qpoints_and_path_connections`
fills `band_npoints` (=31) **q**-points per segment;
`run_band_structure(..., with_eigenvectors=True)` diagonalizes D(**q**) along it.
The plot is written to `band_structure.png`.

The reciprocal lattice used to convert fractional **q** to Cartesian
**q** (Å⁻¹) is the standard

  **b**-rows = 2π (A⁻¹)<sup>T</sup>,  with A the real-space cell

(`reciprocal_rows`, `solve.py:16`), satisfying **a**<sub>i</sub>·**b**<sub>j</sub> = 2π δ<sub>ij</sub>.

### 6.4 Uniform mesh (for thermodynamics / scattering)

In parallel the code runs a Γ-centered uniform **q**-mesh
(`run_mesh`, e.g. 24×24×1 / 18×18×1) with `is_mesh_symmetry=False` so *every*
grid point is kept (full, unfolded grid), `is_gamma_center=True`. This regular
grid — not the band path — is what downstream Brillouin-zone sums need: phonon
density of states, free energy, and the mode-resolved structure factors for
ultrafast-electron-diffraction (UED) intensities (`ued_intensity.py`,
`electron_scattering_factors.py`). `expand_full_fractional_mesh` wraps **q** into
[0,1) and lexicographically sorts to a reproducible order.

### 6.5 Outputs

- **`band_structure.png`** — the phonon dispersion ω<sub>ν</sub>(**q**) along the
  k-path.
- **`eigenvector_band.h5`** — per band-path q-point: `frequencies_thz`
  [n_q, n_modes], `eigenvectors` reshaped to [n_q, n_modes, n_atoms, 3],
  `q_frac`, `q_cart_anginv`, `distances`, segment lengths, labels.
- **`eigenvector.h5`** — the same on the uniform mesh, plus `weights`, for BZ sums.
- Both carry metadata: `cell_ang`, reciprocal rows, Cartesian/fractional basis
  positions τ, masses, species, units (THz, Å⁻¹).

---

## 7. Reading the physics off the result

The band structure is the diagnostic that the whole chain — relaxation,
potential, and (for bilayers) interlayer coupling — worked:

- **Three acoustic branches** ω → 0 linearly as **q** → Γ. Guaranteed by the
  acoustic sum rule (§6.1); their slope is the sound velocity. (In a strictly 2D
  sheet the out-of-plane flexural branch is quadratic, ω ∝ q².)
- **Optical branches** at finite ω at Γ. For MoSe₂ the in-plane E″/E′ and
  out-of-plane A₁ modes land near ~7–10.5 THz; these are the Raman/IR fingerprints.
- **No (significant) imaginary modes.** A few tiny negative values near Γ are
  numerical; large or widespread imaginary branches mean the structure was
  under-relaxed (tighten `fmax`/`steps`) or the supercell is too small (raise N),
  or a genuine structural instability.
- **Interlayer modes (bilayer only)** — the signature that the interlayer model
  *engaged*. At Γ the two layers move rigidly against each other:
  - a **doubly-degenerate shear (LB/“C”) mode** ≈ 0.57 THz (≈ 19 cm⁻¹) — layers
    slide in-plane, restored only by the weak vdW shear stiffness;
  - a **breathing mode** ≈ 0.85 THz (≈ 28 cm⁻¹) — layers oscillate along the
    out-of-plane spacing.
  These low but **finite** frequencies come entirely from the off-diagonal
  E<sub>inter</sub> force constants. If they collapse to ≈ 0, the interlayer
  model did not contribute — check the `--interlayer` flags and that the input has
  `atom_types`/`layer_ids` (see Troubleshooting in `AGENTS.md`). Frequency unit
  conversion: 1 THz ≈ 33.356 cm⁻¹ ≈ 4.136 meV.

---

## 8. Assumptions and limits (what this theory does *not* capture)

- **Harmonic only.** Truncating at second order gives temperature-independent
  frequencies, infinite phonon lifetimes, and no thermal expansion. Linewidths,
  phonon–phonon scattering, and thermal conductivity need 3rd/4th-order IFCs
  (not computed here).
- **0 K equilibrium geometry.** Phonons are evaluated about the static relaxed
  structure; no zero-point or finite-T lattice expansion.
- **Quality ceiling = the MACE models.** The PES, the equilibrium interlayer
  spacing, and every frequency are only as good as the trained potentials and
  their DFT reference. The interlayer decomposition assumes adjacent-pair, additive
  coupling (no explicit 3-layer or beyond-nearest-layer terms).
- **Finite-size / finite-displacement error.** Real IFCs are truncated at the
  supercell range; one-sided d = 0.01 Å carries an O(d) anharmonic bias (use
  `is_plusminus` for ±d to cancel it). Converge N and the mesh for production runs.
- **No LO–TO splitting.** No Born effective charges / non-analytic correction is
  applied, so polar optical branches are not split at Γ. (For these 2D systems the
  3D macroscopic-field correction is in any case ill-defined and usually omitted.)

---

*Cross-references:* run instructions and flag meanings — `AGENTS.md`;
environment build, branch names, and the four implementation "gotchas" —
`SETUP_MACE_PHONON.md`; code — `repos/mlip_phonon_scattering/` and
`repos/mace-interlayer/examples/interlayer_helpers/`.
