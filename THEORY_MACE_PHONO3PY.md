# Theory — anharmonic phonons: linewidths & lifetimes from MACE + phono3py_einsum

This document explains **what each stage of the anharmonic pipeline is doing
physically and mathematically**: how third-order force constants become
three-phonon scattering, an imaginary self-energy, and finally the phonon
**linewidth γ(q)** and **lifetime τ(q)** band-path figures. It is the "why"
companion to [`AGENTS.md`](AGENTS.md) (how to *run* it) and
[`SETUP_MACE_PHONO3PY.md`](SETUP_MACE_PHONO3PY.md) (how to *build* it).

It **builds on** the harmonic theory in
[`THEORY_MACE_PHONON.md`](THEORY_MACE_PHONON.md) and does **not** re-derive it. The
harmonic doc already covers: the MACE potential and the bilayer `NLayerCalculator`
energy decomposition (E<sub>tot</sub> = Σ<sub>L</sub> E<sub>intra</sub>(L) +
Σ<sub>⟨L,L′⟩</sub> E<sub>inter</sub>(L,L′)), relaxation to the equilibrium geometry
**R⁰**, the finite-displacement (frozen-phonon) method, the second-order force
constants Φ, the dynamical matrix **D**(**q**), and the eigenvalue problem
**D**(**q**)**e**<sub>ν</sub> = ω²<sub>ν</sub>**e**<sub>ν</sub>. Read that first;
here we assume the harmonic solution {ω<sub>ν</sub>(**q**), **e**<sub>ν</sub>(**q**)}
exists and ask what the *cubic* term of the energy expansion does to it.

The physical goal: harmonic phonons live forever (real frequencies, infinite
lifetime). Real phonons **decay** — they scatter off each other through the
anharmonicity of the potential. That decay is a finite **linewidth** γ (a broadening
of each mode in frequency) and a finite **lifetime** τ. This is what sets, e.g., the
timescales seen in ultrafast electron diffraction and the lattice thermal
conductivity.

Everything below is implemented in the **`mlip_phonon_scattering.linewidth`** package
(`w_scatter_lib.py`, `gpu_scattering_W_phonons_bilayer_comm_mesh.py`,
`extract_gamma_example.py`, `linewidth_path.py`, `plot_linewidth_example.py`), with the
bilayer `mlip_phonon_scattering.interlayer` helpers alongside it, on top of the
`phono3py_einsum` GPU fork (`run_imag_self_energy(..., lang="GPU")`). It is driven by the
`mlip-linewidth-*` console commands. **Scope:** the MoSe₂/WSe₂ bilayer is validated
end-to-end and **reproduces the golden reference** (γ_max 0.020164 vs 0.020166 THz); the
MoSe₂ monolayer, on the same code path, is now **GPU-validated by self-contained physics
checks** (γ_max 0.01741 THz, 9 bands) — there is **no golden reference for it**, so it is
physics-validated, not golden-reproduced.

---

## 1. Block diagram

```
        ┌─────────────────────────────────────────────────────────────┐
        │  HARMONIC SOLUTION  (from THEORY_MACE_PHONON.md)              │
        │  fc2 Φ  →  D(q)  →  ω_ν(q), e_ν(q)   on a 6×6×1 phonon cell   │
        └───────────────┬─────────────────────────────────────────────┘
                        │  frequencies + eigenvectors feed every matrix element
   R⁰ (relaxed)         ▼
 ┌──────────────┐  ┌─────────────────────────────────┐  linewidth/forces3
 │ STAGE A       │  │ STAGE B — 3rd-ORDER IFCs         │  MACE forces on
 │ relax (MACE)  │─►│ Φ₃ = ∂³E/∂u∂u∂u by finite disp   │  displaced 3×3×1
 │ relax.py      │  │ on a SMALL fc3 supercell (3×3×1) │  supercells (MPI)
 └──────────────┘  └───────────────┬─────────────────┘
                                   │  fc3 (symmetrized), fc2 (6×6×1)
                                   ▼   phonon_cache_m36x36x1/{fc3,fc2}.npy
                   ┌─────────────────────────────────────────────┐  linewidth/
                   │ STAGE C — THREE-PHONON SCATTERING (GPU)       │  gpu_scattering_
                   │ bubble self-energy Im Σ on a 36×36×1 q-mesh   │  ..._comm_mesh.py
                   │ real→reciprocal→normal fc3 transform (cupy)   │  lang="GPU"
                   │ per-triplet gamma_detail[t,b0,b1,b2]          │  run_imag_self_energy
                   └───────────────┬─────────────────────────────┘
                                   │  gamma_detail-*.hdf5  (+ optional W matrix)
                                   ▼
                   ┌─────────────────────────────────────────────┐  linewidth/
                   │ STAGE D — EXACT BZ REDUCTION                  │  extract_gamma_example
                   │ γ(q0,b0)=Σ_t w·Σ_{b1,b2} gamma_detail        │  → gamma_W_full_T50.npz
                   └───────────────┬─────────────────────────────┘
                                   │  γ(q,b) in THz on the full mesh
                                   ▼
                   ┌─────────────────────────────────────────────┐  linewidth/
                   │ STAGE E — BAND-PATH FIGURES                   │  plot_linewidth_example
                   │ map γ onto Γ–K–M–Γ by joint (q,ω) interp;     │  (linewidth_path.py)
                   │ τ = 1/(4πγ);  γ-panel + τ-panel + meshpoints  │
                   └───────────────┬─────────────────────────────┘
                                   ▼
     mose2_wse2_aligned_linewidth_lifetime_band_path{,_meshpoints}.{png,pdf}
```

The single most important idea: **the harmonic phonons are the "particles"; the
cubic anharmonicity Φ₃ is the interaction that lets them scatter.** The linewidth is
the imaginary part of the phonon self-energy from the lowest-order (bubble)
scattering diagram, and MACE makes the huge number of fc3 force evaluations cheap.

---

## 2. Stage A — prerequisites: the equilibrium and the harmonic modes

Identical in spirit to the harmonic pipeline. We relax to **R⁰** (max<sub>i</sub>
|**F**<sub>i</sub>| < 1.3×10⁻⁵ eV/Å here) with the bilayer `NLayerCalculator`, then
solve the harmonic problem to get {ω<sub>ν</sub>(**q**), **e**<sub>ν</sub>(**q**)}.
Two reasons these are needed *before* any anharmonic work:

1. The cubic expansion below is only meaningful about a force-free minimum (the
   linear term must vanish; see harmonic doc §3). If fc2 has imaginary modes, the
   harmonic quasiparticles are ill-defined and the on-shell self-energy below is
   meaningless.
2. The scattering matrix elements are built from the harmonic **eigenvectors**
   **e**<sub>ν</sub> and evaluated at the harmonic **frequencies** ω<sub>ν</sub>. So
   the same fc2 that gives the dispersion also enters the self-energy. The pipeline
   builds fc2 on the **6×6×1 phonon supercell** (via the 2nd-from-3rd force set)
   precisely so the harmonic input to the scattering is well converged.

For the bilayer the relaxation uses E<sub>tot</sub> = E<sub>MoSe₂</sub> +
E<sub>WSe₂</sub> + E<sub>inter</sub>; the interlayer term sets the equilibrium
spacing/registry and the restoring forces behind the shear and breathing modes.

---

## 3. Stage B — third-order interatomic force constants Φ₃

**Physics.** Continue the energy expansion of the harmonic doc (§4) one order in the
displacements **u** = **R** − **R⁰**:

  E = E₀ + ½ Σ<sub>iα,jβ</sub> Φ<sub>iα,jβ</sub> u<sub>iα</sub> u<sub>jβ</sub>
        + (1/3!) Σ<sub>iα,jβ,kγ</sub> Φ₃<sub>iα,jβ,kγ</sub> u<sub>iα</sub> u<sub>jβ</sub> u<sub>kγ</sub> + …

(the linear term is zero at **R⁰**). The **third-order IFCs**

  Φ₃<sub>iα,jβ,kγ</sub> = ∂³E / ∂u<sub>iα</sub> ∂u<sub>jβ</sub> ∂u<sub>kγ</sub>

measure how the force on one atom changes as *two* others are displaced —
equivalently, how the harmonic force constant Φ<sub>iα,jβ</sub> itself shifts under a
third displacement. Φ₃ is the leading anharmonic coupling and the entire source of
finite phonon lifetimes at this order.

**Method — finite displacements on a smaller supercell.** phono3py measures Φ₃ by
displacing atoms *in pairs* (double displacements ±d on the fc3 supercell) and
finite-differencing the forces,

  Φ₃<sub>iα,jβ,kγ</sub> ≈ −Δ²F<sub>iα</sub> / (Δu<sub>jβ</sub> Δu<sub>kγ</sub>).

The number of symmetry-inequivalent pair displacements grows steeply with cell size
— far faster than the single displacements of fc2. So the pipeline uses a **small
3×3×1 fc3 supercell** (`mlip-linewidth-phono3py-yaml --supercell 3 3 1`) while
keeping the **larger 6×6×1 phonon supercell** for fc2 (`--phonon-supercell 6 6 1`).
MACE supplies forces for *all* of these displaced supercells cheaply
(`mlip-linewidth-forces3` for fc3, `mlip-linewidth-forces2-from3` for
the matching fc2), which is exactly what makes an anharmonic calculation with
DFT-quality forces tractable. For the bilayer these forces come from the summed
intra+inter potential, so Φ₃ contains genuine anharmonic interlayer coupling.

Stage 4 of the pipeline then symmetrizes (`produce_fc3(symmetrize_fc3r=True)`,
`produce_fc2(symmetrize_fc2=True)`, inside `set_phono3py_forces_with_mesh_fc_cache`, run
by `mlip-linewidth-cache-fc`): it imposes the permutation symmetry of Φ₃ (symmetric under
any exchange of the three (atom, direction) indices) and the acoustic sum rule, then
caches `phonon_cache_m36x36x1/{fc3,fc2}.npy`.

---

## 4. Stage C — three-phonon scattering and the imaginary self-energy

### 4.1 What Φ₃ does to phonons

Rewrite Φ₃ in the phonon (normal-mode) basis. Fourier-transforming over lattice
vectors and contracting with the mass-weighted eigenvectors turns the real-space Φ₃
into a **three-phonon matrix element** Φ(λ₀,λ₁,λ₂), where each label λ = (**q**, b)
is a wavevector plus band:

  Φ(λ₀,λ₁,λ₂) ∝ Σ<sub>iα,jβ,kγ</sub>
      Φ₃<sub>iα,jβ,kγ</sub> e<sup>λ₀</sup><sub>iα</sub> e<sup>λ₁</sup><sub>jβ</sub> e<sup>λ₂</sup><sub>kγ</sub>
      / √( m<sub>i</sub> m<sub>j</sub> m<sub>k</sub> ω<sub>λ₀</sub> ω<sub>λ₁</sub> ω<sub>λ₂</sub> ) × (lattice phases).

Because it is cubic, it couples modes *three at a time*: a phonon λ₀ can **decay
into** two phonons (λ₀ → λ₁ + λ₂) or **coalesce** with one to form another
(λ₀ + λ₁ → λ₂). Two conservation laws select the allowed processes:

- **Crystal-momentum conservation** (from lattice periodicity):
  **q₀** = **q₁** + **q₂** + **G**, with **G** a reciprocal-lattice vector
  (**G** = 0 → Normal process, **G** ≠ 0 → Umklapp). The allowed (λ₀,λ₁,λ₂) are the
  **triplets** of the mesh — this is what `gamma_detail` is indexed over.
- **Energy conservation** (on the mass shell), enforced by the δ-functions below.

### 4.2 The bubble self-energy → linewidth γ

The lowest-order (bubble) diagram gives a complex phonon self-energy
Σ<sub>λ₀</sub>(ω); its **imaginary part is the linewidth**. In phono3py's form,
evaluated on-shell at ω = ω<sub>λ₀</sub>:

  γ<sub>λ₀</sub>(ω) ∝ Σ<sub>λ₁,λ₂</sub> |Φ(λ₀,λ₁,λ₂)|²
      × { (n₁ + n₂ + 1)·δ(ω − ω₁ − ω₂)                        ← decay
        + (n₁ − n₂)·[ δ(ω − ω₁ + ω₂) − δ(ω + ω₁ − ω₂) ] }     ← coalescence

where n<sub>i</sub> = n(ω<sub>i</sub>, T) = 1/(e<sup>ℏω<sub>i</sub>/k<sub>B</sub>T</sup> − 1)
is the Bose–Einstein occupation. The two channels are exactly the decay and
coalescence processes; the occupation factors carry all the **temperature
dependence** (here T = 50 K). The δ-functions impose energy conservation and are
evaluated by **tetrahedron integration** over the mesh — the GPU path uses
`sigma=None` (no Gaussian smearing; `compute_gamma_detail_on_mesh_gpu` sets
`ph3.sigmas = [None]`).

The engine calls `ph3.run_imag_self_energy(grid_points=..., temperatures=[T],
frequency_points_at_bands=True, write_gamma_detail=True, lang="GPU")`. It does **not**
collapse the sum: it writes the **per-triplet, per-band-triple** contribution

  gamma_detail[σ=0, t, b₀, b₁, b₂]

to a per-grid-point file (e.g. `gamma_detail-m36361-g0.hdf5`) inside the
temperature-tagged cache dir `phono3py_cache_36x36x1_T50.0K/`, one file per irreducible
grid point.
`frequency_points_at_bands=True` (required under `lang="GPU"`) means the self-energy
is evaluated at the band frequencies ω<sub>λ₀</sub> themselves — the on-shell
linewidth.

### 4.3 The GPU einsum engine

The fork's `lang="GPU"` path is what makes the |Φ(λ₀,λ₁,λ₂)|² evaluation fast. It
performs the fc3 interaction transform in two contractions —
**real→reciprocal** (Fourier transform of the real-space Φ₃ over lattice vectors)
and **reciprocal→normal** (contraction with the three mass-weighted eigenvectors) —
as CuPy `einsum` calls on the GPU, adding this backend on top of stock phono3py's
C/Python paths. `init_phph_interaction(symmetrize_fc3q=True)` symmetrizes the
interaction in reciprocal space. Work is parallelized by splitting the
**irreducible** grid points across MPI ranks (one GPU per rank,
`remaining_gps[rank::size]`); each rank writes its own `gamma_detail` files. The proven
run geometry is `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`); `srun -n 16`
hangs (16-rank mpi4py/PMI deadlock), as documented in
[`AGENTS.md`](AGENTS.md) / [`SETUP_MACE_PHONO3PY.md`](SETUP_MACE_PHONO3PY.md).

The engine can optionally assemble a symmetric state-to-state scattering matrix
`W_rate` (and `W_tau = 1/(4π·W_rate)`) on rank 0, but that is a large dense object
(guarded by `--max-w-gb`); the band-path figures need only `gamma_detail`, so the
bilayer run keeps it and reduces it in Stage D.

---

## 5. Stage D — the exact linewidth by Brillouin-zone reduction

The canonical linewidth of a mode (q₀, b₀) is the **exact** BZ sum over all
scattering triplets and partner bands — *not* an interpolation:

  γ(q₀, b₀) = Σ<sub>triplets t</sub> weight(t) · Σ<sub>b₁,b₂</sub> gamma_detail[0, t, b₀, b₁, b₂]

implemented by `mlip_phonon_scattering.linewidth.w_scatter_lib` and driven by the
`mlip-linewidth-extract-gamma` step. The
weight(t) are the triplet multiplicities of the symmetry-reduced mesh; summing over
b₁, b₂ closes the two partner-band sums. The result is expanded from the irreducible
grid to the full 36×36×1 grid and saved as **`gamma_W_full_T50.npz`**: `gamma_W_full`
of shape (num_grid, nband) = (1296, 18) in **THz**, plus `qpoints_frac`, `nband`,
`mesh_numbers`, `temperature_K`.

The extractor **enforces the physics acceptance criteria at write time**: it raises
if γ is not finite, and raises if any γ < −1e-9. So a `gamma_W_full_T50.npz` that was
produced at all is already guaranteed finite and non-negative.

> **Why this, and not the dn/dt collision diagonal.** There is a linearized
> collision-operator diagonal (a γ₀ from a dn/dt route) that is *not* used for
> linewidths: it diverges for Bose-enhanced soft-partner modes. The canonical γ is
> the imaginary-self-energy / HWHM `gamma_detail` reduction above (recorded project
> decision `feedback-linewidth-definition`; see the
> `mlip_phonon_scattering.linewidth.linewidth_path` docstring).

---

## 6. Stage E — linewidth γ and lifetime τ, and the band-path figures

### 6.1 From γ to τ — the exact convention (pinned from the code)

The lifetime is (`mlip_phonon_scattering.linewidth.linewidth_path.lifetime_from_gamma`,
and identically the engine's `W_tau`):

  **τ [ps] = 1 / (2 · 2π · γ) = 1 / (4π · γ)**,  with γ in THz,
  and **τ = +∞ where γ ≤ 0.**

The two factors are physical, not fudge:

- the **2** turns the **HWHM** imaginary self-energy γ into the full width
  FWHM = 2γ (the actual spectral broadening of the mode);
- the **2π** converts ordinary-frequency THz to **angular** frequency, so 1/ω comes
  out directly in ps (since 1 THz = 1 ps⁻¹).

Numerically, γ = 0.01 THz ⇒ τ ≈ 7.96 ps; γ = 0.1 THz ⇒ τ ≈ 0.80 ps. A γ = 0 mode
(e.g. the acoustic branches as **q** → Γ, which have no allowed decay channel)
correctly has infinite lifetime.

### 6.2 Map mesh γ onto the Γ–K–M–Γ path

Two figures are produced, both by `mlip_phonon_scattering.linewidth.linewidth_path`:

- **Interpolated.** The mesh γ lives on the regular 36×36×1 grid; the band path is a
  different set of q-points. Each mesh mode is treated as a sample point
  (q<sub>x</sub>, q<sub>y</sub>, ω) → γ, and γ is evaluated on the path by
  **inverse-distance interpolation in a scaled (q, ω) metric**:

    d² = |Δ**q**|²/σ<sub>q</sub>² + (Δω)²/σ<sub>ω</sub>²,   w<sub>i</sub> = 1/d<sub>i</sub><sup>p</sup>.

  The wavevector scale σ<sub>q</sub> is one mesh spacing; the frequency scale
  σ<sub>ω</sub> is half the median adjacent-branch gap on the mesh. Matching in
  *joint* (q, ω) space — rather than by band index — is robust across band crossings,
  and the weighting is *interpolating*: it returns the exact mesh value where a path
  point coincides with a mesh point (e.g. at K and M). **Crucially, it is γ that is
  interpolated — never τ = 1/γ, which would diverge for soft modes.** Mesh and path
  frequencies both come from the *same* cached fc2, which is what makes the
  interpolation exact at K and M.

- **Exact meshpoints.** Only mesh grid points (and their ±1 periodic images) whose
  Cartesian **q** lies *exactly* on a Γ–K–M–Γ segment (perpendicular distance < tol)
  are plotted, with their **raw** γ — no interpolation. This is the ground-truth
  cross-check of the smooth figure.

Each figure is a two-panel dispersion: the top panel colors modes by **linewidth γ
(THz)**, the bottom by **lifetime τ = 1/(4πγ) (ps)**. The soft acoustic region below
`min_freq_thz = 0.1` THz is masked (there γ → 0, τ → ∞).

The `mlip-linewidth-plot` command (`plot_linewidth_example`) is the thin CLI (step 7):
it takes `--yaml phono3py_disp.yaml`, `--fc2 phonon_cache_m36x36x1/fc2.npy`, `--gamma-w
gamma_W_full_T50.npz`, `--mesh 36 36 1` and writes both figures to `figures/`.

---

## 7. Reading the physics off the result

The linewidth/lifetime dispersion is the diagnostic that the anharmonic chain worked
and tells you *how* modes decay. **Read the gray harmonic skeleton first** — 18
branches (6 atoms), three acoustic modes → 0 at Γ, a doubly-degenerate interlayer
shear pair near 0.575 THz and a breathing mode near 0.853 THz. If those interlayer
modes collapse to zero, the linewidth is being built on the wrong bilayer
calculator. Then read the color:

- **γ → 0 for the acoustic branches near Γ.** Long-wavelength acoustic phonons have
  vanishing phase space and matrix elements for three-phonon decay, so their
  linewidth collapses and their lifetime diverges. This is physical, and it is why
  the near-Γ soft region is masked in the figures (τ = 1/(4πγ) → ∞). A finite, large
  γ for an acoustic mode *at* Γ would signal a bug or an under-relaxed structure.
- **Larger γ (broader, shorter-lived modes) where scattering phase space is rich** —
  typically optical branches and near zone-boundary points K, M, where many energy-
  and momentum-conserving triplets are available. Read the γ-panel as "how fast this
  mode decays" and the τ-panel (its reciprocal, ×1/4π) as "how long it lives." A
  large matrix element alone does *not* guarantee a large γ — a partner pair must
  exist that conserves both energy and momentum.
- **Interlayer modes (bilayer).** The low-frequency shear (~0.57 THz) and breathing
  (~0.85 THz) modes acquire *finite* linewidths here through anharmonic interlayer
  coupling in Φ₃ — a direct signature that the interlayer anharmonicity engaged.
- **Temperature.** Every γ carries the Bose factors n(ω, T); at T = 50 K
  high-frequency partner modes are barely occupied, so decay is dominated by
  spontaneous emission ((n₁ + n₂ + 1) → 1) rather than stimulated coalescence.
  Raising T broadens modes (larger γ, shorter τ) and requires a fresh gamma-detail
  cache and a complete new self-energy evaluation — you cannot rescale the 50 K
  numbers.
- **The `_meshpoints` figure should agree with the smooth figure** wherever a mesh
  point lands on the path (exactly at K and M). Disagreement there means the
  interpolation or the frequency consistency is off.

Unit reminders (from the harmonic doc): 1 THz ≈ 33.356 cm⁻¹ ≈ 4.136 meV.

---

## 8. Assumptions and limits (what this theory does *not* capture)

- **Cubic anharmonicity only, lowest order.** γ is the *bubble* (two-phonon)
  self-energy from Φ₃. Four-phonon scattering (Φ₄), higher-order diagrams, and the
  real part of the self-energy (the frequency *shift*) are not included. Strongly
  anharmonic or high-T regimes need more.
- **On-shell, perturbative linewidth.** γ is evaluated at the harmonic frequency
  ω<sub>λ₀</sub> (`frequency_points_at_bands=True`) and assumes γ ≪ ω (well-defined
  quasiparticles). It is not a self-consistent or full spectral-function treatment;
  where γ approaches ω or the neighboring branch spacing, the simple Lorentzian
  lifetime picture breaks down.
- **Harmonic modes as the basis.** Frequencies and eigenvectors are the fc2 (0 K,
  static-geometry) ones; the theory inherits every harmonic limit from
  `THEORY_MACE_PHONON.md` (0 K equilibrium, no thermal expansion, no LO–TO
  splitting) and adds anharmonic decay on top of them.
- **Finite mesh and finite supercells.** The BZ sums are on a 36×36×1 mesh and the
  δ-functions by tetrahedron integration; Φ₃ is truncated at the 3×3×1 fc3 supercell
  range and Φ at the 6×6×1 range. Converge mesh and supercells for production numbers.
- **Quality ceiling = the MACE models.** Φ₃, the equilibrium interlayer spacing, and
  every γ are only as good as the trained potentials and their DFT reference, and the
  additive adjacent-pair interlayer decomposition.
- **Intrinsic phonon–phonon lifetime only.** Isotope disorder, defects, boundaries,
  substrates, electron–phonon coupling, and experimental resolution are absent.
- **Dense W memory.** The optional state-to-state W matrix scales as (N_ir·n_band)²
  and can become the memory bottleneck; the figures need only `gamma_detail`.
- **Interpolation creates no new exact data.** Only the mesh γ values and the
  `_meshpoints` figure are direct samples of the self-energy calculation; the smooth
  figure is a joint-(q, ω) interpolation of those samples.
- **Lifetime convention is fixed.** All reported lifetimes use τ = 1/(4πγ) with γ as
  HWHM in THz. Results quoted as 1/(2πγ), 1/(2γ), or 1/γ follow different conventions
  and are not directly comparable without conversion.
- **Both cases validated; only the bilayer has a golden reference.** The MoSe₂/WSe₂
  bilayer linewidth path is validated end-to-end and reproduces its golden reference
  (γ_max 0.020164 vs 0.020166 THz). The MoSe₂ monolayer is **implemented on the same
  code path** (`examples/mose2_monolayer/`, single-model flags) and is now
  **GPU-validated by self-contained physics checks** (γ_max 0.01741 THz, 9 bands, fc2
  clean, τ > 0); it has **no golden reference**, so it is physics-validated, not
  golden-reproduced. Converge mesh and supercells before quoting production monolayer
  numbers (see the monolayer subsection in `AGENTS.md`).

---

*Cross-references:* harmonic foundations (fc2, dynamical matrix, eigenvectors,
interlayer decomposition) — [`THEORY_MACE_PHONON.md`](THEORY_MACE_PHONON.md); how to
run the pipeline, flags, and physics sanity checks — [`AGENTS.md`](AGENTS.md);
environment build, branch names, and the phono3py_einsum recipe —
[`SETUP_MACE_PHONO3PY.md`](SETUP_MACE_PHONO3PY.md); code — the
`mlip_phonon_scattering.linewidth` package (with the bilayer
`mlip_phonon_scattering.interlayer` helpers) and the `phono3py_einsum` fork.
