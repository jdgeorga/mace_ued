No files were changed, no write-producing commands were run, and no jobs or QE binaries were launched.

The main conclusion is: the 2D correction should be implemented as a literal, mass-weighted port of QE’s `rgd_blk(...,loto_2d=.true.)` kernel. The corrected dynamical matrix must be constructed on the CPU before diagonalization; the existing GPU ph-ph path will then consume the same corrected frequencies/eigenvectors without any GPU dynamical-matrix changes. For the fastest scientifically reliable production result, however, I recommend generating the full 36×36×1 phonon mesh with QE `matdyn` and injecting it through the existing `set_phonon_data()` API first.

## 1. Dynamical-matrix path

### Call path

The active high-level path is:

1. `Phono3py.init_phph_interaction()` creates the interaction object and then initializes its dynamical matrix:
   - [`api_phono3py.py:1188–1278](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/api_phono3py.py:1188)
   - [`api_phono3py.py:3151–3165](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/api_phono3py.py:3151)

2. Both interaction implementations currently call phonopy’s `get_dynamical_matrix(...)`:
   - standard path: [`interaction.py:626–651](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:626)
   - fork/fast path: [`interaction_fast.py:701–724](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:701)

3. The grid solver is then invoked:
   - standard interaction dispatch: [`interaction.py:685–700](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:685), [`interaction.py:1041–1060](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:1041)
   - fast interaction dispatch: [`interaction_fast.py:753–764](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:753), [`interaction_fast.py:1076–1088](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:1076)

4. In `phono3py/phonon/solver.py`:
   - the C solver constructs the matrices inside `_phononcalc` and either diagonalizes there or with `numpy.linalg.eigh`: [`solver.py:19–163](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:19)
   - the Rust solver builds all matrices and batch-diagonalizes them: [`solver.py:259–327](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:259)
   - the one-q Python path calls `dynamical_matrix.run(q)` and then `numpy.linalg.eigh`: [`solver.py:330–359](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:330)

### Where the eigenvectors go

The resulting host NumPy arrays are exactly those consumed by the fc3 contraction:

- C and Rust CPU interaction kernels receive `_frequencies` and `_eigenvectors` directly: [`interaction.py:977–999](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:977), [`interaction.py:1016–1037](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:1016)
- the Python reference contraction selects the same arrays and contracts them with fc3: [`reciprocal_to_normal.py:140–172](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/reciprocal_to_normal.py:140)

### GPU behavior

The CuPy path does not build or diagonalize a harmonic dynamical matrix on-device.

It first solves needed phonons through the host Python solver and then passes the host frequency/eigenvector arrays to the GPU transform: [`interaction_fast.py:1245–1281](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:1245), [`interaction_fast.py:1291–1314](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:1291). `ReciprocalToNormalSquaredGPU` merely selects those arrays, uploads them with `cp.asarray`, mass-weights them, and contracts with fc3: [`reciprocal_to_normal_gpu.py:22–85](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/reciprocal_to_normal_gpu.py:22), [`reciprocal_to_normal_gpu.py:210–270](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/reciprocal_to_normal_gpu.py:210). `real_to_reciprocal_gpu.py` only transfers and Fourier-transforms fc3-related data: [`real_to_reciprocal_gpu.py:109–146](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/real_to_reciprocal_gpu.py:109).

### The single physical injection point

The 2D physics should live in one builder, for example:

```text
DynamicalMatrixQELoto2D.get_dynamical_matrices(qpoints)
    = D_short_range(qpoints)
    + D_QE_loto2d(qpoints)
```

Both the class’s one-q `run(q)` and the grid solver should call that same function.

A subclassed `run(q)` alone is not sufficient: the C/Rust grid solvers bypass virtual `run(q)` and construct the matrices directly. Moreover, the current C solver interprets any generic `DynamicalMatrixNAC` as the built-in 3D NAC unless it is specifically Gonze: [`solver.py:71–105](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:71), [`solver.py:123–150](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:123). Therefore the C and Rust solver functions need an early explicit `DynamicalMatrixQELoto2D` branch.

Once that branch fills `_frequencies` and `_eigenvectors`, harmonic output and every CPU/GPU fc3 contraction are mutually consistent.

## 2. 3D versus 2D NAC physics

### What phonopy currently implements

Phonopy’s NAC schema contains Born charges, dielectric tensor, a unit factor, and either `gonze` or `wang`: [`dynamical_matrix.py:55–93](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:55). The factory defaults to Gonze and only dispatches Wang explicitly: [`dynamical_matrix.py:1157–1212](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:1157).

The short-range matrix itself is the usual force-constant Fourier transform with mass normalization and phase \(\exp(+2\pi i\,\mathbf q\cdot\mathbf R)\): [`dynamical_matrix.py:303–349](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:303).

For Gonze-Lee, phonopy:

1. evaluates the reciprocal dipole-dipole term at the force-constant commensurate points,
2. subtracts it and inverse-Fourier-transforms the result to short-range force constants,
3. interpolates the short-range matrix,
4. adds the reciprocal dipole-dipole term back at arbitrary q.

That subtraction is in [`dynamical_matrix.py:621–665](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:621); the add-back is in [`dynamical_matrix.py:699–757](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:699).

Schematically, its 3D reciprocal term is

\[
D_{\kappa\alpha,\kappa'\beta}^{3D,\mathrm{LR}}(\mathbf q)
=
\frac{4\pi e^2}{\Omega\sqrt{M_\kappa M_{\kappa'}}}
\sum_{\mathbf G}
\frac{
 [K_i Z^*_{\kappa,i\alpha}]
 [K_j Z^*_{\kappa',j\beta}]
}{
 \mathbf K^\mathsf{T}\epsilon_\infty\mathbf K
}
e^{-\mathbf K^\mathsf{T}\epsilon_\infty\mathbf K/(4\Lambda^2)}
e^{2\pi i\mathbf G\cdot(\tau_\kappa-\tau_{\kappa'})}
-\text{on-site drift},
\]

with \(\mathbf K=\mathbf q+\mathbf G\). The actual C implementation forms \(K_iK_j/(K^\mathsf T\epsilon K)\) with the Gaussian Ewald factor at [`dynmat.c:606–663](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/c/dynmat.c:606), uses the C-type phase containing \(\mathbf G\), not \(\mathbf q+\mathbf G\), at [`dynmat.c:679–703](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/c/dynmat.c:679), contracts with both Born tensors at [`dynmat.c:726–768](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/c/dynmat.c:726), and multiplies by \(4\pi/\Omega\) at [`dynamical_matrix.py:777–832](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:777).

Phonopy normally omits the real-space complement; the code says “no real space sum” at [`dynamical_matrix.py:668–675](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:668). An optional full-term implementation contains the erfc-based real-space functions at [`dynamical_matrix.py:870–995](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:870).

The Wang reference path instead evaluates

\[
\frac{4\pi e^2}{\Omega}
\frac{(\mathbf q\cdot Z^*_\kappa)\otimes
      (\mathbf q\cdot Z^*_{\kappa'})}
     {\mathbf q^\mathsf T\epsilon_\infty\mathbf q},
\]

and distributes that block over the force constants before their Fourier transform: [`dynamical_matrix.py:1080–1154](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:1080). There is no 2D screening or slab cutoff in that branch.

Because numerator and denominator are both \(O(q^2)\), standard 3D NAC has a finite, direction-dependent \(q\rightarrow0\) limit. Phonopy implements this through `q_direction`; at exact zero without a direction it drops NAC: [`dynamical_matrix.py:408–437](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:408).

### QE’s actual `loto_2d` term

QE’s `rgd_blk` restricts the reciprocal sum to periodic directions; because the reference IFC mesh is 6×6×1, `nr3==1` sets the \(G_3\) range to zero: [`rigid.f90:99–126](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:99).

In physical units its term is

\[
\Phi^{2D,\mathrm{LR}}_{\kappa\alpha,\kappa'\beta}(\mathbf q)
=
\frac{2\pi e^2}{A}
\sum_{\mathbf G_\parallel}
\frac{
e^{-K^2/(4\alpha)}
}{
K\,[1+K\,\hat{\mathbf K}_\parallel^\mathsf T
       r_{\rm eff}
       \hat{\mathbf K}_\parallel]
}
[K_i Z^*_{\kappa,i\alpha}]
[K_j Z^*_{\kappa',j\beta}]
e^{i\mathbf K\cdot(\tau_\kappa-\tau_{\kappa'})}
-\delta_{\kappa\kappa'}\Phi_{\kappa}^{\rm onsite},
\]

where

\[
r_{\rm eff}=\frac{c}{2}\left(\epsilon_{\infty,\parallel}-I\right),
\qquad
A=|\mathbf a_1\times\mathbf a_2|,
\qquad
c=\Omega/A
\]

for the aligned slab cell. The exact area prefactor and \(r_{\rm eff}\) construction are in [`rigid.f90:130–142](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:130); the \(1/[K(1+r_{\rm eff}K)]\) kernel is at [`rigid.f90:151–170](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:151) and [`rigid.f90:200–218](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:200). QE’s q-independent on-site subtraction, including its explicit Hermitian symmetrization, is at [`rigid.f90:172–193](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:172); the \(G+q\) pair term is at [`rigid.f90:196–239](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:196).

For the \(G=0\) term,

\[
(\mathbf q\cdot Z^*)^2=O(q^2),\qquad
v^{2D}(q)\sim\frac{1}{q(1+r_{\rm eff}q)},
\]

so

\[
\omega_{\rm LO}^2-\omega_{\rm TO}^2
\propto \frac{q}{1+r_{\rm eff}q}.
\]

Thus LO and TO are degenerate at Γ, while the LO branch has a finite slope. For a nonzero optical frequency, \(\omega_{\rm LO}-\omega_{\rm TO}\) is also linear to leading order. This is the characteristic 2D result described by [Sohier et al.](https://arxiv.org/abs/1612.07191); the Coulomb-cutoff/periodic-image framework is developed in the cited [Sohier–Calandra–Mauri implementation paper](https://arxiv.org/abs/1705.04973).

One requested premise needs refinement: the specific QE `matdyn` kernel being reproduced does not contain an explicit erf/erfc expression. `rigid.f90` states that only the reciprocal-space Ewald term is implemented and \(\alpha\) must make the real-space contribution negligible: [`rigid.f90:18–23](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:18). Its range separation is represented by the reciprocal Gaussian \(\exp[-K^2/(4\alpha)]\). A formal Ewald derivation can express the complementary real-space piece with erfc, but reproducing QE means porting this Gaussian reciprocal implementation, not introducing a different erfc kernel.

The \(c\) entering \(r_{\rm eff}\) is the periodic cell height, not an independent material thickness. For a properly converged isolated-slab dielectric calculation, \(\epsilon_\parallel-1\sim1/c\), so \(r_{\rm eff}\) remains physical as the vacuum is enlarged. Standard 3D NAC instead uses \(4\pi/\Omega\), sums over \(G_z\), yields a finite Γ splitting, and retains spurious electrostatic coupling between periodically repeated slabs.

## 3. Inputs needed

The complete QE 2D term needs:

| Input | Reference source | Notes |
|---|---|---|
| Short-range IFC/fc2 | `ifc.q2r.xml` | Must already have the same long-range term subtracted by q2r. |
| Atomic positions and masses | `ifc.q2r.xml` geometry header | Required for phase factors and mass normalization. |
| Cell, in-plane area, perpendicular repeat \(c\) | `CELL_DIMENSIONS`, `AT`, volume | No separate physical layer thickness is required. |
| Full Born tensors \(Z^*_\kappa\) | `<ZSTAR>/<Z_AT_.n>` | All tensor components should be retained. |
| \(\epsilon_\infty\) | `<EPSILON>` | QE `loto_2d` uses the in-plane 2×2 block for \(r_{\rm eff}\). |
| Ewald \(\alpha\) | `<alpha_ewald>` | Needed for exact finite-q agreement, not just the \(q\to0\) asymptote. |
| Periodic directions | `<MESH_NQ1_NQ2_NQ3>` | Here 6×6×1 implies \(G_z=0\). |
| QE unit factor | calculator metadata | For QE, phonopy uses `nac_factor=2.0`, consistent with Rydberg atomic units. |

The IFC header contains:

- `CELL_DIMENSIONS` with \(a_{\rm lat}=6.226736864532122\) bohr and the third `AT` vector scale \(12.13943139553410\): [`ifc.q2r.xml:8–16](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/ifc.q2r.xml:8). This gives \(c=75.589\) bohr, approximately 40.0 Å.
- `<UNIT_CELL_VOLUME_AU>2538.1104469300121</UNIT_CELL_VOLUME_AU>`: [`ifc.q2r.xml:22](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/ifc.q2r.xml:22).
- dielectric tensor approximately
  \[
  \epsilon_\infty=\operatorname{diag}
  (5.8788441534,\;5.8788441519,\;1.2817570751):
  \]
  [`ifc.q2r.xml:37–42](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/ifc.q2r.xml:37).
- `<Z_AT_.1>` through `<Z_AT_.6>`: [`ifc.q2r.xml:43–74](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/ifc.q2r.xml:43). Their in-plane diagonal values are approximately
  \[
  -1.77642705,\;0.90083347,\;0.89895201,\;
  -1.21360026,\;0.60279676,\;0.58744507.
  \]
- `<MESH_NQ1_NQ2_NQ3>6 6 1</...>` and `<alpha_ewald>0.98211261577626741</alpha_ewald>`: [`ifc.q2r.xml:76–80](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/ifc.q2r.xml:76).

`mose2_wse2.dyn1.xml` contains the same `<EPSILON>` and unprojected `<ZSTAR>` tags: [`mose2_wse2.dyn1.xml:37–74](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/mose2_wse2.dyn1.xml:37). Its atom-1 in-plane charge is \(-1.77666767\), slightly different from the IFC-header value. That difference is expected: q2r applies `zasr='crystal'` to the Born charges before writing the IFC header. For reproduction of the supplied `matdyn` run, use the post-q2r Born charges in `ifc.q2r.xml`, not the original Γ XML values.

The input flags are exactly:

- `q2r`: `zasr='crystal'`, `loto_2d=.true.`: [`q2r.in:1–6](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/q2r.in:1)
- `matdyn`: `asr='crystal'`, `loto_2d=.true.`: [`matdyn.in:1–9](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/matdyn.in:1)

QE q2r chooses \(\alpha=(a_{\rm lat}/2\pi)^2\), subtracts `rgd_blk` with sign \(-1\), and Fourier-transforms the result: [`do_q2r.f90:219–267](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/do_q2r.f90:219). `matdyn` reads \(\alpha\), Born charges, dielectric tensor and IFC, builds the short-range matrix, and adds `rgd_blk` with sign \(+1\): [`matdyn.f90:345–386](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/matdyn.f90:345), [`matdyn.f90:1235–1281](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/matdyn.f90:1235).

Phonopy’s current `nac_params` already carries `born`, `dielectric`, `factor`, and `method`: [`api_phonopy.py:683–711](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/api_phonopy.py:683). The fork should add:

```python
{
    "method": "qe_loto_2d",
    "born": ...,
    "dielectric": ...,
    "factor": 2.0,
    "loto_2d_alpha": ...,          # in inverse cell-length squared
    "loto_2d_periodic_axes": (0, 1),
    "loto_2d_normal_axis": 2,
    "loto_2d_c": ...,              # optional override; normally derived as volume/area
    "loto_2d_fc_is_short_range": True,
}
```

I would not name `loto_2d_c` “thickness”: it is the perpendicular repeat length. No additional effective material thickness is present in QE’s postprocessing formula.

## 4. Implementation plan

### A. Add one fork-local dynamical-matrix implementation

Add a module such as:

```text
phono3py/phonon/dynamical_matrix_loto_2d.py
```

with a class `DynamicalMatrixQELoto2D(DynamicalMatrix)`.

Subclassing the non-NAC base class is safer than subclassing `DynamicalMatrixNAC`, because the existing compiled dispatch treats an unrecognized NAC subclass as 3D Wang. Give the class an explicit marker/property such as `is_qe_loto_2d=True`.

Implement:

```python
run(q)
get_dynamical_matrices(qpoints)
_get_qe_loto_2d_force_constant_blocks(qpoints)
```

The last function should be the single authoritative physical kernel. Its steps should be:

1. Build the short-range matrix with phonopy’s existing no-NAC builder.
2. Convert reduced q to Cartesian reciprocal units consistent with the primitive cell.
3. Enumerate only reciprocal vectors along `loto_2d_periodic_axes`, using QE’s cutoff condition
   \[
   K^2/(4\alpha)<14.
   \]
4. Form \(A\), \(c=\Omega/A\), and \(r_{\rm eff}\).
5. Precompute and cache QE’s q-independent on-site subtraction.
6. Add the \(G+q\) term, skipping exactly \(K=0\).
7. Convert force-constant blocks to dynamical-matrix blocks with \(1/\sqrt{M_\kappa M_{\kappa'}}\).
8. Hermitianize and optionally assert the residual is below a tight tolerance.

For the reference system, the initial implementation should reject nonzero \(q_z\), a non-2D periodic-axis specification, or a tilted slab normal unless those cases are explicitly supported and tested.

### B. Match phonopy’s dynamical-matrix gauge

QE’s long-range phase contains \(q+G\): [`rigid.f90:221–233](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/rigid.f90:221). Phonopy’s built-in Gonze implementation explicitly uses the C-type phase containing only \(G\), noting that the D-type convention would use \(q+G\): [`dynmat.c:679–703](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/c/dynmat.c:679).

Therefore the literal QE term must either:

- be constructed in QE’s convention and transformed by the corresponding atom-diagonal unitary \(U_\kappa(q)=e^{\pm2\pi i q\cdot\tau_\kappa}\), or
- be written directly in phonopy’s C-type convention using the \(G\)-only phase.

Do not choose the sign from memory. Determine it from the dynamical-matrix/eigenvector overlap test in section 5, including whether complex conjugation or \(q\rightarrow-q\) is required.

### C. Add a shared fork-local factory

Add a helper such as `get_phph_dynamical_matrix(...)` and use it in both:

- [`interaction.py:626–650](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:626)
- [`interaction_fast.py:701–723](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:701)

The helper should intercept `nac_params["method"] == "qe_loto_2d"`; all other cases should continue to phonopy’s current factory unchanged.

This avoids duplicated method selection and ensures `Phono3py.dynamical_matrix` returns the same builder owned by the ph-ph interaction: [`api_phono3py.py:559–565](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/api_phono3py.py:559).

### D. Intercept the compiled grid solvers

At the top of both `run_phonon_solver_c` and `run_phonon_solver_rust`:

```python
if isinstance(dm, DynamicalMatrixQELoto2D):
    dms = dm.get_dynamical_matrices(qpoints_for_undone_grid_points)
    eigvals, eigvecs = diagonalize_dynamical_matrices(...)
    fill frequencies, eigenvectors, phonon_done
    return
```

This belongs in [`solver.py:19](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:19) and [`solver.py:166](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:166). The one-q Python solver already calls the class’s `run(q)` and needs no special physics branch.

A 36×36×1 mesh contains only 1296 matrices of size 18×18, so a vectorized NumPy correction plus batched eigensolver is negligible relative to the fc3/scattering work. There is no reason to add this kernel to CuPy initially.

### E. Handle the QE IFC correctly

The supplied IFC is already the analytic/short-range part: q2r subtracted its 2D long-range term before writing it. It must not be passed through phonopy’s Gonze short-range reconstruction, which would subtract a different 3D term again.

Phonopy’s existing QE q2r reader is for the legacy text format and explicitly warns that IFCs produced with QE Born/dielectric data have long-range contributions removed and are not directly usable with phonopy’s current NAC treatment: [`qe.py:436–473](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/interface/qe.py:436), [`qe.py:514–549](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/interface/qe.py:514).

For validation, implement or use a tightly tested XML-to-phonopy compact-fc converter that preserves:

- atom ordering,
- 6×6×1 translation indexing,
- Ry/bohr² units,
- QE’s Wigner-Seitz weights,
- the post-q2r Born charges,
- the exact cell and masses.

Also reproduce `matdyn`’s `asr='crystal'`. `matdyn` applies that projection after reading the IFC: [`matdyn.f90:560–571](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/matdyn.f90:560), with the crystal-ASR projection implemented in [`matdyn.f90:1369–1418](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/matdyn.f90:1369) and [`matdyn.f90:1904–1957](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/matdyn.f90:1904). A mismatch in ASR processing could otherwise be mistaken for a NAC error.

### F. Tests

Add:

1. a scalar literal translation of `rgd_blk` as the reference for the vectorized implementation;
2. Hermiticity and \(D(-q)=D(q)^*\), after accounting for the selected gauge;
3. exact Γ behavior: no direction-dependent finite LO–TO splitting;
4. linear small-q LO–TO slope;
5. invariance under increasing vacuum while keeping \(r_{\rm eff}\) fixed;
6. exact comparison with the 152-point QE path;
7. identical frequency/eigenvector arrays from the C-dispatch, Rust-dispatch and GPU-pre-solve paths;
8. a sampled CPU-versus-GPU fc3 interaction-strength comparison.

### Does phonopy itself need patching?

No hard phonopy patch is required. A fork-local subclass, factory, and solver branch can implement the complete ph-ph requirement.

A cleaner general-purpose integration would add `DynamicalMatrixQELoto2D`, extended `NacParams`, and a `method=="qe_loto_2d"` factory branch to [`phonopy/harmonic/dynamical_matrix.py:55](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:55) and [`phonopy/harmonic/dynamical_matrix.py:1157](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phonopy/phonopy/harmonic/dynamical_matrix.py:1157). Even then, phono3py’s solver would still need its explicit branch unless the C/Rust dynamical-matrix extensions were also extended. For this project, the fork-only implementation is the smaller and safer first implementation.

## 5. Validation plan

### A. Reproduce the existing band path

1. Parse the exact 152 crystallographic q-points from [`matdyn.in:9–161](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/matdyn.in:9). Do not parse q coordinates back from `phband.freq`, because that file prints transformed coordinates.

2. Load/convert the same `ifc.q2r.xml` short-range IFC, then apply the same `asr='crystal'` treatment.

3. Use the IFC-header \(\epsilon_\infty\), post-ZASR Born charges, \(\alpha\), masses, positions, and 6×6×1 periodicity.

4. Evaluate and diagonalize the new matrix at every path point.

5. Compare against [`phband.freq:1](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/phband.freq:1), matching branches by continuity/overlap near crossings rather than assuming sorted indices are always sufficient.

Acceptance targets:

- maximum absolute error: **<0.5 cm⁻¹**
- RMS error: **<0.1 cm⁻¹**
- investigate any systematic error above **0.1 cm⁻¹**

A literal unit- and gauge-correct port should normally do substantially better than 0.5 cm⁻¹. Five cm⁻¹ is not acceptable: it is comparable to physically relevant LO shifts and would contaminate the eigenvectors entering the scattering matrix elements.

As an especially strong checkpoint, test Γ and the original 6×6×1 commensurate q-points. Because q2r subtracted the same rigid term there, the add-back should reconstruct the original DFPT matrices to near serialization/ASR precision.

The supplied “no-LOTO” comparison is not valid reference output: `matdyn_noloto.in` lacks the required q-point count, and its output terminates with a Fortran input error: [`matdyn_noloto.in:1–4](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/matdyn_noloto.in:1), [`matdyn_noloto.out:21–38](/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect/matdyn_noloto.out:21). It should not be used to estimate the correction.

### B. Eigenvector comparison and gauge alignment

QE’s internal `dyndiag` converts the mass-weighted eigenvectors to displacements, but `matdyn` multiplies them back by \(\sqrt{M_\kappa}\) before storing/output: [`matdyn.f90:692–701](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/matdyn.f90:692). `write_eigenvectors` does the same for `matdyn.modes`: [`write_eigenvectors.f90:37–53](/pscratch/sd/j/jdgeorga/ued/q-e-epw-tdbe/PHonon/PH/write_eigenvectors.f90:37). Thus the printed modes are suitable for comparison with phonopy’s normalized dynamical-matrix eigenvectors after gauge alignment.

The comparison procedure should be:

1. Reconstruct
   \[
   D_{\rm QE}(q)=E_{\rm QE}(q)\,\omega^2(q)\,E_{\rm QE}^\dagger(q)
   \]
   from the reference modes and frequencies.

2. Test the finite set of convention transformations:
   \[
   D_{\rm ph}(q)\stackrel{?}{=}
   U(q)D_{\rm QE}(q)U^\dagger(q),
   \quad
   U_\kappa=e^{\pm2\pi i q\cdot\tau_\kappa},
   \]
   including complex conjugation or \(q\rightarrow-q\). Choose the transformation minimizing the matrix residual over several generic, non-symmetry q-points.

3. For each nondegenerate mode, match by maximum absolute overlap and multiply one vector by a global phase so the overlap is real and positive.

4. For degenerate or nearly degenerate clusters, compare subspaces:
   - singular values of \(E_{\rm ph}^\dagger E_{\rm QE}\),
   - projector difference \(E E^\dagger\),
   - optional unitary Procrustes/SVD alignment.

Suggested eigenvector criteria are \(|\langle e_{\rm ph}|e_{\rm QE}\rangle|>0.9999\) for isolated modes and minimum subspace singular value \(>0.9999\) for degenerate clusters.

Raw component-wise diffs are invalid because global mode phases and unitary rotations within a degenerate subspace are arbitrary.

### C. Extend to 36×36×1

1. Construct q-points exactly as phono3py does:
   \[
   q=\texttt{grid.addresses}\,\texttt{QDinv}^{\mathsf T},
   \]
   as seen in [`solver.py:347–351](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon/solver.py:347).

2. Evaluate all unique 36×36×1 q-points with both the new builder and QE `matdyn`.

3. Expand/check phono3py’s BZ-boundary duplicate addresses explicitly; the BZ array can exceed \(36\times36\).

4. Require the same <0.5 cm⁻¹ maximum frequency tolerance over the complete mesh and gauge-aligned eigenvector/subspace criteria.

5. Check:
   - Hermiticity;
   - reciprocal periodicity in the chosen C gauge;
   - exact Γ LO/TO degeneracy;
   - linear small-\(q\) LO slope;
   - three acoustic modes at Γ;
   - no unintended q-direction finite splitting;
   - no discontinuities caused by inconsistent BZ-boundary gauges.

6. Run one small CPU and GPU ph-ph calculation using the same stored harmonic arrays and compare sampled \(|V^{(3)}|^2\) values. Since the GPU reads the host arrays, agreement should be limited only by the existing CPU/GPU fc3 numerical tolerance.

## 6. Effort, risks, and the precomputed-QE alternative

### Fork implementation estimate

A focused implementation for this aligned 2D system should take approximately:

- **3–5 focused engineering days** for the literal NumPy kernel, fork dispatch, input plumbing and initial QE-path validation;
- **1–2 weeks** to harden it for arbitrary 2D cells, multiple unit conventions, C/Rust backend regression tests, documentation and robust IFC import.

The principal risks are:

- QE-versus-phonopy C/D Bloch-gauge and q-sign conventions;
- exact `alat`, reciprocal-vector and \(\alpha\) units;
- accidentally applying a second NAC subtraction;
- reproducing `asr='crystal'`;
- correct Γ treatment, including the nonzero-\(G\) terms and q-independent on-site subtraction;
- atom/translation ordering in the QE XML IFC;
- BZ-boundary eigenvector gauge;
- generalizing beyond an aligned slab with \(q_z=0\).

GPU performance is not a significant risk because the harmonic matrices are only 18×18 and are solved once on the CPU.

### Existing precomputed-phonon injection API

The fork already exposes:

```python
Phono3py.set_phonon_data(frequencies, eigenvectors, grid_address)
```

at [`api_phono3py.py:1280–1311](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/api_phono3py.py:1280). It explicitly supports BZ-grid arrays whose length exceeds `prod(mesh)` because of boundary points.

Both interaction implementations copy the arrays and mark all points solved:

- [`interaction.py:653–683](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction.py:653)
- [`interaction_fast.py:726–751](/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum/phono3py/phonon3/interaction_fast.py:726)

A no-fork workflow is therefore:

1. Set the 36×36×1 mesh and initialize the ph-ph interaction with `nac_params=None`. Its transient bare Γ solution will later be overwritten.
2. Obtain the exact `grid.addresses` and `QDinv`.
3. Generate a `matdyn` input containing every q represented by those addresses, or every unique q plus an exact BZ-boundary expansion.
4. Run `matdyn` with `loto_2d=.true.` and `asr='crystal'`.
5. Parse THz frequencies and mass-weighted eigenvectors.
6. Transform QE eigenvectors into phonopy’s C-type Bloch gauge using the matrix-level convention test described above.
7. Preserve QE’s orthonormal basis within unique degenerate manifolds; use subspace alignment only for comparisons or for making symmetry-equivalent/BZ-duplicate representations consistent.
8. Call:
   ```python
   ph3.set_phonon_data(freqs_thz, eigvecs_phonopy_gauge, ph3.grid.addresses.copy())
   ```
9. Assert that every `phonon_done` entry is one and that a small interaction calculation does not overwrite the injected arrays.

The atom-dependent \(e^{\pm iq\cdot\tau_\kappa}\) correction is essential. A single global phase per phonon mode cancels from \(|V^{(3)}|^2\), but an atom-dependent gauge transformation does not cancel unless the fc3 Fourier convention is transformed consistently. Injecting raw `matdyn.modes` without this correction can therefore produce wrong matrix elements even if the frequencies are perfect.

This route should take roughly **0.5–1.5 days** to implement and validate, assuming the XML/mode parsing and gauge relation are handled carefully. `matdyn` is only interpolating and diagonalizing 1296 matrices of dimension 18, so the additional QE workload is small compared with the DFPT calculation that already produced the IFC.

### Decisive recommendation

For this project, use **QE `matdyn` on the complete 36×36×1 grid and inject the gauge-corrected phonons first**.

It is faster and more robust because:

- QE is the ground-truth implementation that created the supplied short-range IFC;
- it automatically uses the same \(\alpha\), ASR, Born-charge projection, finite-q G sum and Γ treatment;
- the target mesh is small enough that precomputation is inexpensive;
- the existing interaction API already accepts complete frequency/eigenvector arrays;
- the only genuinely delicate external step is the Bloch-gauge conversion, which must also be solved to validate a fork implementation.

Implement the fork-native 2D builder as a second phase, using the full QE 36×36×1 dataset as its regression gold standard.
