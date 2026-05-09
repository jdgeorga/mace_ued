# Implementation Pass: phonopy UED q*tau gauge correction

## Goal

Apply the confirmed post-hoc phase correction whenever phonopy HDF5 eigenvectors are used in the PH.x-style UED intensity calculation.

The correction is:

```python
phase = np.exp(+2j * np.pi * (q_reduced_frac @ tau_frac.T))
eigenvectors = eigenvectors * phase[:, None, :, None]
```

where:

- `q_reduced_frac` has shape `(nq, 3)` and is the q coordinate associated with each stored eigenvector.
- `tau_frac` has shape `(nat, 3)` and should be the atom positions used by the UED structure-factor calculation.
- `eigenvectors` has shape `(nq, nmodes, nat, 3)`.

This is a local reader/convention fix. Do not modify phonopy source.

## Why this is needed

The local UED calculation evaluates one-phonon amplitudes with:

```text
F_nu(Q) = sum_atom f_atom(Q) / sqrt(M_atom)
          * [Q dot e_atom,nu(q)]
          * exp(-i 2*pi * Q dot tau_atom)

Q = G + q
```

Phonopy eigenvectors are in a dynamical-matrix Bloch gauge that is not a raw drop-in for this PH.x-style expression. Applying:

```text
e_atom(q) -> e_atom(q) * exp(+i 2*pi * q dot tau_atom)
```

aligns the phonopy eigenvectors with the local UED phase convention:

```text
exp(-i 2*pi * (G + q).tau_atom) * exp(+i 2*pi * q.tau_atom)
= exp(-i 2*pi * G.tau_atom)
```

This fixes the asymmetric patch failure where `conj_sorted_q` looked reasonable around `G=(2,0,0)` but failed around `G=(0,2,0)`.

## Files likely involved

Primary implementation target:

```text
/scratch2/08526/jdgeorga/ued/one_phonon_structure/MoS2/ued_intensity.py
```

Related package copy:

```text
/scratch2/08526/jdgeorga/ued/one_phonon_structure/MoS2/mlip_phonon_scattering/ued_intensity.py
```

Existing diagnostic implementation:

```text
/scratch2/08526/jdgeorga/ued/one_phonon_structure/MoS2/test_mos2_phx_center_36_posthoc_qtau_plus_comparison/compare_phx_center_h5_to_phx_posthoc.py
```

Reference result files:

```text
/scratch2/08526/jdgeorga/ued/one_phonon_structure/MoS2/test_mos2_phx_center_36_posthoc_qtau_plus_comparison/ued_intensity_variant_metrics.csv
/scratch2/08526/jdgeorga/ued/one_phonon_structure/MoS2/test_mos2_phx_center_36_posthoc_qtau_plus_comparison/report.md
```

## Proposed API behavior

Add an explicit option to the phonopy HDF5 loading path rather than silently changing every load.

Recommended option name:

```text
eigenvector_gauge="phonopy_to_phx"
```

Suggested accepted values:

```text
"raw"             # no correction
"phonopy_to_phx"  # apply +q*tau atom-dependent phase
```

Default choice depends on call site:

- For legacy comparisons and debugging, keep `"raw"` available.
- For production UED intensity from phonopy HDF5, use `"phonopy_to_phx"`.

Avoid ambiguous names like `"fixed"` or `"corrected"` because the correction is convention-specific.

## Implementation steps

1. Add a helper function near the HDF5 mesh-loading utilities:

```python
def apply_phonopy_to_phx_eigenvector_gauge(mesh: PhononMeshData) -> PhononMeshData:
    phase = np.exp(+2j * np.pi * (mesh.q_reduced_frac @ mesh.tau_frac.T))
    return replace(mesh, eigenvectors=mesh.eigenvectors * phase[:, None, :, None])
```

If `replace` is not already imported:

```python
from dataclasses import dataclass, replace
```

2. Add an optional argument to `load_phonon_mesh_h5`:

```python
def load_phonon_mesh_h5(
    path: Path,
    q_convention: str = "reduced",
    eigenvector_gauge: str = "raw",
) -> PhononMeshData:
```

3. Build the `PhononMeshData` object exactly as before, then apply the gauge option:

```python
mesh = PhononMeshData(
    q_frac=q_frac,
    q_reduced_frac=q_reduced_frac,
    bvec_rows_anginv=np.array(h5["bvec_rows_anginv"], dtype=float),
    tau_frac=np.array(h5["tau_frac"], dtype=float),
    masses_amu=np.array(h5["masses_amu"], dtype=float),
    species=species,
    frequencies_thz=np.array(h5["frequencies_thz"], dtype=float),
    eigenvectors=np.array(h5["eigenvectors"], dtype=np.complex128),
)

if eigenvector_gauge == "raw":
    return mesh
if eigenvector_gauge == "phonopy_to_phx":
    return apply_phonopy_to_phx_eigenvector_gauge(mesh)
raise ValueError("eigenvector_gauge must be 'raw' or 'phonopy_to_phx'.")
```

4. Update any production phonopy-UED command-line path to expose this option.

Suggested CLI option:

```text
--eigenvector-gauge {raw,phonopy_to_phx}
```

For a command intended to compute physical UED intensities from phonopy HDF5, set the default to:

```text
phonopy_to_phx
```

For comparison scripts, pass the option explicitly so old and new behavior remain easy to test.

5. If a function maps HDF5 modes onto the PH.x q grid before computing UED, apply the gauge after the final q assignment and with the same `tau_frac` used by the structure factor.

For example:

```python
h5_on_qe_grid = load_h5_modes_on_qe_grid(...)
h5_on_qe_grid = apply_phonopy_to_phx_eigenvector_gauge(h5_on_qe_grid)
```

This avoids applying the phase with one q/tau convention and then evaluating with another.

## Important ordering detail

The safest order is:

```text
load HDF5 eigenvectors
determine q_reduced_frac associated with each eigenvector
map/sort onto target UED q grid if needed
set/use the tau_frac that will appear in exp(-i 2*pi*Q.tau)
apply +q*tau gauge
compute UED intensity
```

Do not apply the gauge using native HDF5 `tau_frac` and then later switch to PH.x/QE `tau_frac` for the structure factor. The winning diagnostic variant was:

```text
raw_qtau_plus_qe_tau
```

meaning the phase used the PH.x/QE atom positions after q mapping.

## Validation checklist

Run the post-hoc comparison:

```bash
source /scratch2/08526/jdgeorga/ued/load_epw_env.sh
cd /scratch2/08526/jdgeorga/ued/one_phonon_structure/MoS2
python3 test_mos2_phx_center_36_posthoc_qtau_plus_comparison/compare_phx_center_h5_to_phx_posthoc.py
```

Expected result in:

```text
test_mos2_phx_center_36_posthoc_qtau_plus_comparison/ued_intensity_variant_metrics.csv
```

The corrected variant should remain best or near-best for UED intensity:

```text
raw_qtau_plus_qe_tau
```

Reference metrics from the confirmed run:

```text
variant                 total_log_corr   G=(0,2,0) corr   G=(2,0,0) corr   G=(0,2,0) log-ratio std
raw_qtau_plus_qe_tau     0.992987         0.990557          0.989922          0.039705
conj_sorted_q            0.919181         0.718402          0.943376          0.189821
raw_sorted_q             0.922679         0.753838          0.901079          0.180832
```

Visual checks:

- `patch_G_0_2_variant_grid.png`: corrected result should match PH.x and remove the bad asymmetric patch.
- `patch_G_2_0_variant_grid.png`: corrected result should also match PH.x for the patch that previously looked deceptively good.
- `total_one_phonon_log_ratio_maps.png`: corrected result should strongly suppress structured phase errors.

## Regression test idea

Add a small numerical test that loads the MoS2 HDF5, constructs the PH.x mapped mesh, applies `phonopy_to_phx`, computes tiled intensity for `gmax=3`, and asserts:

```python
G_0_2_log_corr > 0.98
G_2_0_log_corr > 0.98
```

This directly protects the failure mode that motivated the fix.

For a lighter unit test, test only the gauge helper:

```python
phase = np.exp(+2j * np.pi * (q_reduced_frac @ tau_frac.T))
expected = raw_eigenvectors * phase[:, None, :, None]
assert np.allclose(apply_phonopy_to_phx_eigenvector_gauge(mesh).eigenvectors, expected)
```

The full intensity regression is more valuable because it catches sign mistakes and q/tau ordering mistakes.

## Non-goals

Do not:

- Modify `/scratch2/08526/jdgeorga/ued/phonopy`.
- Change phonopy's dynamical-matrix convention.
- Treat `conj_sorted_q` as the production fix.
- Apply the correction globally to PH.x eigenvectors.
- Apply the phase before q sorting/mapping if the final UED q grid differs from the native HDF5 grid.

## Final expected behavior

After this implementation, production phonopy-HDF5 UED intensity calculations should use the corrected eigenvectors by default, while still allowing raw eigenvectors for debugging.

The PH.x comparison should show both `G=(0,2,0)` and `G=(2,0,0)` patches matching consistently. The fix is complete only when the two-patch asymmetry is gone.
