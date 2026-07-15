# DFPT-harmonic + split-MLIP-anharmonic Linewidths — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DFPT-harmonic mode to the `mlip-linewidth-*` pipeline that computes phonon linewidths γ(q,ν)/lifetimes τ from DFPT (2D-LO-TO) harmonic phonons + split-MLIP FC3, with an ablation matrix and dispersion-colored-by-γ/τ comparison figures.

**Architecture:** Reuse the existing phono3py_einsum GPU scattering pipeline; swap only the *harmonic* input. Phase 1 obtains DFPT phonons by running QE `matdyn` (loto_2d) on phono3py's exact mesh and injecting freqs+eigenvectors via `Phono3py.set_phonon_data` (gauge-corrected). Phase 2 adds a fork-native `DynamicalMatrixQELoto2D` validated against the Phase-1 dataset. FC3 is split-MLIP forces on phono3py displacements at the DFPT geometry, with an optional reference-force subtraction.

**Tech Stack:** Python 3.12, phono3py_einsum fork (`port/phono3py-4.3.3`), phonopy (`PH_Q2R`, `set_phonon_data`), MACE (`NLayerCalculator`), QE `matdyn.x`/`q2r`, CuPy/MPI (GPU scatter), numpy, matplotlib, pytest.

**Companion references (read before implementing):**
- Design spec: `docs/superpowers/specs/2026-07-15-dfpt-harmonic-mlip-linewidth-design.md`
- **loto_2d port + injection details (exact math, `file:line`):** `docs/loto2d_implementation_plan.md`
- Gauge convention: `docs/phonopy_eigenvector_gauge.md`
- Pipeline operating guide: `AGENTS.md`

## Global Constraints

- **Env every shell:** `source /pscratch/sd/j/jdgeorga/ued/split_mlip_generators/load_mace_phonon_env.sh` (Python 3.12 + fork). Login-node `python` is 2.7. Never pipe the source (subshell loses it).
- **Branch:** `split_mlip_phonon_lifetime`. **Local commits OK; never `git push` without explicit approval** (AGENTS.md).
- **Package install:** editable (`pip install -e .`); new console scripts must be added to `setup.py` `entry_points`/`scripts` and re-installed.
- **GPU stages** run only inside an allocation: `salloc -N 4 -C gpu --gpus-per-node=4 -A m4480_g -q interactive -t 240 --job-name sd_dfpt_linewidth` (grab with `--no-shell`, then `srun --jobid=<id>`); launch with `srun --overlap -n 4 --gpus-per-task=1` (**`-n 16` hangs**). Nodes have 64 CPUs each — use them for the CPU/force/matdyn stages, then the 4 GPUs/node for scatter. Single-GPU steps: `srun --overlap -N1 -n1 --gpus-per-node=4`. Set `MPICH_GPU_SUPPORT_ENABLED=0` is NOT needed here (MACE uses torch-CUDA); the CuPy engine needs a GPU.
- **Never run codex/grok with Bash `run_in_background`.** For gpt-5.6 in the implementation workflow use the wrappers (codex-implement/review) or run codex detached+poll from the main loop.
- **Do not change existing all-MLIP default behavior:** every new capability is flag-gated and defaults OFF; the bilayer golden (γ_max 0.020164 THz) must stay reproducible.
- **Units/conventions (verbatim):** τ = 1/(4π·γ); interpolate γ never τ; QE post-ZASR Born/ε from `ifc.q2r.xml` (not the Γ dyn XML); α_ewald = 0.98211261577626741; BvK supercell = 6×6×1; matdyn/q2r `asr='crystal'`/`zasr='crystal'`; leave phonopy `symmetrize_fc2` **OFF** for DFPT fc2.
- **DFPT data root (`$DFPT`):** `/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix` — `collect/ifc.q2r.xml`, `collect/matdyn.in`, `collect/phband.freq`, `collect/matdyn.modes`, `q1/mose2_wse2.dyn1.xml`; structure/scf in `../out_scf`.
- **Fork root (`$FORK`):** `/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/repos/phono3py_einsum`.
- **Acceptance gates (apply throughout):** dispersion vs `phband.freq` max|Δω|<0.5 cm⁻¹ / RMS<0.1; eigenvector overlap >0.9999 (subspace singular value >0.9999 for degenerate clusters); γ≥0 & finite; τ>0; Γ acoustic→0; interlayer shear ~0.57 / breathing ~0.85 THz present.

---

## File Structure

New modules (all under `mlip_phonon_scattering/linewidth/` unless noted):
- `dfpt_read.py` — DFPT dir → `dfpt_structure.xyz`, `dfpt_fc2.npy`, `nac_2d.npz`; D3-trap + atom-order + Γ-acoustic gates. CLI `mlip-linewidth-read-dfpt`.
- `matdyn_modes.py` — phono3py grid → matdyn q-list → run matdyn (loto on/off) → `dfpt_modes_{loto,noloto}.npz`. CLI `mlip-linewidth-matdyn-modes`.
- `gauge.py` — atom-dependent Bloch-gauge transform of QE eigenvectors + overlap validation (pure, unit-tested).
- `compare_dispersion.py` — dispersion colored by γ and by τ, multi-variant panels + Δ CSV. CLI `mlip-linewidth-compare`.
- `$FORK/phono3py/phonon/dynamical_matrix_loto_2d.py` — Phase-2 fork-native `DynamicalMatrixQELoto2D`.

Modified:
- `linewidth/forces3.py` — add `--subtract-reference-forces` (reference-force subtraction) to `main_forces3`/`main_forces2_from3`.
- `linewidth/fc_cache.py` — add `--fc2-source {mlip,dfpt}` + `--dfpt-fc2`; skip `symmetrize_fc2` when `dfpt`.
- `linewidth/gpu_scattering_W_phonons_bilayer_comm_mesh.py` — add `--injected-modes` (→ `set_phonon_data`) and `--nac-2d` (→ Phase-2 dyn matrix).
- `$FORK/phono3py/phonon/solver.py` — Phase-2 `DynamicalMatrixQELoto2D` branch in C/Rust solver dispatch.
- `$FORK/phono3py/phonon3/interaction.py` & `interaction_fast.py` — Phase-2 shared `get_phph_dynamical_matrix` factory intercept.
- `setup.py` — register `mlip-linewidth-read-dfpt`, `mlip-linewidth-matdyn-modes`, `mlip-linewidth-compare`.

Tests under `tests/` (repo) and `$FORK/.../tests/` (fork kernel).

---

## Task 1: DFPT reader — structure, fc2 (PH_Q2R), NAC arrays

**Files:**
- Create: `mlip_phonon_scattering/linewidth/dfpt_read.py`
- Test: `tests/test_dfpt_read.py`
- Read first: `phonopy.interface.qe` (`PH_Q2R`, `read_pwscf`), `mlip_phonon_scattering/phonopy_io.py`, `$DFPT/collect/ifc.q2r.xml`, `$DFPT/q1/ph.out`.

**Interfaces:**
- Produces:
  - `read_dfpt(dfpt_dir: str, layer_symbols=None, atom_types=None, layer_ids=None, symprec=1e-4) -> DfptData` where `DfptData` is a dataclass with `structure` (ase.Atoms w/ `atom_types`,`layer_ids` arrays), `fc2` (np.ndarray `(N,N,3,3)` phono3py supercell order), `supercell_matrix` (=[6,6,1]), `nac` (dict: `born (nat,3,3)`, `dielectric (3,3)`, `alpha_ewald float`, `area float`, `c float`, `periodic_axes=(0,1)`, `factor=2.0`).
  - `check_d3_and_acoustic(ph_out_path, gamma_freqs) -> dict` (gate results).
  - `atom_order_map(qe_positions, phonopy_supercell, symprec) -> np.ndarray` + assertion helper.

- [ ] **Step 1: Failing test — fc2 atom-order map is a valid permutation with matched positions**

```python
# tests/test_dfpt_read.py
import numpy as np, pytest
from mlip_phonon_scattering.linewidth.dfpt_read import read_dfpt
DFPT = "/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix"

@pytest.mark.slow
def test_read_dfpt_fc2_and_mapping():
    d = read_dfpt(DFPT, layer_symbols=[['Mo','Se','Se'],['W','Se','Se']])
    N = 6 * 6 * 1 * 6                      # 6x6x1 supercell of a 6-atom cell
    assert d.fc2.shape == (N, N, 3, 3)
    assert list(np.diag(d.supercell_matrix)) == [6, 6, 1]
    # fc2 acoustic sum rule per atom-row should be ~0 (QE zasr='crystal' already applied)
    row = d.fc2.sum(axis=1)               # (N,3,3)
    assert np.abs(row).max() < 1e-6
    # structure carries split-MLIP tags for the bilayer force model
    assert set(np.unique(d.structure.arrays['layer_ids'])) == {0, 1}
    assert d.structure.arrays['atom_types'].shape[0] == d.structure.get_global_number_of_atoms()
```

- [ ] **Step 2: Failing test — NAC arrays match the known DFPT values**

```python
@pytest.mark.slow
def test_read_dfpt_nac_values():
    d = read_dfpt(DFPT, layer_symbols=[['Mo','Se','Se'],['W','Se','Se']])
    eps = d.nac['dielectric']
    assert np.allclose(np.diag(eps), [5.8788, 5.8788, 1.2818], atol=1e-3)
    assert d.nac['born'].shape == (6, 3, 3)
    assert np.isclose(d.nac['born'][0, 0, 0], -1.77642705, atol=1e-5)   # post-ZASR
    assert np.isclose(d.nac['alpha_ewald'], 0.98211261577626741, atol=1e-9)
    assert d.nac['periodic_axes'] == (0, 1)
    assert np.isclose(d.nac['factor'], 2.0)
```

- [ ] **Step 3: Run tests to verify they fail** — `pytest tests/test_dfpt_read.py -v` → FAIL (module missing).

- [ ] **Step 4: Implement `dfpt_read.py`**

Implementation content (concrete):
- Parse `ifc.q2r.xml` via phonopy `PH_Q2R(filename, ...)` to get the fc2 ndarray + supercell; use its `_get_q2r_positions`/`_get_site_mapping` to build the QE→phonopy atom-order permutation. Assert `max|pos_qe[map] − pos_phonopy| < symprec` (raise on failure — a silent swap corrupts fc2; two distinct Se environments).
- Extract `<EPSILON>`, `<ZSTAR>/<Z_AT_.n>`, `<alpha_ewald>`, `<UNIT_CELL_VOLUME_AU>`, `<MESH_NQ1_NQ2_NQ3>` from `ifc.q2r.xml` (post-ZASR Born). Compute in-plane area `A=|a1×a2|`, `c=Ω/A`.
- Read the equilibrium structure from `../out_scf` (or the `ifc.q2r.xml` header cell/positions); attach `atom_types`/`layer_ids` from `layer_symbols` (bottom layer types 0/1/2, top 3/4/5; `layer_ids` 0/1) so the split-MLIP `NLayerCalculator` engages.
- `check_d3_and_acoustic`: grep the DFPT `q1/ph.out` for the Grimme-D3 Hessian line; require Γ acoustic (min of the 3 lowest) > −0.05 THz; return a dict and raise if the D3 line is absent AND Γ acoustics are bad (the `nscf-clobbers-vdW` trap).

- [ ] **Step 5: Run tests to verify they pass** — `pytest tests/test_dfpt_read.py -v` → PASS.

- [ ] **Step 6: Commit** — `git add mlip_phonon_scattering/linewidth/dfpt_read.py tests/test_dfpt_read.py && git commit -m "feat(dfpt): read DFPT structure+fc2(PH_Q2R)+NAC; atom-order/D3 gates"`

---

## Task 2: `mlip-linewidth-read-dfpt` CLI

**Files:**
- Modify: `mlip_phonon_scattering/linewidth/dfpt_read.py` (add `main()`), `setup.py` (register console script)
- Test: `tests/test_dfpt_read.py` (add CLI smoke)

**Interfaces:** Consumes `read_dfpt`. Produces on disk: `dfpt_structure.xyz`, `dfpt_fc2.npy`, `nac_2d.npz` in `--out-dir`.

- [ ] **Step 1: Failing test — CLI writes the three artifacts**

```python
@pytest.mark.slow
def test_read_dfpt_cli(tmp_path):
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "mlip_phonon_scattering.linewidth.dfpt_read",
        "--dfpt-dir", DFPT, "--layer-symbols", "[['Mo','Se','Se'],['W','Se','Se']]",
        "--out-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("dfpt_structure.xyz", "dfpt_fc2.npy", "nac_2d.npz"):
        assert (tmp_path / f).exists()
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `main()`** — argparse (`--dfpt-dir`, `--layer-symbols`, `--out-dir`, `--symprec`); call `read_dfpt`; write extxyz, `np.save` fc2, `np.savez` nac dict; print the gate results.
- [ ] **Step 4: Register in `setup.py`** `entry_points`: `mlip-linewidth-read-dfpt = linewidth.dfpt_read:main`; `pip install -e .`.
- [ ] **Step 5: Run → PASS.**
- [ ] **Step 6: Commit** — `git commit -am "feat(dfpt): mlip-linewidth-read-dfpt CLI"`

---

## Task 3: matdyn on the phono3py mesh → DFPT modes

**Files:**
- Create: `mlip_phonon_scattering/linewidth/matdyn_modes.py`
- Test: `tests/test_matdyn_modes.py`
- Read first: `docs/loto2d_implementation_plan.md` §5 (q-list = `grid.addresses · QDinv^T`, BZ-boundary duplicates; matdyn.modes √M rescale), `$DFPT/collect/matdyn.in`.

**Interfaces:**
- Produces: `matdyn_qpoints_for_mesh(phono3py_yaml, mesh) -> (grid_address (Ngrid,3), q_cryst (Ngrid,3))`; `run_matdyn(ifc_xml, qlist, loto_2d: bool, workdir) -> str(flvec_path)`; `parse_matdyn_modes(flfrq, flvec, masses) -> dict{frequencies (Ngrid,nb) THz, eigenvectors (Ngrid,nb,nat,3) complex, grid_address}`.
- CLI writes `dfpt_modes_loto.npz` / `dfpt_modes_noloto.npz` (keys: `frequencies`,`eigenvectors`,`grid_address`,`mesh`).

- [ ] **Step 1: Failing test — q-list matches phono3py's grid addresses**

```python
# tests/test_matdyn_modes.py
import numpy as np, pytest
from mlip_phonon_scattering.linewidth.matdyn_modes import matdyn_qpoints_for_mesh

@pytest.mark.slow
def test_qlist_matches_phono3py_grid(phono3py_yaml_bilayer):
    ga, q = matdyn_qpoints_for_mesh(phono3py_yaml_bilayer, [12, 12, 1])
    assert ga.shape[0] >= 12 * 12 * 1          # >= product; BZ-boundary duplicates allowed
    assert q.shape == (ga.shape[0], 3)
    # q reconstructs from addresses and QD^-1 (bit-exact to phono3py's own convention)
    # (verified against Phono3py grid in implementation)
```

- [ ] **Step 2: Failing test — parsed modes reproduce QE `phband.freq` on the band path (gate)**

```python
@pytest.mark.slow
def test_matdyn_reproduces_phband(tmp_path, ifc_xml, phband_freq, masses):
    from mlip_phonon_scattering.linewidth.matdyn_modes import run_matdyn, parse_matdyn_modes, load_phband_freq
    # use the exact 152-pt path parsed from collect/matdyn.in
    qpath = load_matdyn_path("$DFPT/collect/matdyn.in".replace("$DFPT", DFPT))
    flvec = run_matdyn(ifc_xml, qpath, loto_2d=True, workdir=tmp_path)
    modes = parse_matdyn_modes(tmp_path / "phband.freq", flvec, masses)
    ref = load_phband_freq(phband_freq)        # (nq, nb) cm^-1
    dperr = np.abs(modes['frequencies_cm'] - ref)
    assert dperr.max() < 0.5 and np.sqrt((dperr**2).mean()) < 0.1
```

- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Implement `matdyn_modes.py`** — build grid via `phono3py.load(yaml).mesh` grid addresses; convert to crystal q; write a matdyn input (`flfrc=ifc.q2r.xml`, `asr='crystal'`, `loto_2d=<bool>`, `q_in_cryst_coord=.true.`, `flvec=matdyn.modes`) with the q-list; `subprocess` matdyn.x (env-sourced); parse `.freq`/`.modes`, undo matdyn's √M eigenvector rescale to phono3py mass convention; return arrays. THz factor per Global Constraints.
- [ ] **Step 5: Run → PASS** (this is the make-or-break harmonic gate). Investigate any systematic >0.1 cm⁻¹.
- [ ] **Step 6: Commit** — `git commit -am "feat(dfpt): matdyn on phono3py mesh; phband.freq gate <0.5 cm-1"`

---

## Task 4: Bloch-gauge transform + eigenvector overlap validation

**Files:**
- Create: `mlip_phonon_scattering/linewidth/gauge.py`
- Test: `tests/test_gauge.py`
- Read first: `docs/loto2d_implementation_plan.md` §5B; `docs/phonopy_eigenvector_gauge.md`.

**Interfaces:**
- Produces: `apply_bloch_gauge(eigvecs (Nq,nb,nat,3), q_cryst (Nq,3), tau_frac (nat,3), sign: int) -> eigvecs'`; `select_gauge(e_ph, e_qe, q, tau) -> (sign, residual)`; `overlap(e_a, e_b) -> per-mode |<a|b>|`; `subspace_overlap(e_a, e_b, degen_tol)`.

- [ ] **Step 1: Failing test — gauge helper matches the closed-form phase and round-trips**

```python
# tests/test_gauge.py
import numpy as np
from mlip_phonon_scattering.linewidth.gauge import apply_bloch_gauge, overlap
def test_gauge_phase_and_roundtrip():
    rng = np.random.default_rng(0)
    Nq, nb, nat = 4, 6, 2
    e = rng.standard_normal((Nq, nb, nat, 3)) + 1j*rng.standard_normal((Nq, nb, nat, 3))
    q = rng.standard_normal((Nq, 3)); tau = rng.standard_normal((nat, 3))
    e1 = apply_bloch_gauge(e, q, tau, sign=+1)
    phase = np.exp(1j*2*np.pi*(q @ tau.T))            # (Nq,nat)
    assert np.allclose(e1, e * phase[:, None, :, None])
    # +then- restores
    assert np.allclose(apply_bloch_gauge(e1, q, tau, sign=-1), e)
```

- [ ] **Step 2: Failing test — overlap of identical eigenvectors is 1**

```python
def test_overlap_identity():
    rng = np.random.default_rng(1); e = rng.standard_normal((3, 4, 2, 3)) + 0j
    ov = overlap(e, e)
    assert np.allclose(ov, 1.0, atol=1e-12)
```

- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Implement `gauge.py`** — `apply_bloch_gauge` multiplies by `exp(sign·2πi q·τ)[:,None,:,None]`; `overlap` = `|Σ_{atom,cart} conj(e_a)·e_b|` per (q,mode) with per-mode normalization; `subspace_overlap` = min singular value of the block overlap within degenerate clusters; `select_gauge` tries sign ∈ {+1,−1} (and complex-conjugate/q→−q as needed) and returns the one maximizing mean overlap against a reference dynamical matrix reconstruction `D=EΩ²E†`.
- [ ] **Step 5: Run → PASS.**
- [ ] **Step 6: Commit** — `git commit -am "feat(dfpt): Bloch-gauge transform + overlap/subspace validation"`

---

## Task 5: `mlip-linewidth-matdyn-modes` CLI

**Files:** Modify `matdyn_modes.py` (`main()`), `setup.py`. Test: append CLI smoke to `tests/test_matdyn_modes.py`.

**Interfaces:** `--phono3py-yaml --ifc-xml --mesh MX MY MZ --loto-2d {on,off} --out <npz>`. Applies the gauge (Task 4) with the sign chosen by `select_gauge` against the fc2 dynamical matrix at a few generic q; writes gauge-corrected modes.

- [ ] **Step 1: Failing CLI test** — run for a 12×12×1 mesh, loto on and off; assert both npz exist with `frequencies`,`eigenvectors`,`grid_address`; assert loto vs noloto differ near Γ (LO-TO) but agree at a mid-BZ q.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `main()`**; register `mlip-linewidth-matdyn-modes = linewidth.matdyn_modes:main`; `pip install -e .`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(dfpt): mlip-linewidth-matdyn-modes CLI (loto on/off, gauge-corrected)"`

---

## Task 6: Reference-force subtraction in FC3/FC2 forces

**Files:**
- Modify: `mlip_phonon_scattering/linewidth/forces3.py` (`main_forces3`, `main_forces2_from3`, shared runner `_run_force_evaluation`)
- Test: `tests/test_residual_subtraction.py`
- Read first: `forces3.py:27-60,138-169`; `forces.py:26-46`.

**Interfaces:** New flag `--subtract-reference-forces` (default OFF globally; the DFPT run scripts pass it ON). Behavior: evaluate MLIP forces on the *undisplaced* supercell `F0` once; write `F_corrected[i] = F[i] − F0` before `np.save`. Also save `reference_forces.npy` (F0) for auditing.

- [ ] **Step 1: Failing test — corrected forces zero out a constant reference**

```python
# tests/test_residual_subtraction.py
import numpy as np
from mlip_phonon_scattering.linewidth.forces3 import subtract_reference_forces
def test_subtract_reference():
    F = np.random.default_rng(0).standard_normal((5, 6, 3))
    F0 = F.mean(axis=0)                         # pretend nonzero reference
    Fc = subtract_reference_forces(F, F0)
    assert np.allclose(Fc, F - F0[None])
    # a set whose every frame equals F0 -> all zero
    assert np.allclose(subtract_reference_forces(np.tile(F0, (3,1,1)), F0), 0.0)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — add pure `subtract_reference_forces(F, F0)`; in the force mains, when `--subtract-reference-forces`, build the undisplaced supercell (phono3py `supercell`/`phonon_supercell`), evaluate `F0` via `forces.evaluate_structures` (same calculator/nlayer config), subtract, and save both arrays. MPI: compute `F0` on rank 0 and broadcast (or all-ranks identical).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(dfpt): --subtract-reference-forces (F(u)-F(0)) for un-relaxed DFPT geom"`

---

## Task 7: cache-fc DFPT fc2 source + scatter-gpu mode injection (Phase 1 wiring)

**Files:**
- Modify: `linewidth/fc_cache.py` (`--fc2-source {mlip,dfpt}`, `--dfpt-fc2`; skip `symmetrize_fc2` when `dfpt`), `linewidth/gpu_scattering_W_phonons_bilayer_comm_mesh.py` (`--injected-modes <npz>` → `set_phonon_data`)
- Test: `tests/test_injection.py`
- Read first: `docs/loto2d_implementation_plan.md` §6 (set_phonon_data API `api_phono3py.py:1280`, interaction copy `interaction.py:653`, `interaction_fast.py:726`); `gpu_scattering_*.py:272-273,600-605`.

**Interfaces:** Consumes `dfpt_fc2.npy` (Task 1) and `dfpt_modes_*.npz` (Task 5). When `--injected-modes` is given, after `init_phph_interaction` call `ph3.set_phonon_data(frequencies, eigenvectors, grid_address)`; assert all `phonon_done == 1`; skip the fc2 `run_phonon_solver`. When `--fc2-source dfpt`, load `--dfpt-fc2` as `fc2.npy` in the cache and do **not** `symmetrize_fc2`.

- [ ] **Step 1: Failing test — injected modes populate phonon data and are not overwritten**

```python
# tests/test_injection.py
import numpy as np, pytest
@pytest.mark.slow
def test_injection_sets_phonon_data(bilayer_yaml, dfpt_modes_npz, fc3_npy, dfpt_fc2_npy):
    import phono3py
    ph3 = phono3py.load(bilayer_yaml, log_level=0); ph3.mesh_numbers = [12,12,1]
    ph3.fc3 = np.load(fc3_npy, allow_pickle=True); ph3.fc2 = np.load(dfpt_fc2_npy, allow_pickle=True)
    ph3.init_phph_interaction()
    z = np.load(dfpt_modes_npz)
    ph3.set_phonon_data(z['frequencies'], z['eigenvectors'], z['grid_address'])
    itr = ph3.phph_interaction
    assert np.all(itr.get_phonons()[2] == 1)         # phonon_done all set
    f = itr.get_phonons()[0]
    assert np.allclose(f[itr.bz_grid.grg2bzg[:len(z['frequencies'])]], z['frequencies'], atol=1e-6)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** cache-fc `--fc2-source`/`--dfpt-fc2` (skip symmetrize); scatter-gpu `--injected-modes` (load npz → `set_phonon_data` after `init_phph_interaction`, before the solver; guard against a later `run_phonon_solver` overwrite). Keep both flags OFF by default.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(dfpt): cache-fc --fc2-source dfpt (no double-ASR) + scatter-gpu --injected-modes"`

---

## Task 8: Comparison figures (dispersion colored by γ and by τ)

**Files:**
- Create: `linewidth/compare_dispersion.py` (CLI `mlip-linewidth-compare`); Modify `setup.py`
- Test: `tests/test_compare_dispersion.py`
- Read first: `linewidth/linewidth_path.py` (`compute_band_structure:76`, `gamma_on_path:203`, `lifetime_from_gamma:284`, `_render_two_panel:359`).

**Interfaces:** `--variants name1=yaml1,fc2_1,gamma1 name2=... --mesh --temperature --out-dir`. For each variant compute the band structure from its fc2 + γ-on-path (interpolate γ, never τ); render **Fig 1** (ω colored by γ) and **Fig 2** (ω colored by τ) as side-by-side panels with a shared colorbar; write `comparison_summary.csv` (K/M top-optical γ, τ + ratios V2/V1, V3/V1, V2/V3, V2loto/V2noloto).

- [ ] **Step 1: Failing test — figures + CSV are produced for ≥2 variants**

```python
# tests/test_compare_dispersion.py
import pathlib, pytest
@pytest.mark.slow
def test_compare_outputs(tmp_path, two_variant_inputs):
    from mlip_phonon_scattering.linewidth.compare_dispersion import compare
    compare(two_variant_inputs, mesh=[12,12,1], temperature=50, out_dir=tmp_path)
    assert (tmp_path / "dispersion_linewidth.png").exists()
    assert (tmp_path / "dispersion_lifetime.png").exists()
    assert (tmp_path / "comparison_summary.csv").exists()
```

- [ ] **Step 2: Failing test — τ is derived from γ, not interpolated (project rule)**

```python
def test_tau_from_gamma_rule():
    import numpy as np
    from mlip_phonon_scattering.linewidth.linewidth_path import lifetime_from_gamma
    g = np.array([0.0, 1e-3, 1e-2])
    tau = lifetime_from_gamma(g)               # 1/(4*pi*g); inf at g=0
    assert np.isinf(tau[0]) and np.all(tau[1:] > 0)
    assert np.allclose(tau[1:], 1.0/(4*np.pi*g[1:]))
```

- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Implement `compare_dispersion.py`** reusing `linewidth_path` helpers; matplotlib scatter on the band path colored by γ (inferno) / τ (viridis), shared `Normalize`+colorbar across panels; CSV writer. **This module is user-facing (taste ≥ 7) → implement/refine with Opus in the workflow.**
- [ ] **Step 5: Register** `mlip-linewidth-compare = linewidth.compare_dispersion:main`; `pip install -e .`.
- [ ] **Step 6: Run → PASS.**
- [ ] **Step 7: Commit** — `git commit -am "feat(dfpt): mlip-linewidth-compare — dispersion colored by gamma/tau across variants"`

---

## Task 9: End-to-end Phase-1 integration + gates (small mesh)

**Files:**
- Create: `examples/mose2_wse2_bilayer_dfpt/run_dfpt_linewidth.sh`, `.../run_salloc_pipeline.sh`
- Test: `tests/test_dfpt_pipeline_smoke.py`

**Interfaces:** A driver that runs, for the bilayer at a small mesh (12×12×1): read-dfpt → matdyn-modes(loto,noloto) → phono3py-yaml → forces3(+residual on/off) → cache-fc(dfpt fc2) → scatter-gpu(injected) → extract-gamma → validate → compare. Produces all 7 variant γ npz + the two comparison figures.

- [ ] **Step 1: Failing smoke test (GPU-marked)** — asserts, after a 12×12×1 run: each variant `gamma_*.npz` has γ≥0/finite; τ>0; harmonic dispersion for V2-loto passes the `phband.freq` gate (<0.5 cm⁻¹) and eigenvector overlap >0.9999; interlayer shear/breathing present; both figures render.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement the run scripts** following `examples/mose2_wse2_bilayer/run_linewidth.sh` patterns (`srun --overlap -n 4 --gpus-per-task=1`, `need()` guards, `OUTPUT_DIR`); wire the 7 variants (share FC3 across V2/V3 per residual setting; share DFPT fc2/modes across residual settings).
- [ ] **Step 4: Run inside `salloc` → PASS** (record γ_max, K/M top-optical γ/τ per variant).
- [ ] **Step 5: Commit** — `git commit -am "feat(dfpt): end-to-end Phase-1 pipeline + 7-variant driver + gates"`

---

## Task 10: Production 36×36×1 run + comparison deliverable

**Files:** Modify the run scripts (mesh 36×36×1, T=50 K). Output: `runs/mose2_wse2_bilayer_dfpt_36x36x1_T50/` with 7 γ npz + `dispersion_linewidth.png`/`dispersion_lifetime.png`/`comparison_summary.csv`.

- [ ] **Step 1:** Run the full 36×36×1 pipeline in `salloc` (V1 reuse `runs/mose2_wse2_bilayer_36x36x1_T50`).
- [ ] **Step 2:** Verify all gates (§Global Constraints) at production mesh; confirm V1 reproduces the golden γ_max 0.020164 THz.
- [ ] **Step 3:** Inspect the two figures + CSV; sanity-check the V2/V1, V2/V3, V2loto/V2noloto ratios physically.
- [ ] **Step 4: Commit** the run outputs (figures + CSV + a short `RESULTS.md`) — `git commit -m "results(dfpt): 36x36x1 bilayer 7-variant linewidths + comparison figures"`

---

## Task 11: Phase-2 fork-native `DynamicalMatrixQELoto2D` kernel

**Files:**
- Create: `$FORK/phono3py/phonon/dynamical_matrix_loto_2d.py`
- Test: `$FORK/.../tests/test_dynamical_matrix_loto_2d.py`
- Read first: `docs/loto2d_implementation_plan.md` §2/§4 (exact `rgd_blk` math + `rigid.f90:99-239` refs; on-site subtraction; G-sum cutoff K²/4α<14; area/c/r_eff), §7 pitfalls.

**Interfaces:**
- Produces: `class DynamicalMatrixQELoto2D(DynamicalMatrix)` (subclass the plain base — NOT `DynamicalMatrixNAC`) with `run(q)`, `get_dynamical_matrices(qpoints)`, and the single authoritative `_qe_loto2d_blocks(qpoints)` kernel. `nac_params` extended fields: `method="qe_loto_2d"`, `loto_2d_alpha`, `loto_2d_periodic_axes=(0,1)`, `loto_2d_normal_axis=2`, `loto_2d_c`, `loto_2d_fc_is_short_range=True`.

- [ ] **Step 1: Failing test — scalar `rgd_blk` reference == vectorized kernel**

```python
# $FORK/.../tests/test_dynamical_matrix_loto_2d.py — port QE rgd_blk(loto_2d) literally as ground truth
def test_vectorized_matches_scalar_rgd_blk(loto2d_case):
    D_scalar = rgd_blk_reference(**loto2d_case)          # literal loop port
    D_vec = DynamicalMatrixQELoto2D(**loto2d_case).get_dynamical_matrices(loto2d_case['q'])
    assert np.allclose(D_vec, D_scalar, atol=1e-10)
```

- [ ] **Step 2: Failing test — physics invariants** (Hermiticity; `D(−q)=D(q)*`; exact Γ LO/TO degeneracy with finite linear LO slope; invariance under increased vacuum at fixed r_eff; 3 acoustic modes→0 at Γ).
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Implement the kernel** exactly per `docs/loto2d_implementation_plan.md` §4A (short-range via phonopy no-NAC builder + `D_QE_loto2d`: Cartesian-reciprocal q, enumerate G∥ under cutoff, form A/c/r_eff, cached q-independent on-site subtraction, G+q term skipping K=0, mass-weight, Hermitianize). Build in QE's q+G phase convention then apply the atom-diagonal gauge (Task 4) — sign fixed by the §5 overlap test.
- [ ] **Step 5: Run → PASS.**
- [ ] **Step 6: Commit (in fork repo)** — `git -C $FORK commit -am "feat: DynamicalMatrixQELoto2D (QE 2D LO-TO port)"`

---

## Task 12: Fork solver + factory wiring for `DynamicalMatrixQELoto2D`

**Files:**
- Modify: `$FORK/phono3py/phonon/solver.py` (C & Rust dispatch), `$FORK/phono3py/phonon3/interaction.py:626`, `interaction_fast.py:701` (shared `get_phph_dynamical_matrix` factory intercept on `method=="qe_loto_2d"`)
- Test: `$FORK/.../tests/test_loto2d_solver_dispatch.py`

**Interfaces:** Add an early `isinstance(dm, DynamicalMatrixQELoto2D)` branch in `run_phonon_solver_c`/`run_phonon_solver_rust` that fills `_frequencies`/`_eigenvectors`/`phonon_done` from the kernel (bypassing the compiled 3D-NAC path); a shared factory used identically by both interaction impls.

- [ ] **Step 1: Failing test** — a Phono3py built with `nac_params={method:"qe_loto_2d",...}` yields frequencies from the 2D kernel on both `interaction` and `interaction_fast` paths (identical arrays).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the solver branch + shared factory.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit (fork)** — `git -C $FORK commit -am "feat: route DynamicalMatrixQELoto2D through C/Rust solvers + phph factory"`

---

## Task 13: Phase-2 vs Phase-1 regression gate + scatter `--nac-2d`

**Files:**
- Modify: `linewidth/gpu_scattering_*.py` (`--nac-2d <nac_2d.npz>` → build `DynamicalMatrixQELoto2D` from the DFPT fc2 + NAC arrays)
- Test: `tests/test_phase2_matches_phase1.py`

**Interfaces:** `--nac-2d` path (Phase 2) is an alternative to `--injected-modes` (Phase 1); both feed the same downstream γ extraction.

- [ ] **Step 1: Failing test — Phase-2 freqs/eigvecs match Phase-1 injected dataset**

```python
@pytest.mark.slow
def test_phase2_matches_phase1(bilayer_yaml, dfpt_fc2_npy, nac_2d_npz, dfpt_modes_loto_npz):
    # build phono3py with DynamicalMatrixQELoto2D and compare mesh freqs/eigvecs to the matdyn-injected npz
    ... # frequencies: max|Δ| < 0.5 cm^-1; eigenvector overlap > 0.9999 (subspace for degenerate)
```

- [ ] **Step 2: Failing test — γ from Phase-2 matches γ from Phase-1** on a small mesh (K/M top-optical within a few %, limited by CPU/GPU fc3 tolerance).
- [ ] **Step 3: Run → FAIL.**
- [ ] **Step 4: Implement** `--nac-2d` wiring; run both phases; assert the gates.
- [ ] **Step 5: Run → PASS.**
- [ ] **Step 6: Commit** — `git commit -am "feat(dfpt): scatter --nac-2d (Phase 2) + Phase2==Phase1 regression gate"`

---

## Self-Review (spec coverage)

- Read DFPT (structure/no-relax/fc2/NAC) → **T1,T2**. Residual subtraction toggle → **T6**. Split-MLIP FC3 @ DFPT geom → **T6+existing forces3**. DFPT harmonic Phase 1 (matdyn+inject+gauge) → **T3,T4,T5,T7**. loto on/off axis → **T5** (matdyn flag) + **T13** (Phase-2). FC2-source dfpt / no double-ASR → **T7**. Figures (γ/τ colored dispersion, both pipelines) → **T8**. Variant matrix / ablation (V1/V2loto/V2noloto/V3 × residual) → **T9,T10**. Validation gates (phband.freq, overlap, D3-trap, atom-order, units, γ≥0/τ>0) → **T1,T3,T9,T10**. Phase-2 fork-native loto_2d → **T11,T12,T13**. Science caveat (D3-in-fc2 not in MLIP-FC3) → documented in spec §7 (no task; methods note). **No gaps.**
- Placeholder scan: none (`TODO`/`TBD`-free; every step has test/impl content or a precise companion-doc `file:line` reference).
- Type consistency: `read_dfpt`→`DfptData(fc2, nac, supercell_matrix, structure)`; `matdyn_modes`→npz `{frequencies,eigenvectors,grid_address,mesh}` consumed verbatim by `set_phonon_data` (T7) and Phase-2 gate (T13); `apply_bloch_gauge(...,sign)` reused in T5/T11; `subtract_reference_forces(F,F0)` in T6. Consistent.

---

## Understand-phase corrections (VERIFIED LIVE — these OVERRIDE the task text above)

Confirmed against phonopy 4.3.1 / phono3py 4.3.4.dev7 (fork) on 2026-07-15. Where this
section conflicts with a task above, **this section wins**.

**C1 — `PH_Q2R` cannot read `ifc.q2r.xml` (Task 1).** `PH_Q2R._parse_q2r` is a plain-text
readline parser; the file is XML → verified crash (`invalid literal for int(): '<?xml'`).
There is no text `.fc` companion in `collect/`. **Parse the XML directly** (`xml.etree.ElementTree`):
fc blocks are tags `<s_s1_m1_m2_m3.s.s1.m1.m2.m3>` with an `<IFC>` 3×3 child; `s,s1∈1..6`,
`(m1,m2,m3)∈1..(6,6,1)`; block = Φ between atom `s` in cell 0 and atom `s1` in cell
`R=(m1-1,m2-1,m3-1)`, units **Ry/au²**. Fill phonopy's compact layout exactly as
`qe.py:_parse_fc` does (index transpose `fc[j, i*ndim+i_dim, ll, k]`), then reuse
`PH_Q2R._get_q2r_positions(cell)` + `_get_site_mapping(scell.scaled_positions, q2r_spos, scell.cell)`
+ `fc[pcell.p2s_map,:]=q2r_fc[:,site_map]` + `distribute_force_constants_by_translations` →
`(N,N,3,3)`, N=216. (`_get_site_mapping` already enforces the plan's atom-order assertion.)
Read the QE primitive cell via `read_pwscf` on `../out_scf` (or the XML `<AT>`/`<ATOM TAU>` header;
identify species by `<MASS.*>`, NOT `<TYPE_NAME.*>` which are unreliable). NAC arrays
(`<EPSILON>`, `<ZSTAR>/<Z_AT_.n>`, `<alpha_ewald>`, `<UNIT_CELL_VOLUME_AU>`, `<MESH_NQ1_NQ2_NQ3>`,
`<AT>`+alat) are all in `ifc.q2r.xml` as the plan assumes (post-ZASR Born; `born[0,0,0]=-1.77642705`).

**C2 — `set_phonon_data` contract (Tasks 3, 5, 7).** Signature
`Phono3py.set_phonon_data(frequencies, eigenvectors, grid_address)` — requires
`init_phph_interaction()` to have run first (else silent no-op). Shapes:
`frequencies (num_grid, nband) float64` THz; `eigenvectors (num_grid, nband, nband) complex128`
with **mode = column** (row index = atom*3+cart); `grid_address` **must equal `ph3.grid.addresses`**
(the full BZ grid). So the stored npz `eigenvectors (Ngrid, nb, nat, 3)` must be
`reshape(Ngrid, nb, nat*3).transpose(0,2,1)` before injection. Task 7 Step-1 test: pass the
reshaped array; replace the malformed `grg2bzg[:len]` assertion with
`np.allclose(itr.get_phonons()[0], z['frequencies'])` (both already full-BZ order);
`itr.get_phonons()[2]` is `phonon_done` (all 1 after injection).

**C3 — Injection order & overwrite safety (Task 7).** Call `init_phph_interaction(nac_q_direction=None)`
FIRST, then `set_phonon_data`. Overwrite is auto-prevented: both C (`c/phonon.c:162`) and Rust
(`solver.py:208`) solvers skip `phonon_done==1` points, so a later `run_phonon_solver()` is a
no-op over injected points. The ONLY overwrite risk is Γ under `is_nac=True` — avoided by keeping
`nac_q_direction=None`. (Still safe to also skip the explicit `run_phonon_solver()` block at
gpu_scattering line 604-607, but not required.)

**C4 — matdyn.modes is ALREADY mass-weighted (Task 3 Step 4 is wrong).** matdyn writes
`z·√(amu_ry·M)` which reconstructs phonopy's unit-norm mass-weighted eigenvector `w` exactly —
there is **NO √M to undo**. Do only: (a) per-mode renormalize to unit norm, (b) reshape/transpose
to `(gp, nat*3, mode)`, (c) atom-dependent Bloch gauge `exp(±2πi q·τ_κ)` (Task 4). Parse the
`[THz]` column directly (ordinary ν, matches phono3py); use `[cm⁻¹]` only for the phband gate.
Set the phono3py primitive-cell masses explicitly to the QE values (Mo 95.95, W 183.84, Se 78.971).

**C5 — Full BZ grid is much larger than prod(mesh) (Task 3).** For this anisotropic slab,
`ph3.grid.addresses` has **443 pts at 12×12×1** and **3979 at 36×36×1** (q_z=±1 boundary images
because the reciprocal c-axis is short). The matdyn q-list AND the injected arrays must have length
`len(ph3.grid.addresses)`, enumerated in that order; `q_crystal = grid.addresses @ grid.QDinv.T`.

**C6 — QE fc2 units when diagonalized (Tasks 8, 11).** The XML fc2 is **Ry/au²**. Harmless for
Phase-1 injection (not diagonalized). But `compare_dispersion` (T8) and the Phase-2 native matrix
(T11) DO diagonalize → use `frequency_factor_to_THz = 108.97077184367376` (QE) or convert fc2 to
eV/Å², NOT the default VaspToTHz 15.6333.

**C7 — Residual F0 needs no MPI broadcast (Task 6).** `np.save` is rank-0-only (forces3.py:57);
compute F0 on rank 0 and subtract there. Build the undisplaced cell from `ph3.supercell` /
`ph3.phonon_supercell` via `phonopy_io.phonopy_atoms_to_ase(..., include_masses=False)` +
`copy_repeated_arrays(relaxed_atoms, ..., ("atom_types","layer_ids"))` (same transform as
forces3.py:104-112). Add `--subtract-reference-forces` in `_parse_phono3py_force_args` (scopes to
the two phono3py mains only).

**C8 — DFPT-fc2 edit is in the shared helper, not fc_cache.py (Task 7).**
`ph3.produce_fc2(symmetrize_fc2=True)` / `np.save(fc2_mesh,...)` are in
`gpu_scattering_W_phonons_bilayer_comm_mesh.py:265-267` inside
`set_phono3py_forces_with_mesh_fc_cache` (signature line 219, `need_compute` block 259-268). Thread
a `dfpt_fc2: Optional[str]=None` kwarg through it; when set, still `produce_fc3(symmetrize_fc3r=True)`
but replace 265/267 with `ph3.fc2 = np.load(dfpt_fc2)` (NO symmetrize) → save to `fc2_mesh`. Add
`--fc2-source {mlip,dfpt}` + `--dfpt-fc2` in `fc_cache.py:parse_args`; default None preserves both
callers (`fc_cache.main` and `run_scattering`).

**C9 — setup.py entry points must be fully-qualified (Tasks 2, 5, 8):**
`mlip-linewidth-read-dfpt = mlip_phonon_scattering.linewidth.dfpt_read:main` (and
`...matdyn_modes:main`, `...compare_dispersion:main`) — NOT `linewidth.<mod>:main`.

**C10 — No `need()` guard exists (Task 9).** Example scripts are linear numbered steps with
`multi_rank(){ srun --overlap -n "$MPI_RANKS" --gpus-per-task=1 "$@"; }` and
`single_rank(){ srun --overlap -N1 -n1 --gpus-per-node=4 "$@"; }` (run_linewidth.sh:36-37). Model
run_dfpt_linewidth.sh on that (no `need()`); reuse the `INTERLAYER_ARGS` array (line 24).

**C11 — figure reuse (Task 8).** `_render_two_panel(name,cfg,x_skel,freqs_skel,hs_x,Xp,Yp,Gp,suffix,...)`
(linewidth_path.py:359) reads `cfg['label']` and, only when `out_dirs is None`, `cfg['root']` — pass
`out_dirs` explicitly + a `cfg` with a `'label'` key. Reuse `compute_band_structure` (76),
`gamma_on_path` (203), `lifetime_from_gamma` (284), `mesh_frequencies` (97).

C1–C11 above are the consolidated, actionable corrections; implement to them verbatim.

---

## Review findings R1–R7 (gpt-5.6-sol, 2026-07-15) — MUST-FIX before committing Phase-1

Sol reviewed the T1/T3/T4/T6 diff and found 5 P1 + 2 P2 defects the unit tests missed. Fix all:

- **R1 [P1] fc2 index convention (dfpt_read.py ~267-269).** QE `io_dyn_mat.f90:write_ifc`
  writes `phid(nn,:,:,s,s1)`; phonopy `PH_Q2R._parse_fc` stores `fc[s1, s*ndim+i_dim, ll, k]`.
  Current code uses `s` as row atom, `s1` as translated, and does NOT transpose the 3×3 block →
  silently wrong. **Fix:** assign `q2r_fc[s1-1, (s-1)*ndim+i_dim] = block.T`. (Unit tests can't
  catch this; the phband.freq dispersion gate in the compute phase is the definitive check.)
- **R2 [P1] Remove the ASR hack (dfpt_read.py ~277-281).** `q2r zasr='crystal'` (do_q2r.f90 →
  `set_zasr`) ASR's only the Born charges; the FC-ASR is applied by `matdyn asr='crystal'`
  (matdyn.f90 → `set_asr`, an iterative translational+symmetry projection) at diagonalization.
  Delete the uniform row-mean subtraction; **store raw IFCs**. Change the row-sum test that
  enshrined it (assert the realistic raw residual, or drop it). Phase-2 applies a faithful
  `set_asr('crystal')`-equivalent to a working copy at dyn-matrix build time (not baked in).
- **R3 [P1] Convert cell bohr→Å for the structure (dfpt_read.py ~287-289).** phonopy `read_pwscf`
  returns lengths in **bohr** even for `angstrom` input; `phonopy_atoms_to_ase` treats them as Å →
  cell 6.227 Å × 75.6 Å instead of 3.295 Å × 40 Å (1.889× too big) → corrupts ALL MLIP FC3 forces
  (Phase-1 too). **Fix:** keep the bohr cell for the q2r site-mapping, but multiply cell+positions
  by Bohr→Å (0.529177210903) when building `DfptData.structure`. **Add a test** asserting in-plane
  a ≈ 3.295 Å and out-of-plane c ≈ 40 Å.
- **R4 [P1] Keep one unit system (dfpt_read.py ~275-276).** Do NOT convert fc2 to eV/Å² while NAC
  stays in QE a.u. **Decision: keep fc2 raw Ry/bohr²**; document `nac['area']` as bohr², `nac['c']`
  as the out-of-plane repeat in bohr (rename/comment so it's not read as "speed of light"); Phase-2
  uses the QE THz factor 108.97077184367376 (not 15.6333).
- **R5 [P1] Gauge transform must be fully returned+applied (gauge.py ~201-204).** `select_gauge`
  may win by conjugation but returns only `sign` → not reproducible; its `qflip` only negates the
  phase, not the e_qe(q)→e at −q grid permutation. **Fix:** return a complete transform descriptor
  (sign, conjugation, and a real −q grid permutation) and an `apply_selected_gauge(...)` that
  reproduces it; select via reconstructed-dyn-matrix residual or degenerate-subspace overlap, not
  fixed-index per-mode overlaps.
- **R6 [P2] Fix matdyn test self-comparison (test_matdyn_modes.py ~47-52).** `frequencies_cm` and
  `ref` both load the same `phband.freq` → trivially zero. Compare the parsed `[THz]`→cm⁻¹ (or the
  `.modes` cm⁻¹ field) against the independent `phband.freq`.
- **R7 [P2] Per-stage reference-force filename (forces3.py ~84-85).** Both mains hardcode
  `reference_forces.npy`; the FC2-from-3 stage overwrites the FC3 one. Derive the name from the
  `--output` stem (e.g. `<stem>_reference_forces.npy`).

Also: register the `slow`/`gpu` pytest markers (pyproject/pytest.ini) to clear the warnings.
Note: `dfpt_fc2.npy` (R1/R2/R4) is Phase-2-critical; the DFPT **structure** (R3) is Phase-1-critical.
