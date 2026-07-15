"""CPU-only tests for QE matdyn mode-file handling; never invokes matdyn.x."""

from pathlib import Path

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms
import phono3py

from mlip_phonon_scattering.linewidth.matdyn_modes import (
    load_matdyn_path,
    load_phband_freq,
    matdyn_qpoints_for_mesh,
    parse_matdyn_modes,
    run_matdyn,
)


ROOT = Path("/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect")
FLFRQ = ROOT / "phband.freq"
FLVEC = ROOT / "matdyn.modes"
MATDYN_IN = ROOT / "matdyn.in"


def test_matdyn_qpoints_for_mesh_uses_bzgrid_formula(tmp_path):
    """Use a saved minimal Phono3py cell to exercise the public YAML API end-to-end."""
    cell = PhonopyAtoms(
        symbols=["Si"],
        cell=np.diag([5.0, 5.0, 18.0]),
        scaled_positions=[[0.0, 0.0, 0.0]],
    )
    ph3 = phono3py.Phono3py(cell, supercell_matrix=[1, 1, 1])
    yaml_path = tmp_path / "phono3py.yaml"
    ph3.save(filename=str(yaml_path))
    mesh = [12, 12, 1]
    ga, q = matdyn_qpoints_for_mesh(str(yaml_path), mesh)
    check = phono3py.load(str(yaml_path), log_level=0)
    check.mesh_numbers = mesh
    assert ga.shape[0] >= np.prod(mesh)
    assert q.shape == (ga.shape[0], 3)
    assert np.array_equal(ga, check.grid.addresses)
    assert np.allclose(q, ga.astype(float) @ check.grid.QDinv.T)


def test_parse_reference_modes_and_phband_frequencies():
    masses = [95.95, 78.971, 78.971, 183.84, 78.971, 78.971]
    modes = parse_matdyn_modes(FLFRQ, FLVEC, masses)
    ref = load_phband_freq(FLFRQ)
    assert modes["eigenvectors"].shape == (152, 18, 6, 3)
    norms = np.linalg.norm(modes["eigenvectors"].reshape(152, 18, 18), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    parsed_cm = modes["frequencies"] * 33.35641
    assert np.abs(parsed_cm - ref).max() < 3.0


def test_load_phband_freq_reference_shape():
    assert load_phband_freq(FLFRQ).shape == (152, 18)


def test_load_matdyn_path_reference():
    path = load_matdyn_path(MATDYN_IN)
    assert path.shape == (152, 3)
    assert np.allclose(path[0], [0.0, 0.0, 0.0])
    assert np.all(np.abs(path) <= 1.0)


@pytest.mark.skip(reason="deferred to compute phase - requires matdyn.x binary, not available on this node")
def test_matdyn_run_phband_acceptance_gate(tmp_path):
    """Deferred gate: run, parse, then require max |Δ| < .5 and RMS < .1 cm^-1."""
    modes_path = run_matdyn("force_constants.xml", np.zeros((1, 3)), True, tmp_path)
    modes = parse_matdyn_modes(tmp_path / "phband.freq", modes_path, [1.0])
    delta = modes["frequencies_cm"] - load_phband_freq(tmp_path / "phband.freq")
    assert np.abs(delta).max() < 0.5
    assert np.sqrt(np.mean(delta**2)) < 0.1
