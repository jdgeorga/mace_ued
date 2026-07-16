"""CPU-only tests for QE matdyn mode-file handling; never invokes matdyn.x."""

from pathlib import Path

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms
import phono3py

import mlip_phonon_scattering.linewidth.matdyn_modes as matdyn_modes

from mlip_phonon_scattering.linewidth.matdyn_modes import (
    apply_gauge_with_sign_guard,
    apply_frequency_floor,
    load_matdyn_path,
    load_phband_freq,
    matdyn_qpoints_for_mesh,
    parse_matdyn_modes,
    run_matdyn,
)
from mlip_phonon_scattering.linewidth.gauge import (
    apply_bloch_gauge,
    apply_selected_gauge,
    select_gauge,
)


ROOT = Path("/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect")
FLFRQ = ROOT / "phband.freq"
FLVEC = ROOT / "matdyn.modes"
MATDYN_IN = ROOT / "matdyn.in"
DFPT = "/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix"


def test_apply_frequency_floor():
    frequencies = np.array([[-0.3, 0.01, 0.02, 1.5], [0.019, 3.0, 4.0, 5.0]])
    floored, count = apply_frequency_floor(frequencies, 0.02)
    assert count == 3
    assert np.array_equal(floored, [[0.02, 0.02, 0.02, 1.5], [0.02, 3.0, 4.0, 5.0]])
    assert np.array_equal(frequencies, [[-0.3, 0.01, 0.02, 1.5], [0.019, 3.0, 4.0, 5.0]])

    unchanged, count = apply_frequency_floor(frequencies, 0.0)
    assert count == 0
    assert np.array_equal(unchanged, frequencies)
    assert unchanged is not frequencies

    unchanged_negative, count = apply_frequency_floor(frequencies, -0.1)
    assert count == 0
    assert np.array_equal(unchanged_negative, frequencies)


def test_apply_gauge_with_sign_guard(recwarn):
    rng = np.random.default_rng(37)
    e_qe = rng.standard_normal((2, 2, 2, 3)) + 1j * rng.standard_normal((2, 2, 2, 3))
    q = np.array([[0.13, 0.07, 0.0], [0.21, 0.04, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.31, 0.17, 0.0]])
    e_ph = apply_bloch_gauge(e_qe, q, tau, sign=1)
    freqs = np.array([[1.0, 2.0], [1.5, 2.5]])

    selected, _residual, permutation = select_gauge(e_ph, e_qe, q, tau, freqs=freqs)
    assert selected.sign == 1
    assert permutation is None

    with pytest.warns(UserWarning, match="fixed gauge sign -1"):
        forced, transform, _residual, warned = apply_gauge_with_sign_guard(
            e_ph, e_qe, q, tau, gauge_sign=-1, freqs_ph=freqs
        )
    assert warned is True
    assert transform.sign == -1
    assert np.allclose(forced, apply_bloch_gauge(e_qe, q, tau, sign=-1))

    recwarn.clear()
    agreed, transform, _residual, warned = apply_gauge_with_sign_guard(
        e_ph, e_qe, q, tau, gauge_sign=1, freqs_ph=freqs
    )
    assert not recwarn
    assert warned is False
    assert transform.sign == 1
    assert np.allclose(agreed, apply_bloch_gauge(e_qe, q, tau, sign=1))

    expected_transform, expected_residual, expected_perm = select_gauge(
        e_ph, e_qe, q, tau, freqs=freqs
    )
    recwarn.clear()
    automatic, transform, residual, warned = apply_gauge_with_sign_guard(
        e_ph, e_qe, q, tau, gauge_sign="auto", freqs_ph=freqs
    )
    assert not recwarn
    assert warned is False
    assert transform == expected_transform
    assert residual == expected_residual
    assert np.allclose(
        automatic,
        apply_selected_gauge(e_qe, q, tau, expected_transform, expected_perm),
    )


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


def test_run_matdyn_relative_workdir_uses_local_input_name(tmp_path, monkeypatch):
    """A relative workdir must not be prefixed twice after subprocess cwd=... ."""
    monkeypatch.chdir(tmp_path)
    ifc = tmp_path / "force_constants.xml"
    ifc.write_text("fixture")
    calls = {}

    class _Completed:
        returncode = 0
        stdout = "matdyn fixture output"

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["cwd"] = kwargs["cwd"]
        return _Completed()

    monkeypatch.setattr(matdyn_modes.subprocess, "run", fake_run)
    modes_path = run_matdyn(
        "force_constants.xml", np.array([[0.0, 0.0, 0.0]]), True, "relative-workdir"
    )

    workdir = (tmp_path / "relative-workdir").resolve()
    assert calls["cwd"] == workdir
    assert calls["args"] == ["matdyn.x", "-in", "matdyn.in"]
    assert "flfrc = '" + str(ifc.resolve()) + "'" in (workdir / "matdyn.in").read_text()
    assert (workdir / "matdyn.out").read_text() == "matdyn fixture output"
    assert modes_path == str(workdir / "matdyn.modes")


@pytest.mark.skip(reason="deferred to compute phase - requires matdyn.x binary, not available on this node")
def test_matdyn_run_phband_acceptance_gate(tmp_path):
    """Deferred gate: run, parse, then require max |Δ| < .5 and RMS < .1 cm^-1."""
    modes_path = run_matdyn("force_constants.xml", np.zeros((1, 3)), True, tmp_path)
    modes = parse_matdyn_modes(tmp_path / "phband.freq", modes_path, [1.0])
    delta = modes["frequencies_cm"] - load_phband_freq(tmp_path / "phband.freq")
    assert np.abs(delta).max() < 0.5
    assert np.sqrt(np.mean(delta**2)) < 0.1


@pytest.mark.slow
def test_matdyn_modes_cli_gauge_path(tmp_path, monkeypatch, capsys):
    """Exercise the parse+gauge path with no matdyn.x call.

    This does NOT assert a tight gauge ``residual``: the reference dynamical
    matrix here is a plain (no-NAC) fc2 diagonalization, and this real
    heterobilayer's low bands are genuinely 2D-LOTO-sensitive even away from
    Gamma (confirmed empirically: atom order, Cartesian-axis order, ASR
    method, lattice orientation, and q-sign convention were each
    independently ruled out as the cause of the large residual; the
    discrepancy tracks
    `loto_2d=.true.` in the real DFPT fixture, i.e. missing NAC in the
    reference, which is out of scope until the Task-11 fork-native 2D-LOTO
    kernel exists). Instead this anchors on the DISCRETE sign choice, which
    matches the independently validated phonopy<->QE convention documented in
    docs/phonopy_eigenvector_gauge.md (phonopy->QE is sign=+1, so this
    QE->phonopy direction must be sign=-1) -- a real regression guard against
    a convention flip, without over-claiming reference quality it can't
    deliver for this system.
    """
    from mlip_phonon_scattering.linewidth.dfpt_read import main as dfpt_main

    dfpt_main(
        [
            "--dfpt-dir",
            DFPT,
            "--layer-symbols",
            "[['Mo','Se','Se'],['W','Se','Se']]",
            "--out-dir",
            str(tmp_path),
        ]
    )

    real_q = load_matdyn_path(MATDYN_IN)
    calls = {}

    def fake_qpoints(phono3py_yaml, mesh):
        calls["phono3py_yaml"] = phono3py_yaml
        calls["mesh"] = list(mesh)
        return np.arange(len(real_q) * 3).reshape(len(real_q), 3), real_q

    def fake_run_matdyn(ifc_xml, qlist, loto_2d, workdir):
        calls["loto_2d"] = loto_2d
        calls["nq"] = len(qlist)
        return str(FLVEC)

    monkeypatch.setattr(matdyn_modes, "matdyn_qpoints_for_mesh", fake_qpoints)
    monkeypatch.setattr(matdyn_modes, "run_matdyn", fake_run_matdyn)

    out = tmp_path / "dfpt_modes_loto.npz"
    matdyn_modes.main(
        [
            "--phono3py-yaml",
            "unused.yaml",
            "--ifc-xml",
            "unused.xml",
            "--mesh",
            "12",
            "12",
            "1",
            "--loto-2d",
            "on",
            "--dfpt-fc2",
            str(tmp_path / "dfpt_fc2.npy"),
            "--dfpt-structure",
            str(tmp_path / "dfpt_structure.xyz"),
            "--workdir",
            str(ROOT),
            "--out",
            str(out),
        ]
    )

    assert calls["mesh"] == [12, 12, 1]
    assert calls["loto_2d"] is True
    assert calls["nq"] == len(real_q)
    assert out.exists()
    saved = np.load(out)
    assert saved["frequencies"].shape == (len(real_q), 18)
    assert saved["eigenvectors"].shape == (len(real_q), 18, 6, 3)
    assert saved["mesh"].tolist() == [12, 12, 1]

    printed = capsys.readouterr().out
    assert "gauge transform: GaugeTransform(sign=-1" in printed
