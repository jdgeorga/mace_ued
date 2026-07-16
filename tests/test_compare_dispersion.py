"""Tests for the linewidth/lifetime dispersion comparison figures (Task 8).

Two layers of coverage:

* Pure / synthetic (no phono3py, always runs): the plot-ready core
  (``VariantDispersion`` -> figures + CSV) and the project tau = 1/(4 pi gamma) rule.
* End-to-end (``slow``): ``compare()`` driven through a genuine tiny 2-atom phono3py
  fixture (same idiom as ``tests/test_injection.py``), asserting the three deliverables.
"""

import csv
import numpy as np
import pytest

from mlip_phonon_scattering.linewidth.compare_dispersion import (
    VariantDispersion,
    _compare_core,
    _write_summary,
    compare,
)
from mlip_phonon_scattering.linewidth.linewidth_path import lifetime_from_gamma

_OUTPUTS = ("dispersion_linewidth.png", "dispersion_lifetime.png", "comparison_summary.csv")


# ---------------------------------------------------------------------------
# Pure project rule: tau derived from gamma (never interpolated).
# ---------------------------------------------------------------------------
def test_tau_from_gamma_rule():
    g = np.array([0.0, 1e-3, 1e-2])
    tau = lifetime_from_gamma(g)                     # 1/(4*pi*g); +inf at g == 0
    assert np.isinf(tau[0]) and np.all(tau[1:] > 0)
    assert np.allclose(tau[1:], 1.0 / (4.0 * np.pi * g[1:]))


# ---------------------------------------------------------------------------
# Pure core: synthetic VariantDispersion arrays -> figures + CSV.
# ---------------------------------------------------------------------------
def _synthetic_vdisp(label, seed, npts=61, nb=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 3.0, npts)
    hs_x = [0.0, 1.0, 2.0, 3.0]                       # Gamma, K, M, Gamma
    base = np.linspace(0.5, 9.0, nb)[None, :]         # branch baseline (THz)
    freqs = base + 0.4 * np.abs(rng.normal(size=(npts, nb)))
    gamma = 0.05 * np.abs(rng.normal(size=(npts, nb)))
    return VariantDispersion(label=label, x=x, freqs=freqs, hs_x=hs_x, gamma_path=gamma)


def _assert_outputs_nonempty(out_dir):
    for name in _OUTPUTS:
        p = out_dir / name
        assert p.exists(), f"missing {name}"
        assert p.stat().st_size > 0, f"empty {name}"
    # both figures are also written as vector PDF
    for stem in ("dispersion_linewidth", "dispersion_lifetime"):
        pdf = out_dir / f"{stem}.pdf"
        assert pdf.exists() and pdf.stat().st_size > 0, f"missing/empty {stem}.pdf"


def test_core_pure_synthetic_two_variants(tmp_path):
    vds = [_synthetic_vdisp("dfpt_harm", 1), _synthetic_vdisp("mlip_strict", 2)]
    _compare_core(vds, tmp_path, temperature=50)
    _assert_outputs_nonempty(tmp_path)

    text = (tmp_path / "comparison_summary.csv").read_text()
    # per-variant rows at both high-symmetry points, plus the labelled pairwise ratio
    assert "dfpt_harm" in text and "mlip_strict" in text
    assert "dfpt_harm / mlip_strict" in text
    assert ",K," in text and ",M," in text
    assert "variant" in text and "ratio" in text


def test_core_single_variant(tmp_path):
    """A single variant still renders (one panel) and has no ratio rows."""
    _compare_core([_synthetic_vdisp("only", 3)], tmp_path, temperature=77)
    _assert_outputs_nonempty(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "comparison_summary.csv")))
    assert rows and all(r["kind"] == "variant" for r in rows)


def test_summary_derives_tau_from_gamma(tmp_path):
    """CSV top-optical tau at K/M equals 1/(4*pi*gamma) of the interpolated gamma."""
    npts, nb = 61, 6
    x = np.linspace(0.0, 3.0, npts)                   # K at x=1 (idx 20), M at x=2 (idx 40)
    hs_x = [0.0, 1.0, 2.0, 3.0]
    freqs = np.tile(np.arange(nb, dtype=float)[None, :], (npts, 1))  # top branch = col nb-1
    gamma = np.zeros((npts, nb))
    gK, gM = 0.01, 0.02
    gamma[20, nb - 1] = gK
    gamma[40, nb - 1] = gM
    vd = VariantDispersion(label="v", x=x, freqs=freqs, hs_x=hs_x, gamma_path=gamma)

    _write_summary([vd], tmp_path)
    rows = {(r["label"], r["point"]): r
            for r in csv.DictReader(open(tmp_path / "comparison_summary.csv"))}
    assert np.isclose(float(rows[("v", "K")]["gamma_THz"]), gK)
    assert np.isclose(float(rows[("v", "K")]["tau_ps"]), 1.0 / (4.0 * np.pi * gK), rtol=1e-4)
    assert np.isclose(float(rows[("v", "M")]["tau_ps"]), 1.0 / (4.0 * np.pi * gM), rtol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end: compare() via a real tiny phono3py fixture.
# ---------------------------------------------------------------------------
def _toy_variant(tmp_path, name, mesh, seed):
    """Build a genuine 2-atom phono3py yaml + fc2 + synthetic gamma npz on `mesh`."""
    import phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    a, c = 3.3, 20.0
    cell = np.array([[a, 0.0, 0.0],
                     [-a / 2, a * np.sqrt(3) / 2, 0.0],
                     [0.0, 0.0, c]])
    atoms = PhonopyAtoms(
        symbols=["Mo", "Se"], cell=cell,
        scaled_positions=[[0.0, 0.0, 0.5], [1.0 / 3, 2.0 / 3, 0.55]],
        masses=[95.95, 78.971],
    )
    ph3 = phono3py.Phono3py(
        atoms, supercell_matrix=[2, 2, 1], phonon_supercell_matrix=[2, 2, 1], log_level=0,
    )
    ph3.generate_displacements()
    rng = np.random.default_rng(seed)
    nsc = len(ph3.phonon_supercell)
    ph3.phonon_forces = rng.normal(
        scale=1e-3, size=(len(ph3.phonon_supercells_with_displacements), nsc, 3)
    )
    ph3.produce_fc2(symmetrize_fc2=True)

    yaml = tmp_path / f"{name}.yaml"
    ph3.save(str(yaml))
    fc2 = tmp_path / f"{name}_fc2.npy"
    np.save(fc2, ph3.fc2)

    nb = 3 * len(ph3.primitive)
    mx, my, _mz = mesh
    q = np.array([[i / mx, j / my, 0.0] for i in range(mx) for j in range(my)], float)
    gamma = np.abs(rng.normal(scale=0.05, size=(q.shape[0], nb)))
    gnpz = tmp_path / f"{name}_gamma.npz"
    np.savez(gnpz, qpoints_frac=q, gamma_W_full=gamma)

    return dict(label=name, phono3py_yaml=str(yaml), fc2_path=str(fc2), gamma_npz=str(gnpz))


@pytest.mark.slow
def test_compare_end_to_end(tmp_path):
    mesh = [4, 4, 1]
    try:
        variants = [
            _toy_variant(tmp_path, "dfpt_harm", mesh, 1),
            _toy_variant(tmp_path, "mlip_strict", mesh, 2),
        ]
    except Exception as exc:  # pragma: no cover - environment/API guard only
        pytest.skip(f"could not build phono3py fixture: {exc!r}")

    # compare() runs OUTSIDE the guard so real bugs surface as failures, not skips.
    compare(variants, mesh=mesh, temperature=50, out_dir=tmp_path, npoints=21)
    _assert_outputs_nonempty(tmp_path)

    text = (tmp_path / "comparison_summary.csv").read_text()
    assert "dfpt_harm / mlip_strict" in text
