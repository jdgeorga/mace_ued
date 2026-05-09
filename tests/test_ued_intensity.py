from __future__ import annotations

import csv

import numpy as np

from mlip_phonon_scattering.ued_intensity import (
    PhononMeshData,
    TiledIntensityData,
    align_reduced_q_to_full_mesh,
    apply_phonopy_to_phx_eigenvector_gauge,
    center_fractional_coords,
    compute_temperature_dependent_ued,
    compute_zero_phonon_intensity,
    write_temperature_csv,
    write_wide_csv,
)


def test_zero_phonon_intensity_only_nonzero_at_reciprocal_lattice_points() -> None:
    q_frac_all = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [1.0, -2.0, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    q_cart_all = q_frac_all.copy()
    tau_frac = np.array([[0.0, 0.0, 0.0]], dtype=float)
    sigma_dw = np.zeros((1, 3, 3), dtype=float)

    _dw_exponent, _dw_factor, i0 = compute_zero_phonon_intensity(
        q_frac_all,
        q_cart_all,
        tau_frac,
        sigma_dw,
        ["Si"],
        electron_scattering_model="unity",
    )

    assert np.all(i0[[0, 2]] > 0.0)
    assert np.all(i0[[1, 3]] == 0.0)


def test_reduced_q_alignment_preserves_phonopy_reciprocal_image() -> None:
    q_frac_reduced = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.5333333333333333, 0.0],
            [0.3333333333333333, -0.6666666666666666, 0.0],
            [-0.6666666666666666, 0.3333333333333333, 0.0],
        ],
        dtype=float,
    )
    q_frac = np.mod(q_frac_reduced, 1.0)
    order = np.lexsort(
        (
            np.round(q_frac[:, 2], 10),
            np.round(q_frac[:, 1], 10),
            np.round(q_frac[:, 0], 10),
        )
    )
    q_frac = q_frac[order]

    aligned = align_reduced_q_to_full_mesh(q_frac, q_frac_reduced)

    assert np.allclose(np.mod(aligned, 1.0), q_frac)
    assert not np.allclose(aligned, center_fractional_coords(q_frac))
    assert np.any(np.abs(aligned - center_fractional_coords(q_frac)) > 0.5)


def test_phonopy_to_phx_eigenvector_gauge_applies_q_tau_phase() -> None:
    raw_eigenvectors = np.ones((2, 1, 2, 3), dtype=np.complex128)
    mesh = PhononMeshData(
        q_frac=np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.0]], dtype=float),
        q_reduced_frac=np.array([[0.0, 0.0, 0.0], [0.25, -0.5, 0.0]], dtype=float),
        bvec_rows_anginv=np.eye(3),
        tau_frac=np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]], dtype=float),
        masses_amu=np.array([1.0, 2.0], dtype=float),
        species=["A", "B"],
        frequencies_thz=np.ones((2, 1), dtype=float),
        eigenvectors=raw_eigenvectors,
    )

    gauged = apply_phonopy_to_phx_eigenvector_gauge(mesh)

    phase = np.exp(+2j * np.pi * (mesh.q_reduced_frac @ mesh.tau_frac.T))
    assert np.allclose(gauged.eigenvectors, raw_eigenvectors * phase[:, None, :, None])
    assert np.allclose(mesh.eigenvectors, raw_eigenvectors)


def test_wide_csv_combined_intensity_uses_sparse_zero_phonon_term(tmp_path) -> None:
    data = TiledIntensityData(
        q_frac_all=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=float),
        q_cart_all=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=float),
        source_iq=np.array([0, 1], dtype=int),
        h=np.array([0, 0], dtype=int),
        k=np.array([0, 0], dtype=int),
        omega_ev=np.array([[0.01], [0.02]], dtype=float),
        n0=np.array([[1.0], [2.0]], dtype=float),
        lq2=np.array([[3.0], [4.0]], dtype=float),
        dw_exponent=np.zeros((2, 1), dtype=float),
        dw_factor=np.ones((2, 1), dtype=float),
        zero_phonon_intensity=np.array([10.0, 0.0], dtype=float),
        one_phonon_structure_factor=np.array([[1.0], [2.0]], dtype=float),
        one_phonon_intensity=np.array([[5.0], [7.0]], dtype=float),
        species=["Si"],
    )

    out = write_wide_csv(tmp_path / "intensity.csv", data)

    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [float(row["combined_intensity"]) for row in rows] == [15.0, 7.0]


def test_temperature_dependent_ued_tracks_requested_bragg_points(tmp_path) -> None:
    mesh = PhononMeshData(
        q_frac=np.array([[0.0, 0.0, 0.0]], dtype=float),
        q_reduced_frac=np.array([[0.0, 0.0, 0.0]], dtype=float),
        bvec_rows_anginv=np.eye(3),
        tau_frac=np.array([[0.0, 0.0, 0.0]], dtype=float),
        masses_amu=np.array([1.0], dtype=float),
        species=["X"],
        frequencies_thz=np.array([[1.0]], dtype=float),
        eigenvectors=np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.complex128),
    )

    data = compute_temperature_dependent_ued(
        mesh,
        [0.0, 300.0],
        target_g_frac=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        electron_scattering_model="unity",
    )
    out = write_temperature_csv(tmp_path / "temperature.csv", data)

    assert data.dw_factor.shape == (2, 2, 1)
    assert data.zero_phonon_intensity.shape == (2, 2)
    assert np.allclose(data.target_g_frac, [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert data.dw_factor[1, 0, 0] < data.dw_factor[0, 0, 0]
    assert data.zero_phonon_intensity[1, 0] < data.zero_phonon_intensity[0, 0]
    assert out.exists()
