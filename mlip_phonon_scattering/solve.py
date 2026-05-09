"""Solve harmonic phonons and export band/mesh eigenvectors."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ase.dft.kpoints import parse_path_string
from ase.io import read
from phonopy import load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


def reciprocal_rows(cell_ang: np.ndarray) -> np.ndarray:
    """Return reciprocal lattice vectors as rows in inverse Angstrom."""

    return 2.0 * np.pi * np.linalg.inv(cell_ang).T


def expand_full_fractional_mesh(
    q_frac: np.ndarray,
    bvec_rows: np.ndarray,
    frequencies_thz: np.ndarray,
    eig: np.ndarray,
    weights: np.ndarray,
):
    """Sort a gamma-centered mesh into a stable full fractional-grid order."""

    q_frac_reduced_sorted = q_frac.copy()
    q_frac_full = np.mod(q_frac, 1.0)
    q_frac_full[np.isclose(q_frac_full, 1.0, atol=1.0e-10)] = 0.0
    order = np.lexsort(
        (
            np.round(q_frac_full[:, 2], 10),
            np.round(q_frac_full[:, 1], 10),
            np.round(q_frac_full[:, 0], 10),
        )
    )
    q_frac_reduced_sorted = q_frac_reduced_sorted[order]
    q_frac_full = q_frac_full[order]
    q_cart_full = q_frac_full @ bvec_rows
    return q_frac_full, q_frac_reduced_sorted, q_cart_full, frequencies_thz[order], eig[order], weights[order]


def load_fractional_qpoints(path: Path) -> np.ndarray:
    """Load explicit fractional q-points from a whitespace text file."""

    qpoints = np.loadtxt(path, dtype=float)
    qpoints = np.atleast_2d(qpoints)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError(f"Expected q-points with shape (nq, 3), got {qpoints.shape} from {path}.")
    return qpoints


def expand_explicit_fractional_qpoints(
    q_frac_reduced: np.ndarray,
    bvec_rows: np.ndarray,
    frequencies_thz: np.ndarray,
    eig: np.ndarray,
):
    """Sort explicit q-points by wrapped coordinates while preserving reduced coordinates."""

    q_frac_full = np.mod(q_frac_reduced, 1.0)
    q_frac_full[np.isclose(q_frac_full, 1.0, atol=1.0e-10)] = 0.0
    order = np.lexsort(
        (
            np.round(q_frac_full[:, 2], 10),
            np.round(q_frac_full[:, 1], 10),
            np.round(q_frac_full[:, 0], 10),
        )
    )
    q_frac_full = q_frac_full[order]
    q_frac_reduced_sorted = q_frac_reduced[order]
    q_cart_full = q_frac_full @ bvec_rows
    return q_frac_full, q_frac_reduced_sorted, q_cart_full, frequencies_thz[order], eig[order]


def _write_common_metadata(h5, cell_ang, bvec_rows, tau_cart, tau_frac, masses_amu, species) -> None:
    h5.create_dataset("cell_ang", data=cell_ang)
    h5.create_dataset("bvec_rows_anginv", data=bvec_rows)
    h5.create_dataset("tau_cart_ang", data=tau_cart)
    h5.create_dataset("tau_frac", data=tau_frac)
    h5.create_dataset("masses_amu", data=masses_amu)
    h5.create_dataset("species", data=species)
    h5.attrs["freq_unit"] = "THz"
    h5.attrs["q_cart_unit"] = "1/Angstrom"


def _format_band_label(label: str) -> str:
    """Format compact ASE special-point labels for phonopy plots."""

    if label == "G":
        return "$\\Gamma$"
    return label


def _ase_band_path_from_structure(
    input_xyz: Path,
    band_path_string: str | None = None,
) -> tuple[list[list[list[float]]], list[str], str]:
    """Derive a phonopy band path from ASE's lattice-special-point machinery."""

    atoms = read(input_xyz)
    ase_band_path = atoms.cell.bandpath(path=band_path_string)
    branch_labels = parse_path_string(ase_band_path.path)
    special_points = ase_band_path.special_points

    path: list[list[list[float]]] = []
    labels: list[str] = []
    for branch in branch_labels:
        path.append([special_points[label].tolist() for label in branch])
        labels.extend(_format_band_label(label) for label in branch)

    return path, labels, ase_band_path.path


def solve_phonons(
    input_xyz: Path,
    forces_path: Path,
    phonopy_yaml: Path,
    output_dir: Path,
    mesh: tuple[int, int, int],
    qpoints_file: Path | None = None,
    band_npoints: int = 31,
    band_path: list[list[list[float]]] | None = None,
    band_labels: list[str] | None = None,
    band_path_string: str | None = None,
) -> None:
    """Build force constants, plot bands, and write eigenvector HDF5 files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    phonon = load(str(phonopy_yaml))
    phonon.forces = np.load(forces_path)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    nat = len(phonon.primitive)
    nmodes = 3 * nat
    cell_ang = np.array(phonon.primitive.cell, dtype=float)
    bvec_rows = reciprocal_rows(cell_ang)
    tau_cart = np.array(phonon.primitive.positions, dtype=float)
    tau_frac = np.array(phonon.primitive.scaled_positions, dtype=float)
    masses_amu = np.array(phonon.primitive.masses, dtype=float)
    species = np.array([s.encode("ascii", errors="ignore") for s in phonon.primitive.symbols])

    if band_path is None:
        band_path, derived_labels, derived_path_string = _ase_band_path_from_structure(
            input_xyz,
            band_path_string=band_path_string,
        )
        if band_labels is None:
            band_labels = derived_labels
    else:
        derived_path_string = band_path_string or "custom"
        if band_labels is None:
            raise ValueError("band_labels must be supplied when band_path is supplied directly.")

    qpoints, connections = get_band_qpoints_and_path_connections(band_path, npoints=band_npoints)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=band_labels, with_eigenvectors=True)

    band_plot = phonon.plot_band_structure()
    band_plot.xlabel("")
    band_plot.ylabel("Frequency (THz)", fontsize=14)
    band_plot.savefig(output_dir / "band_structure.png", dpi=240, bbox_inches="tight")
    plt.close()

    band_dict = phonon.get_band_structure_dict()
    band_q_frac_segs = [np.array(seg, dtype=float) for seg in band_dict["qpoints"]]
    band_freq_segs = [np.array(seg, dtype=float) for seg in band_dict["frequencies"]]
    band_eig_segs = [np.array(seg) for seg in band_dict["eigenvectors"]]
    band_dist_segs = [np.array(seg, dtype=float) for seg in band_dict["distances"]]

    band_q_frac = np.concatenate(band_q_frac_segs, axis=0)
    band_frequencies_thz = np.concatenate(band_freq_segs, axis=0)
    band_eig_flat = np.concatenate(band_eig_segs, axis=0)
    band_distances = np.concatenate(band_dist_segs, axis=0)
    band_segment_lengths = np.array([seg.shape[0] for seg in band_q_frac_segs], dtype=int)
    band_q_cart = band_q_frac @ bvec_rows
    band_eig = np.transpose(band_eig_flat, (0, 2, 1)).reshape(band_q_frac.shape[0], nmodes, nat, 3)
    band_labels_ascii = np.array([lab.encode("utf-8") for lab in band_labels])

    with h5py.File(output_dir / "eigenvector_band.h5", "w") as h5:
        h5.create_dataset("q_frac", data=band_q_frac)
        h5.create_dataset("q_cart_anginv", data=band_q_cart)
        h5.create_dataset("distances", data=band_distances)
        h5.create_dataset("segment_lengths", data=band_segment_lengths)
        h5.create_dataset("labels", data=band_labels_ascii)
        h5.create_dataset("frequencies_thz", data=band_frequencies_thz)
        h5.create_dataset("eigenvectors", data=band_eig)
        _write_common_metadata(h5, cell_ang, bvec_rows, tau_cart, tau_frac, masses_amu, species)
        h5.attrs["band_path"] = derived_path_string
        h5.attrs["description"] = "Phonopy band-path eigenvectors and metadata"

    if qpoints_file is None:
        phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True)
        mesh_dict = phonon.get_mesh_dict()
        q_frac_reduced = np.array(mesh_dict["qpoints"], dtype=float)
        frequencies_thz = np.array(mesh_dict["frequencies"], dtype=float)
        eig_flat = np.array(mesh_dict["eigenvectors"])
        weights = np.array(mesh_dict["weights"], dtype=int)
        eig = np.transpose(eig_flat, (0, 2, 1)).reshape(q_frac_reduced.shape[0], nmodes, nat, 3)
        q_frac, q_frac_reduced_sorted, q_cart, frequencies_thz, eig, weights = expand_full_fractional_mesh(
            q_frac_reduced,
            bvec_rows,
            frequencies_thz,
            eig,
            weights,
        )
        description = "Phonopy uniform-mesh eigenvectors and metadata for UED intensity calculations"
    else:
        q_frac_reduced = load_fractional_qpoints(qpoints_file)
        phonon.run_qpoints(q_frac_reduced, with_eigenvectors=True)
        qpoints_dict = phonon.get_qpoints_dict()
        frequencies_thz = np.array(qpoints_dict["frequencies"], dtype=float)
        eig_flat = np.array(qpoints_dict["eigenvectors"])
        weights = np.ones(q_frac_reduced.shape[0], dtype=int)
        eig = np.transpose(eig_flat, (0, 2, 1)).reshape(q_frac_reduced.shape[0], nmodes, nat, 3)
        q_frac, q_frac_reduced_sorted, q_cart, frequencies_thz, eig = expand_explicit_fractional_qpoints(
            q_frac_reduced,
            bvec_rows,
            frequencies_thz,
            eig,
        )
        weights = weights[np.lexsort(
            (
                np.round(np.mod(q_frac_reduced, 1.0)[:, 2], 10),
                np.round(np.mod(q_frac_reduced, 1.0)[:, 1], 10),
                np.round(np.mod(q_frac_reduced, 1.0)[:, 0], 10),
            )
        )]
        description = "Phonopy explicit-qpoint eigenvectors and metadata for UED intensity calculations"

    with h5py.File(output_dir / "eigenvector.h5", "w") as h5:
        h5.create_dataset("q_frac", data=q_frac)
        h5.create_dataset("q_frac_reduced", data=q_frac_reduced)
        h5.create_dataset("q_frac_reduced_sorted", data=q_frac_reduced_sorted)
        h5.create_dataset("q_cart_anginv", data=q_cart)
        h5.create_dataset("weights", data=weights)
        h5.create_dataset("frequencies_thz", data=frequencies_thz)
        h5.create_dataset("eigenvectors", data=eig)
        _write_common_metadata(h5, cell_ang, bvec_rows, tau_cart, tau_frac, masses_amu, species)
        h5.attrs["mesh"] = np.array(mesh, dtype=int)
        h5.attrs["mesh_centering"] = "gamma" if qpoints_file is None else "explicit"
        if qpoints_file is not None:
            h5.attrs["qpoints_file"] = str(qpoints_file)
        h5.attrs["description"] = description
