"""UED zero- and one-phonon intensity calculations for phonopy HDF5 output."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from mlip_phonon_scattering.electron_scattering_factors import (
    HIGH_S_MAX_INV_A,
    LOW_S_MAX_INV_A,
    PENG_NEUTRAL_HIGH_S,
    PENG_NEUTRAL_LOW_S,
)

ANG_TO_M = 1.0e-10
AMU_SI = 1.66053906660e-27
HBAR_SI = 1.054571817e-34
ELECTRONVOLT_TO_J = 1.602176634e-19
THZ_TO_EV = 4.135667696e-3
K_BOLTZMANN_EV_PER_K = 8.617333262145e-5
EPS = 1.0e-18


@dataclass(frozen=True)
class PhononMeshData:
    q_frac: np.ndarray
    q_reduced_frac: np.ndarray
    bvec_rows_anginv: np.ndarray
    tau_frac: np.ndarray
    masses_amu: np.ndarray
    species: list[str]
    frequencies_thz: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class TiledIntensityData:
    q_frac_all: np.ndarray
    q_cart_all: np.ndarray
    source_iq: np.ndarray
    h: np.ndarray
    k: np.ndarray
    omega_ev: np.ndarray
    n0: np.ndarray
    lq2: np.ndarray
    dw_exponent: np.ndarray
    dw_factor: np.ndarray
    zero_phonon_intensity: np.ndarray
    one_phonon_structure_factor: np.ndarray
    one_phonon_intensity: np.ndarray
    species: list[str]


@dataclass(frozen=True)
class TemperatureDependentUEDData:
    temperatures_k: np.ndarray
    target_g_frac: np.ndarray
    target_q_cart: np.ndarray
    dw_factor: np.ndarray
    zero_phonon_intensity: np.ndarray
    species: list[str]


def apply_phonopy_to_phx_eigenvector_gauge(mesh: PhononMeshData) -> PhononMeshData:
    """Convert phonopy HDF5 eigenvectors to the PH.x-style UED phase gauge."""

    phase = np.exp(+2j * np.pi * (mesh.q_reduced_frac @ mesh.tau_frac.T))
    return replace(mesh, eigenvectors=mesh.eigenvectors * phase[:, None, :, None])


def load_phonon_mesh_h5(path: Path, eigenvector_gauge: str = "raw") -> PhononMeshData:
    """Load the uniform-mesh phonon data written by ``solve_phonons.py``."""

    with h5py.File(path, "r") as h5:
        q_frac = np.array(h5["q_frac"], dtype=float)
        species = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["species"][:]]
        mesh = PhononMeshData(
            q_frac=q_frac,
            q_reduced_frac=load_reduced_q_for_eigenvectors(h5, q_frac),
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


def center_fractional_coords(q_frac: np.ndarray) -> np.ndarray:
    """Map fractional q coordinates from [0, 1) to a Gamma-centered cell."""

    return ((q_frac + 0.5) % 1.0) - 0.5


def _wrapped_fractional_coords(q_frac: np.ndarray) -> np.ndarray:
    q_wrapped = np.mod(q_frac, 1.0)
    q_wrapped[np.isclose(q_wrapped, 1.0, atol=1.0e-10)] = 0.0
    return q_wrapped


def _fractional_mesh_sort_order(q_frac_wrapped: np.ndarray) -> np.ndarray:
    return np.lexsort(
        (
            np.round(q_frac_wrapped[:, 2], 10),
            np.round(q_frac_wrapped[:, 1], 10),
            np.round(q_frac_wrapped[:, 0], 10),
        )
    )


def align_reduced_q_to_full_mesh(q_frac: np.ndarray, q_frac_reduced: np.ndarray) -> np.ndarray:
    """Return reduced q coordinates in the same order as wrapped mesh eigenvectors."""

    q_reduced_wrapped = _wrapped_fractional_coords(q_frac_reduced)
    order = _fractional_mesh_sort_order(q_reduced_wrapped)
    q_reduced_sorted = q_frac_reduced[order]
    q_reduced_wrapped_sorted = q_reduced_wrapped[order]
    if q_reduced_sorted.shape != q_frac.shape or not np.allclose(
        q_reduced_wrapped_sorted,
        q_frac,
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise ValueError("Stored reduced q mesh cannot be aligned with wrapped q_frac eigenvector order.")
    return q_reduced_sorted


def load_reduced_q_for_eigenvectors(h5: h5py.File, q_frac: np.ndarray) -> np.ndarray:
    """Load the q coordinates whose phase convention matches each stored eigenvector."""

    if "q_frac_reduced_sorted" in h5:
        return np.array(h5["q_frac_reduced_sorted"], dtype=float)
    if "q_frac_reduced" in h5:
        return align_reduced_q_to_full_mesh(q_frac, np.array(h5["q_frac_reduced"], dtype=float))
    return center_fractional_coords(q_frac)


def hex_shell_index(h: int, k: int) -> int:
    return max(abs(h), abs(k), abs(h + k))


def build_tiled_q_grid(
    q_frac: np.ndarray,
    bvec_rows: np.ndarray,
    gmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hk_list = [
        (h, k)
        for h in range(-gmax, gmax + 1)
        for k in range(-gmax, gmax + 1)
        if hex_shell_index(h, k) <= gmax
    ]
    nq = q_frac.shape[0]
    npts = nq * len(hk_list)
    q_frac_all = np.zeros((npts, 3), dtype=float)
    q_cart_all = np.zeros((npts, 3), dtype=float)
    source_iq = np.zeros(npts, dtype=int)
    h_all = np.zeros(npts, dtype=int)
    k_all = np.zeros(npts, dtype=int)
    for iq in range(nq):
        for j, (h, k) in enumerate(hk_list):
            idx = iq * len(hk_list) + j
            qf = q_frac[iq] + np.array([h, k, 0.0], dtype=float)
            q_frac_all[idx] = qf
            q_cart_all[idx] = qf @ bvec_rows
            source_iq[idx] = iq
            h_all[idx] = h
            k_all[idx] = k
    return q_frac_all, q_cart_all, source_iq, h_all, k_all


def get_atomic_number(symbol: str) -> int:
    try:
        from ase.data import atomic_numbers  # type: ignore

        return int(atomic_numbers[symbol])
    except Exception as exc:
        raise KeyError(f"No atomic number available for species {symbol!r}; install ASE or use Peng-table species.") from exc


def _evaluate_five_gaussian(s_inv_a: np.ndarray, a_coeff: Sequence[float], b_coeff: Sequence[float]) -> np.ndarray:
    a = np.asarray(a_coeff, dtype=float)
    b = np.asarray(b_coeff, dtype=float)
    return np.sum(a[None, :] * np.exp(-b[None, :] * s_inv_a[:, None] ** 2), axis=1)


def elastic_electron_atomic_scattering_factors(
    q_norm_inv_a: np.ndarray,
    species: Sequence[str],
    model: str = "peng",
) -> np.ndarray:
    """Return neutral-atom elastic electron scattering amplitudes.

    The default `peng` model selects the low-s or high-s Peng/Ren/Dudarev/Whelan
    coefficient block point-by-point using s = |Q|/(4*pi). Values above the
    tabulated high-s range use the high-s block with an explicit warning.
    """

    amplitudes = np.zeros((q_norm_inv_a.shape[0], len(species)), dtype=float)
    s_inv_a = q_norm_inv_a / (4.0 * np.pi)
    if np.any(s_inv_a > HIGH_S_MAX_INV_A):
        warnings.warn(
            "Some Q values have s = |Q|/(4*pi) > 6 A^-1; extrapolating the Peng high-s "
            "electron scattering-factor coefficients.",
            RuntimeWarning,
            stacklevel=2,
        )
    if model == "vand5_or_z":
        warnings.warn(
            "electron_scattering_model='vand5_or_z' is deprecated; use 'peng' for range-aware "
            "neutral-atom elastic electron scattering factors.",
            DeprecationWarning,
            stacklevel=2,
        )
        model = "peng"
    elif model == "vand5":
        warnings.warn(
            "electron_scattering_model='vand5' is deprecated and applies only the high-s Peng "
            "coefficient block; use 'peng' for range-aware coefficients.",
            DeprecationWarning,
            stacklevel=2,
        )
        model = "peng_high_s"

    for ia, sym in enumerate(species):
        if model == "unity":
            amplitudes[:, ia] = 1.0
        elif model == "atomic_number":
            amplitudes[:, ia] = float(get_atomic_number(sym))
        elif model == "peng_low_s":
            low_a, low_b = PENG_NEUTRAL_LOW_S[sym]
            amplitudes[:, ia] = _evaluate_five_gaussian(s_inv_a, low_a, low_b)
        elif model == "peng_high_s":
            if np.any(s_inv_a <= LOW_S_MAX_INV_A):
                warnings.warn(
                    "Applying the high-s Peng coefficient block to Q values with s <= 2 A^-1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            high_a, high_b = PENG_NEUTRAL_HIGH_S[sym]
            amplitudes[:, ia] = _evaluate_five_gaussian(s_inv_a, high_a, high_b)
        elif model == "peng":
            low_a, low_b = PENG_NEUTRAL_LOW_S[sym]
            high_a, high_b = PENG_NEUTRAL_HIGH_S[sym]
            low_mask = s_inv_a <= LOW_S_MAX_INV_A
            high_mask = ~low_mask
            if np.any(low_mask):
                amplitudes[low_mask, ia] = _evaluate_five_gaussian(s_inv_a[low_mask], low_a, low_b)
            if np.any(high_mask):
                amplitudes[high_mask, ia] = _evaluate_five_gaussian(s_inv_a[high_mask], high_a, high_b)
        else:
            raise ValueError(f"Unknown electron scattering-factor model {model!r}.")
    return amplitudes


def bose_occupation(omega_ev: np.ndarray, temperature_k: float) -> np.ndarray:
    if temperature_k < 0.0:
        raise ValueError("temperature_k must be >= 0")
    if temperature_k == 0.0:
        return np.zeros_like(omega_ev, dtype=float)
    x = np.abs(omega_ev) / (K_BOLTZMANN_EV_PER_K * temperature_k)
    x_clip = np.clip(x, 0.0, 700.0)
    out = np.zeros_like(omega_ev, dtype=float)
    mask = x_clip > 1.0e-12
    out[mask] = 1.0 / np.expm1(x_clip[mask])
    return out


def build_lq2(nphon0: np.ndarray, omega_ev: np.ndarray, phwmin_mev: float) -> np.ndarray:
    phwmin_ev = phwmin_mev * 1.0e-3
    omega_clipped = np.maximum(np.abs(omega_ev), phwmin_ev)
    omega_rad = (omega_clipped * ELECTRONVOLT_TO_J) / HBAR_SI
    pref = HBAR_SI / (2.0 * AMU_SI * (ANG_TO_M**2))
    lq2 = pref * (2.0 * nphon0 + 1.0) / omega_rad
    lq2[np.abs(omega_ev) < phwmin_ev] = 0.0
    return lq2


def q_is_set_a(qvec_frac: np.ndarray, tol: float = 1.0e-8) -> bool:
    doubled = 2.0 * qvec_frac
    mods = np.mod(np.abs(doubled), 1.0)
    return bool(np.all((mods < tol) | (np.abs(mods - 1.0) < tol)))


def q_is_integer_reciprocal(qvec_frac: np.ndarray, tol: float = 1.0e-8) -> bool:
    mods = np.mod(np.abs(qvec_frac), 1.0)
    return bool(np.all((mods < tol) | (np.abs(mods - 1.0) < tol)))


def build_dw_tensor(
    lq2: np.ndarray,
    q_frac: np.ndarray,
    eig: np.ndarray,
    atom_mass_amu: np.ndarray,
) -> np.ndarray:
    nq, _nmodes = lq2.shape
    nat = atom_mass_amu.shape[0]
    dw_weight_per_q = np.array([0.5 if q_is_set_a(q_frac[iq]) else 1.0 for iq in range(nq)], dtype=float)
    fac = (dw_weight_per_q[:, None] * lq2) / float(nq)
    sigma_dw = np.zeros((nat, 3, 3), dtype=float)
    for ia in range(nat):
        em = eig[:, :, ia, :]
        outer = np.einsum("qna,qnb->qnab", em, np.conjugate(em)).real
        sigma_dw[ia] = np.sum(fac[:, :, None, None] * outer, axis=(0, 1)) / max(float(atom_mass_amu[ia]), EPS)
    return sigma_dw


def compute_zero_phonon_intensity(
    q_frac_all: np.ndarray,
    q_cart_all: np.ndarray,
    tau_frac: np.ndarray,
    sigma_dw: np.ndarray,
    atom_species: Sequence[str],
    electron_scattering_model: str = "peng",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Debye-Waller factors and elastic zero-phonon intensity.

    The elastic contribution is enforced to satisfy delta_{Q,G} over the full
    3D reciprocal lattice, so I0 is non-zero only at exact reciprocal vectors.
    """

    qnorm = np.linalg.norm(q_cart_all, axis=1)
    ff = elastic_electron_atomic_scattering_factors(qnorm, atom_species, model=electron_scattering_model)
    phase = np.exp(-1j * 2.0 * np.pi * (q_frac_all @ tau_frac.T))
    dw_exponent = np.einsum("pi,pj,aij->pa", q_cart_all, q_cart_all, sigma_dw, optimize=True)
    dw_factor = np.exp(-dw_exponent)
    f0 = np.sum(ff * dw_factor * phase, axis=1)
    i0 = np.real(f0 * np.conjugate(f0))
    reciprocal_lattice_mask = np.array([q_is_integer_reciprocal(q_frac_all[ip]) for ip in range(q_frac_all.shape[0])])
    i0[~reciprocal_lattice_mask] = 0.0
    return dw_exponent, dw_factor, i0


def compute_one_phonon_intensity(
    q_frac_all: np.ndarray,
    q_cart_all: np.ndarray,
    source_iq: np.ndarray,
    tau_frac: np.ndarray,
    eig: np.ndarray,
    lq2: np.ndarray,
    sigma_dw: np.ndarray,
    atom_species: Sequence[str],
    atom_mass_amu: np.ndarray,
    electron_scattering_model: str = "peng",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mode-resolved one-phonon structure factors and intensities."""

    npts = q_cart_all.shape[0]
    _nq, nmodes = lq2.shape
    nat = atom_mass_amu.shape[0]
    qnorm = np.linalg.norm(q_cart_all, axis=1)
    ff = elastic_electron_atomic_scattering_factors(qnorm, atom_species, model=electron_scattering_model)
    phase = np.exp(-1j * 2.0 * np.pi * (q_frac_all @ tau_frac.T))
    dw_exponent = np.einsum("pi,pj,aij->pa", q_cart_all, q_cart_all, sigma_dw, optimize=True)
    dw_factor = np.exp(-dw_exponent)

    qdot_by_ip = np.zeros((npts, nmodes, nat), dtype=np.complex128)
    for iq in range(eig.shape[0]):
        mask = source_iq == iq
        if np.any(mask):
            qdot_by_ip[mask] = np.einsum("nax,px->pna", eig[iq], q_cart_all[mask], optimize=True)
    sqrt_mass = np.sqrt(np.maximum(atom_mass_amu, EPS))
    amp = np.sum(
        ff[:, None, :] * dw_factor[:, None, :] * phase[:, None, :] * qdot_by_ip / sqrt_mass[None, None, :],
        axis=2,
    )
    s1 = np.real(amp * np.conjugate(amp))
    lq2_eff = lq2[source_iq].copy()
    integer_reciprocal = np.array([q_is_integer_reciprocal(q_frac_all[ip]) for ip in range(npts)])
    lq2_eff[integer_reciprocal, : min(3, nmodes)] = 0.0
    return s1, lq2_eff * s1


def compute_tiled_intensities(
    mesh: PhononMeshData,
    *,
    temperature_k: float = 100.0,
    phwmin_mev: float = 0.2,
    gmax: int = 3,
    electron_scattering_model: str = "peng",
) -> TiledIntensityData:
    """Compute tiled zero- and one-phonon intensities from a phonon mesh."""

    q_frac = mesh.q_reduced_frac
    omega_ev = mesh.frequencies_thz * THZ_TO_EV
    n0 = bose_occupation(omega_ev, temperature_k=temperature_k)
    lq2 = build_lq2(n0, omega_ev, phwmin_mev=phwmin_mev)
    sigma_dw = build_dw_tensor(lq2, q_frac=q_frac, eig=mesh.eigenvectors, atom_mass_amu=mesh.masses_amu)
    q_frac_all, q_cart_all, source_iq, h_all, k_all = build_tiled_q_grid(q_frac, mesh.bvec_rows_anginv, gmax=gmax)
    dw_exp, dw_factor, i0 = compute_zero_phonon_intensity(
        q_frac_all,
        q_cart_all,
        mesh.tau_frac,
        sigma_dw,
        mesh.species,
        electron_scattering_model=electron_scattering_model,
    )
    s1, i1 = compute_one_phonon_intensity(
        q_frac_all,
        q_cart_all,
        source_iq,
        mesh.tau_frac,
        mesh.eigenvectors,
        lq2,
        sigma_dw,
        mesh.species,
        mesh.masses_amu,
        electron_scattering_model=electron_scattering_model,
    )
    return TiledIntensityData(
        q_frac_all=q_frac_all,
        q_cart_all=q_cart_all,
        source_iq=source_iq,
        h=h_all,
        k=k_all,
        omega_ev=omega_ev,
        n0=n0,
        lq2=lq2,
        dw_exponent=dw_exp,
        dw_factor=dw_factor,
        zero_phonon_intensity=i0,
        one_phonon_structure_factor=s1,
        one_phonon_intensity=i1,
        species=mesh.species,
    )


def compute_temperature_dependent_ued(
    mesh: PhononMeshData,
    temperatures_k: Sequence[float],
    *,
    target_g_frac: Sequence[Sequence[float]] = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    phwmin_mev: float = 0.2,
    electron_scattering_model: str = "peng",
) -> TemperatureDependentUEDData:
    """Compute Bragg-point Debye-Waller factors and elastic intensities versus T."""

    temperatures = np.asarray(temperatures_k, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures_k must be a non-empty one-dimensional sequence.")
    if np.any(temperatures < 0.0):
        raise ValueError("temperatures_k values must be >= 0.")

    g_frac = np.asarray(target_g_frac, dtype=float)
    if g_frac.ndim != 2 or g_frac.shape[1] != 3:
        raise ValueError("target_g_frac must have shape (n_targets, 3).")

    omega_ev = mesh.frequencies_thz * THZ_TO_EV
    q_cart = g_frac @ mesh.bvec_rows_anginv
    dw_factor = np.zeros((temperatures.size, g_frac.shape[0], len(mesh.species)), dtype=float)
    zero_phonon = np.zeros((temperatures.size, g_frac.shape[0]), dtype=float)

    for itemp, temperature_k in enumerate(temperatures):
        n0 = bose_occupation(omega_ev, temperature_k=float(temperature_k))
        lq2 = build_lq2(n0, omega_ev, phwmin_mev=phwmin_mev)
        sigma_dw = build_dw_tensor(lq2, q_frac=mesh.q_reduced_frac, eig=mesh.eigenvectors, atom_mass_amu=mesh.masses_amu)
        _dw_exp, dw, i0 = compute_zero_phonon_intensity(
            g_frac,
            q_cart,
            mesh.tau_frac,
            sigma_dw,
            mesh.species,
            electron_scattering_model=electron_scattering_model,
        )
        dw_factor[itemp] = dw
        zero_phonon[itemp] = i0

    return TemperatureDependentUEDData(
        temperatures_k=temperatures,
        target_g_frac=g_frac,
        target_q_cart=q_cart,
        dw_factor=dw_factor,
        zero_phonon_intensity=zero_phonon,
        species=mesh.species,
    )


def qz_mask(data: TiledIntensityData, qz: float = 0.0, qz_tol: float = 1.0e-8) -> np.ndarray:
    return np.isclose(data.q_cart_all[:, 2], qz, atol=qz_tol)


def shortest_inplane_primitive_g_modulus(bvec_rows_anginv: np.ndarray) -> float:
    """Smallest modulus of non-zero (h,k) reciprocal vectors in the a–b plane (rows 0,1)."""

    b0 = np.asarray(bvec_rows_anginv[0], dtype=float)
    b1 = np.asarray(bvec_rows_anginv[1], dtype=float)
    return float(
        np.min(
            [
                np.linalg.norm(b0),
                np.linalg.norm(b1),
                np.linalg.norm(b0 - b1),
                np.linalg.norm(b0 + b1),
            ]
        )
    )


def integer_reciprocal_mask(q_frac_all: np.ndarray) -> np.ndarray:
    """Vectorized mask for exact reciprocal-lattice points in fractional coordinates."""

    return np.array([q_is_integer_reciprocal(q_frac_all[ip]) for ip in range(q_frac_all.shape[0])])


def write_wide_csv(path: Path, data: TiledIntensityData, mask: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(data.q_cart_all.shape[0]) if mask is None else np.nonzero(mask)[0]
    atom_labels = [f"atom{ia + 1}_{sym}" for ia, sym in enumerate(data.species)]
    nmodes = data.omega_ev.shape[1]
    header = [
        "point_index",
        "source_iq",
        "h",
        "k",
        "Q_frac_h",
        "Q_frac_k",
        "Q_frac_l",
        "Qx_invA",
        "Qy_invA",
        "Qz_invA",
        "zero_phonon_intensity",
        "one_phonon_structure_factor_total",
        "one_phonon_intensity_total",
        "combined_intensity",
    ]
    for label in atom_labels:
        header.extend([f"dw_exponent_{label}", f"dw_factor_{label}"])
    for nu in range(nmodes):
        header.extend(
            [
                f"omega_eV_mode_{nu + 1}",
                f"n0_mode_{nu + 1}",
                f"lq2_A2_mode_{nu + 1}",
                f"one_phonon_structure_factor_mode_{nu + 1}",
                f"one_phonon_intensity_mode_{nu + 1}",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for ip in rows:
            iq = int(data.source_iq[ip])
            s1_total = float(np.sum(data.one_phonon_structure_factor[ip]))
            i1_total = float(np.sum(data.one_phonon_intensity[ip]))
            row = [
                int(ip),
                iq,
                int(data.h[ip]),
                int(data.k[ip]),
                *data.q_frac_all[ip],
                *data.q_cart_all[ip],
                data.zero_phonon_intensity[ip],
                s1_total,
                i1_total,
                data.zero_phonon_intensity[ip] + i1_total,
            ]
            for ia in range(len(data.species)):
                row.extend([data.dw_exponent[ip, ia], data.dw_factor[ip, ia]])
            for nu in range(nmodes):
                row.extend(
                    [
                        data.omega_ev[iq, nu],
                        data.n0[iq, nu],
                        data.lq2[iq, nu],
                        data.one_phonon_structure_factor[ip, nu],
                        data.one_phonon_intensity[ip, nu],
                    ]
                )
            writer.writerow(row)
    return path


def write_long_csv(path: Path, data: TiledIntensityData, mask: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(data.q_cart_all.shape[0]) if mask is None else np.nonzero(mask)[0]
    atom_labels = [f"atom{ia + 1}_{sym}" for ia, sym in enumerate(data.species)]
    header = [
        "point_index",
        "source_iq",
        "mode",
        "h",
        "k",
        "Q_frac_h",
        "Q_frac_k",
        "Q_frac_l",
        "Qx_invA",
        "Qy_invA",
        "Qz_invA",
        "omega_eV",
        "n0",
        "lq2_A2",
        "zero_phonon_intensity",
        "one_phonon_structure_factor",
        "one_phonon_intensity",
    ]
    for label in atom_labels:
        header.extend([f"dw_exponent_{label}", f"dw_factor_{label}"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for ip in rows:
            iq = int(data.source_iq[ip])
            for nu in range(data.omega_ev.shape[1]):
                row = [
                    int(ip),
                    iq,
                    nu + 1,
                    int(data.h[ip]),
                    int(data.k[ip]),
                    *data.q_frac_all[ip],
                    *data.q_cart_all[ip],
                    data.omega_ev[iq, nu],
                    data.n0[iq, nu],
                    data.lq2[iq, nu],
                    data.zero_phonon_intensity[ip],
                    data.one_phonon_structure_factor[ip, nu],
                    data.one_phonon_intensity[ip, nu],
                ]
                for ia in range(len(data.species)):
                    row.extend([data.dw_exponent[ip, ia], data.dw_factor[ip, ia]])
                writer.writerow(row)
    return path


def _g_label(g_frac: np.ndarray) -> str:
    values = []
    for value in g_frac:
        if np.isclose(value, round(value), atol=1.0e-10):
            values.append(str(int(round(value))))
        else:
            values.append(f"{value:g}")
    return "(" + ",".join(values) + ")"


def _species_group_indices(species: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    species_array = np.asarray(species)
    for sym in species:
        if sym in seen:
            continue
        seen.add(sym)
        groups.append((sym, np.nonzero(species_array == sym)[0]))
    return groups


def write_temperature_csv(path: Path, data: TemperatureDependentUEDData) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_labels = [f"atom{ia + 1}_{sym}" for ia, sym in enumerate(data.species)]
    header = ["temperature_K"]
    for ig, g_frac in enumerate(data.target_g_frac):
        label = _g_label(g_frac)
        header.extend(
            [
                f"G{ig + 1}_h",
                f"G{ig + 1}_k",
                f"G{ig + 1}_l",
                f"G{ig + 1}_Qx_invA",
                f"G{ig + 1}_Qy_invA",
                f"G{ig + 1}_Qz_invA",
                f"zero_phonon_intensity_G{label}",
            ]
        )
        for atom_label in atom_labels:
            header.append(f"dw_factor_G{label}_{atom_label}")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for itemp, temperature_k in enumerate(data.temperatures_k):
            row: list[float] = [float(temperature_k)]
            for ig in range(data.target_g_frac.shape[0]):
                row.extend(
                    [
                        *data.target_g_frac[ig],
                        *data.target_q_cart[ig],
                        data.zero_phonon_intensity[itemp, ig],
                    ]
                )
                row.extend(data.dw_factor[itemp, ig, :])
            writer.writerow(row)
    return path


def plot_dw_factor_vs_temperature(path: Path, data: TemperatureDependentUEDData) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    line_styles = ["-", "--", "-.", ":"]
    for ig, g_frac in enumerate(data.target_g_frac):
        for sym, indices in _species_group_indices(data.species):
            label = f"G = {_g_label(g_frac)}, {sym}"
            y = np.mean(data.dw_factor[:, ig, indices], axis=1)
            ax.plot(data.temperatures_k, y, line_styles[ig % len(line_styles)], label=label)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Debye-Waller factor")
    ax.set_title("Debye-Waller Factor vs Temperature")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_zero_phonon_intensity_vs_temperature(path: Path, data: TemperatureDependentUEDData) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ntargets = data.target_g_frac.shape[0]
    fig, axes = plt.subplots(1, ntargets, figsize=(4.2 * ntargets, 3.8), squeeze=False)
    for ig, ax in enumerate(axes[0]):
        ax.plot(data.temperatures_k, data.zero_phonon_intensity[:, ig], color="tab:blue")
        ax.set_title(f"G = {_g_label(data.target_g_frac[ig])}")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("I0")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Zero-Phonon Bragg Intensity vs Temperature")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_q_map(
    path: Path,
    data: TiledIntensityData,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    origin_exclude_radius_inv_a: float | None = None,
    marker_size: float = 1.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_mask = mask & np.isfinite(data.q_cart_all[:, 0]) & np.isfinite(data.q_cart_all[:, 1])
    rho = np.hypot(data.q_cart_all[:, 0], data.q_cart_all[:, 1])
    if origin_exclude_radius_inv_a is not None and origin_exclude_radius_inv_a > 0.0:
        plot_mask &= rho >= origin_exclude_radius_inv_a
    if not np.any(plot_mask):
        raise ValueError(
            "No tiled Q points left after masking; check Qz filter and "
            "origin_exclude_radius_inv_a relative to reciprocal geometry."
        )
    x = data.q_cart_all[plot_mask, 0]
    y = data.q_cart_all[plot_mask, 1]
    z = values[plot_mask]
    pos = z > EPS
    if not np.any(pos):
        raise ValueError(f"No finite positive intensities left to plot after masking ({path}).")
    x = x[pos]
    y = y[pos]
    z_pos = z[pos]
    vmax = float(np.percentile(z_pos, 99.9))
    vmin = float(np.maximum(EPS, np.min(z_pos)))
    if not np.isfinite(vmax) or not np.isfinite(vmin):
        raise ValueError(f"Non-finite percentile or intensity range for scatter colors ({path}).")
    if vmin >= vmax:
        vmax = vmin * np.sqrt(10.0)

    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    scatter = ax.scatter(x, y, c=z_pos, s=marker_size, linewidths=0.0, cmap="magma", norm=norm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$Q_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$Q_y$ ($\AA^{-1}$)")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label=colorbar_label, extend="max")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def write_qz0_outputs(
    input_h5: Path,
    output_dir: Path,
    *,
    temperature_k: float = 100.0,
    phwmin_mev: float = 0.2,
    gmax: int = 3,
    qz: float = 0.0,
    qz_tol: float = 1.0e-8,
    electron_scattering_model: str = "peng",
    eigenvector_gauge: str = "phonopy_to_phx",
    write_tiled_csv: bool = True,
    write_long: bool = True,
    temperature_sweep_k: Sequence[float] | None = None,
    temperature_target_g_frac: Sequence[Sequence[float]] = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
) -> dict[str, Path]:
    mesh = load_phonon_mesh_h5(input_h5, eigenvector_gauge=eigenvector_gauge)
    g_rad = shortest_inplane_primitive_g_modulus(mesh.bvec_rows_anginv)
    origin_mask_radius = 0.5 * g_rad
    data = compute_tiled_intensities(
        mesh,
        temperature_k=temperature_k,
        phwmin_mev=phwmin_mev,
        gmax=gmax,
        electron_scattering_model=electron_scattering_model,
    )
    mask = qz_mask(data, qz=qz, qz_tol=qz_tol)
    if not np.any(mask):
        raise ValueError(f"No tiled Q points found with Qz={qz:g} within tolerance {qz_tol:g}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if write_tiled_csv:
        outputs["wide_csv"] = write_wide_csv(output_dir / "tiled_intensities_qz0.csv", data, mask)
        if write_long:
            outputs["long_csv"] = write_long_csv(output_dir / "tiled_intensities_qz0_long.csv", data, mask)

    s1_total = np.sum(data.one_phonon_structure_factor, axis=1)
    i1_total = np.sum(data.one_phonon_intensity, axis=1)
    reciprocal_lattice_mask = integer_reciprocal_mask(data.q_frac_all)
    outputs["zero_phonon_png"] = plot_q_map(
        output_dir / "zero_phonon_intensity_qz0.png",
        data,
        data.zero_phonon_intensity,
        mask & reciprocal_lattice_mask,
        title="Zero-Phonon Intensity (Qz = 0)",
        colorbar_label="I0",
        origin_exclude_radius_inv_a=origin_mask_radius,
        marker_size=6.0,
    )
    outputs["one_phonon_structure_factor_png"] = plot_q_map(
        output_dir / "one_phonon_structure_factor_total_qz0.png",
        data,
        s1_total,
        mask,
        title="Total One-Phonon Structure Factor (Qz = 0)",
        colorbar_label="S1 total",
        origin_exclude_radius_inv_a=origin_mask_radius,
    )
    outputs["one_phonon_intensity_png"] = plot_q_map(
        output_dir / "one_phonon_intensity_total_qz0.png",
        data,
        i1_total,
        mask,
        title="Total One-Phonon Intensity (Qz = 0)",
        colorbar_label="I1 total",
        origin_exclude_radius_inv_a=origin_mask_radius,
    )
    outputs["combined_intensity_png"] = plot_q_map(
        output_dir / "combined_intensity_qz0.png",
        data,
        data.zero_phonon_intensity + i1_total,
        mask,
        title="Combined Intensity (Qz = 0)",
        colorbar_label="I0 + I1",
        origin_exclude_radius_inv_a=origin_mask_radius,
    )
    if temperature_sweep_k is not None:
        temp_data = compute_temperature_dependent_ued(
            mesh,
            temperature_sweep_k,
            target_g_frac=temperature_target_g_frac,
            phwmin_mev=phwmin_mev,
            electron_scattering_model=electron_scattering_model,
        )
        outputs["temperature_csv"] = write_temperature_csv(output_dir / "temperature_dependent_bragg.csv", temp_data)
        outputs["dw_factor_temperature_png"] = plot_dw_factor_vs_temperature(
            output_dir / "dw_factor_vs_temperature.png",
            temp_data,
        )
        outputs["zero_phonon_temperature_png"] = plot_zero_phonon_intensity_vs_temperature(
            output_dir / "zero_phonon_intensity_vs_temperature.png",
            temp_data,
        )
    return outputs


def compute_multi_phonon_intensity(*args, **kwargs):
    """Explicit multi-phonon UED intensity is not implemented in this workflow."""

    raise NotImplementedError("Explicit multi-phonon UED intensity is not implemented.")
