#!/usr/bin/env python
"""Equilibrium phonon linewidth along the Gamma-K-M-Gamma band path.

The canonical linewidth gamma (THz) is the phono3py imaginary self-energy / HWHM,
i.e. the W_scatter exact gamma = gamma_detail reduction (figures_band_path/data/).
The dn/dt linearized collision diagonal (gamma0_extract) is NOT used for
linewidths -- it diverges for Bose-enhanced soft-partner modes (see project
memory feedback-linewidth-definition).

Scientific choices
------------------
1. We interpolate gamma (smooth, additive, non-negative), never tau = 1/gamma.
2. Branch matching is done by JOINT (q, omega) interpolation, not by band index:
   every mesh mode is a sample point (qx, qy, omega) -> gamma, and we interpolate
   at the path mode's (qx, qy, omega). This is robust across band crossings and
   avoids the unreliable "mesh band index == path band index" assumption. The
   scheme is inverse-distance weighting in a scaled (q, omega) metric, which is
   *interpolating* (returns the exact mesh value when a path point coincides with
   a mesh point, e.g. at K and M) rather than smoothing.

Mesh gamma lives on the regular BZ grid (gamma_W_full, same q order as the
collision cache). Mesh and path frequencies both come from the same fc2 dynamical
matrix (phonopy), which is what makes the (q,omega) interpolation exact at K/M.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Band path (Gamma-K-M-Gamma) and small geometry helpers (self-contained).
# ---------------------------------------------------------------------------
_PATH_SEGMENTS = [[[0.0, 0.0, 0.0], [1 / 3, 1 / 3, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]]
_PATH_LABELS = ["$\\Gamma$", "K", "M", "$\\Gamma$"]


def _reclat_columns(cell: np.ndarray) -> np.ndarray:
    """Reciprocal lattice with b-vectors as columns (includes 2*pi)."""
    return 2.0 * math.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T


def _to_cart2d(q_frac: np.ndarray, reclat: np.ndarray) -> np.ndarray:
    """Fractional coords (N,3) -> Cartesian 2D (N,2), ignoring z."""
    cart = np.asarray(q_frac, dtype=float) @ reclat.T
    return cart[:, :2]


def _point_spacing(reclat: np.ndarray, mesh) -> float:
    b1 = reclat[:2, 0]
    b2 = reclat[:2, 1]
    return min(float(np.linalg.norm(b1)) / mesh[0],
               float(np.linalg.norm(b2)) / mesh[1])


def _build_phonopy(phono3py_yaml: str, fc2_path: str):
    """Return (Phonopy with symmetrized fc2, reclat) — shared by path & mesh evals."""
    import phono3py as _Phono3py
    from phonopy import Phonopy

    ph3 = _Phono3py.load(str(phono3py_yaml), log_level=0)
    phonon = Phonopy(
        ph3.unitcell,
        supercell_matrix=ph3.phonon_supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
    )
    phonon.force_constants = np.load(fc2_path)
    phonon.symmetrize_force_constants()
    reclat = _reclat_columns(ph3.primitive.cell)
    return phonon, reclat


def compute_band_structure(phono3py_yaml: str, fc2_path: str, npoints: int = 101):
    """Return (q_frac_path (Nq,3), freqs_THz (Nq,nb), x_vals (Nq,), hs_x, reclat)."""
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    phonon, reclat = _build_phonopy(phono3py_yaml, fc2_path)
    qpath_segs, connections = get_band_qpoints_and_path_connections(
        _PATH_SEGMENTS, npoints=npoints
    )
    phonon.run_band_structure(
        qpath_segs, path_connections=connections, labels=_PATH_LABELS,
        with_eigenvectors=False,
    )
    bs = phonon.band_structure
    q_frac = np.concatenate(qpath_segs, axis=0)
    q_frac[:, 2] = 0.0
    freqs = np.reshape(bs.frequencies, (-1, np.shape(bs.frequencies)[-1]))
    x_vals = np.reshape(bs.distances, (-1,)).astype(float)
    hs_x = [float(x) for x in bs._special_points]
    return q_frac, freqs, x_vals, hs_x, reclat


def mesh_frequencies(phono3py_yaml: str, fc2_path: str, qpoints_frac: np.ndarray):
    """fc2 phonon frequencies (THz, ascending per q) at the given mesh q-points.

    Computed from the SAME fc2 as the dispersion so path and mesh frequencies are
    consistent (this is what makes the (q,omega) interpolation exact at K/M)."""
    phonon, _ = _build_phonopy(phono3py_yaml, fc2_path)
    phonon.run_qpoints(np.asarray(qpoints_frac, dtype=float))
    return np.asarray(phonon.get_qpoints_dict()["frequencies"], dtype=np.float64)


# ---------------------------------------------------------------------------
# Core: joint (q, omega) inverse-distance interpolation of gamma.
# ---------------------------------------------------------------------------
def _tile_mesh_samples(q_mesh_frac, freq_mesh, gamma_mesh, reclat):
    """Expand the mesh to periodic images (+/-1 in each recip direction) and
    flatten (grid-image, band) into sample points.

    Returns
    -------
    q_cart : (M, 2)   Cartesian q of each (image) grid point, repeated per band
    omega  : (M,)     mode frequency
    gamma  : (M,)     mode linewidth
    """
    offsets = [np.array([dx, dy, 0.0]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    nb = freq_mesh.shape[1]
    qc, om, ga = [], [], []
    for off in offsets:
        cart = _to_cart2d(q_mesh_frac + off, reclat)          # (Ng, 2)
        qc.append(np.repeat(cart, nb, axis=0))                # (Ng*nb, 2)
        om.append(freq_mesh.ravel())                          # (Ng*nb,)
        ga.append(gamma_mesh.ravel())
    return np.vstack(qc), np.concatenate(om), np.concatenate(ga)


def estimate_sigma_omega(freq_mesh: np.ndarray, factor: float = 0.5) -> float:
    """Characteristic frequency scale = factor * median adjacent-branch gap.

    Keeps the (q,omega) kernel from mixing distinct branches while still letting a
    single branch disperse across a q-step.
    """
    f = np.sort(freq_mesh, axis=1)
    gaps = np.diff(f, axis=1)
    gaps = gaps[gaps > 1e-6]
    if gaps.size == 0:
        return max(1e-3, float(np.ptp(freq_mesh)) * 0.05)
    return float(factor * np.median(gaps))


def interpolate_gamma_qomega(
    q_path_cart: np.ndarray,      # (Nq, 2)
    omega_path: np.ndarray,       # (Nq, nb_path)
    sample_q: np.ndarray,         # (M, 2)
    sample_omega: np.ndarray,     # (M,)
    sample_gamma: np.ndarray,     # (M,)
    sigma_q: float,
    sigma_omega: float,
    k: int = 16,
    max_radius: float = 3.0,
    power: float = 2.0,
    exact_tol: float = 1e-9,
) -> np.ndarray:
    """Inverse-distance interpolation of gamma in scaled (q, omega) space.

    Distance^2 = |dq|^2/sigma_q^2 + (d omega)^2/sigma_omega^2.  Weighting 1/d^power
    makes this interpolating (exact at coincident sample points). Targets with no
    sample within `max_radius` (scaled units) get NaN.

    Returns gamma on the path, shape (Nq, nb_path).
    """
    from scipy.spatial import cKDTree

    X = np.column_stack([sample_q[:, 0] / sigma_q,
                         sample_q[:, 1] / sigma_q,
                         sample_omega / sigma_omega])
    tree = cKDTree(X)

    nq, nbp = omega_path.shape
    targets = np.column_stack([
        np.repeat(q_path_cart[:, 0] / sigma_q, nbp),
        np.repeat(q_path_cart[:, 1] / sigma_q, nbp),
        (omega_path / sigma_omega).ravel(),
    ])
    kq = min(k, X.shape[0])
    dist, idx = tree.query(targets, k=kq)
    if kq == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    out = np.full(targets.shape[0], np.nan, dtype=float)
    g = sample_gamma
    for i in range(targets.shape[0]):
        d = dist[i]
        within = d <= max_radius
        if not np.any(within):
            continue
        d = d[within]
        ids = idx[i][within]
        exact = d <= exact_tol
        if np.any(exact):
            out[i] = float(np.mean(g[ids[exact]]))
            continue
        w = 1.0 / np.power(d, power)
        out[i] = float(np.sum(w * g[ids]) / np.sum(w))
    return out.reshape(nq, nbp)


def gamma_on_path(
    q_path_frac, omega_path, reclat,
    q_mesh_frac, freq_mesh, gamma_mesh,
    mesh, sigma_q=None, sigma_omega=None, **kw,
):
    """Convenience wrapper: tile mesh, set default scales, interpolate."""
    q_path_cart = _to_cart2d(q_path_frac, reclat)
    sq, so, sg = _tile_mesh_samples(q_mesh_frac, freq_mesh, gamma_mesh, reclat)
    if sigma_q is None:
        sigma_q = _point_spacing(reclat, mesh)
    if sigma_omega is None:
        sigma_omega = estimate_sigma_omega(freq_mesh)
    gamma_path = interpolate_gamma_qomega(
        q_path_cart, omega_path, sq, so, sg, sigma_q, sigma_omega, **kw
    )
    return gamma_path, dict(sigma_q=sigma_q, sigma_omega=sigma_omega)


# ---------------------------------------------------------------------------
# Per-system configuration and data loading.
# ---------------------------------------------------------------------------
import os  # noqa: E402

PV4 = Path(os.environ.get(
    "PV4",
    "/pscratch/sd/j/jdgeorga/twist-anything/phonon_unfolding/scratch/phonon_diff/ued_paper/paper_v4",
))
_FBP = PV4 / "dynamics" / "figures_band_path"

SYSTEMS = {
    "mose2_monolayer": dict(
        label="MoSe$_2$ monolayer",
        root=PV4 / "1-MoSe2",
        mesh=(36, 36, 1), mtag="m36x36x1",
        yaml=PV4 / "1-MoSe2/production/MoSe2_relaxed_phono3py_displacements.yaml",
        fc2=PV4 / "1-MoSe2/production/phonon_cache_m36x36x1/fc2.npy",
        gamma_w=_FBP / "data/mose2_monolayer/gamma_W_full_T50.npz",
    ),
    "mose2_wse2_aligned": dict(
        label="MoSe$_2$/WSe$_2$ aligned bilayer",
        root=PV4 / "2-MoSe2-WSe2",
        mesh=(36, 36, 1), mtag="m36x36x1",
        yaml=PV4 / "2-MoSe2-WSe2/production/phono3py_disp.yaml",
        fc2=PV4 / "2-MoSe2-WSe2/production/phonon_cache_m36x36x1/fc2.npy",
        gamma_w=_FBP / "data/mose2_wse2_aligned/gamma_W_full_T50.npz",
    ),
    "mose2_wse2_22deg": dict(
        label="MoSe$_2$/WSe$_2$ 22$\\degree$ twisted",
        root=PV4 / "3-MoSe2-WSe2_42_22deg",
        mesh=(12, 12, 1), mtag="m12x12x1",
        yaml=PV4 / "3-MoSe2-WSe2_42_22deg/production/MoSe2_WSe2_42.22deg_relaxed_phono3py_displacements.yaml",
        fc2=PV4 / "3-MoSe2-WSe2_42_22deg/production/phonon_cache_m12x12x1/fc2.npy",
        gamma_w=_FBP / "data/mose2_wse2_22deg/gamma_W_full_T50.npz",
    ),
}


def load_wscatter_gamma(cfg):
    """Return (qpoints_frac, gamma_mesh) for the W_scatter route (THz)."""
    z = np.load(cfg["gamma_w"])
    return (np.asarray(z["qpoints_frac"], dtype=np.float64),
            np.asarray(z["gamma_W_full"], dtype=np.float64))


def load_mesh(cfg):
    """Return (qpoints_frac, freq_mesh, gamma_mesh) for the canonical linewidth.

    gamma = W_scatter exact gamma (gamma_detail HWHM / imaginary self-energy).
    freq  = fc2 phonon frequencies at the same q-points (NOT the dn/dt npz, which
            must never be used for linewidths -- see project memory).
    """
    qpts, gamma = load_wscatter_gamma(cfg)
    freq = mesh_frequencies(str(cfg["yaml"]), str(cfg["fc2"]), qpts)
    if freq.shape != gamma.shape:
        raise ValueError(f"freq {freq.shape} vs gamma {gamma.shape} mismatch")
    return qpts, freq, gamma


# ---------------------------------------------------------------------------
# Linewidth <-> lifetime relationship (phono3py convention).
# ---------------------------------------------------------------------------
def lifetime_from_gamma(gamma_thz):
    """Phonon lifetime tau (ps) from linewidth gamma (THz).

    phono3py convention:  tau = 1 / (2 * 2*pi * gamma) = 1 / (4*pi*gamma).
      * the factor 2   is FWHM = 2*gamma (gamma is the HWHM imaginary self-energy);
      * the factor 2*pi converts ordinary-frequency THz to angular frequency, so
        1/THz -> ps directly.
    Verified against the installed phono3py source:
      cui/kaccum_script.py:138   tau = 1/(2*2*np.pi*g)
      scripts/phono3py_kdeplot.py:17  tau = 1/g/(2*2*np.pi)
      api_isotope.py:133  "scattering rate in THz (1/4pi-tau)"  -> gamma = 1/(4*pi*tau).
    Returns +inf where gamma <= 0 (a mode with no decay channel).
    """
    g = np.asarray(gamma_thz, dtype=float)
    tau = np.full(g.shape, np.inf, dtype=float)
    pos = np.isfinite(g) & (g > 0.0)
    tau[pos] = 1.0 / (4.0 * np.pi * g[pos])
    return tau


# ---------------------------------------------------------------------------
# Exact on-path mesh modes (NO interpolation): mesh grid points (incl. periodic
# images) whose Cartesian q lies exactly on a Gamma-K-M-Gamma segment.
# ---------------------------------------------------------------------------
def exact_path_points(qpts_frac, freq_mesh, gamma_mesh, reclat, hs_x,
                      tol=1e-6):
    """Return (x, freq, gamma) flat arrays for mesh modes lying ON the path.

    Only grid points (and their +/-1 periodic images) whose perpendicular
    distance to a path segment is < tol [A^-1] are kept -- i.e. points that fall
    directly on Gamma-K-M-Gamma, used with their raw mesh frequency and gamma (no
    interpolation). Each kept grid point contributes all nb bands.
    """
    seg_frac = [(_PATH_SEGMENTS[0][i], _PATH_SEGMENTS[0][i + 1]) for i in range(3)]
    x_ranges = [(hs_x[i], hs_x[i + 1]) for i in range(3)]
    nb = freq_mesh.shape[1]
    offsets = [np.array([dx, dy, 0.0]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    xs, gi = [], []  # path coordinate, originating grid index
    for (p0f, p1f), (x0, x1) in zip(seg_frac, x_ranges):
        p0 = _to_cart2d(np.array([p0f]), reclat)[0]
        p1 = _to_cart2d(np.array([p1f]), reclat)[0]
        v = p1 - p0
        vlen2 = float(v @ v)
        if vlen2 < 1e-24:
            continue
        seen = set()
        for off in offsets:
            P = _to_cart2d(qpts_frac + off, reclat)            # (Ng, 2)
            t = ((P - p0) @ v) / vlen2
            proj = p0 + t[:, None] * v
            perp = np.linalg.norm(P - proj, axis=1)
            on = (perp < tol) & (t > -1e-9) & (t < 1.0 + 1e-9)
            for g in np.flatnonzero(on):
                g = int(g)
                if g in seen:           # one image per grid point per segment
                    continue
                seen.add(g)
                xs.append(x0 + float(np.clip(t[g], 0.0, 1.0)) * (x1 - x0))
                gi.append(g)

    if not gi:
        return (np.array([]), np.array([]), np.array([]))
    gi = np.asarray(gi, dtype=int)
    xs = np.asarray(xs, dtype=float)
    X = np.repeat(xs, nb)
    F = freq_mesh[gi].ravel()
    G = gamma_mesh[gi].ravel()
    return X, F, G


# ---------------------------------------------------------------------------
# Shared renderer: two-panel dispersion (wave vector x frequency), top colored
# by linewidth gamma, bottom by lifetime tau. LINEAR color scales.
# ---------------------------------------------------------------------------
def _render_two_panel(name, cfg, x_skel, freqs_skel, hs_x,
                      Xp, Yp, Gp, suffix, title_extra="",
                      gamma_cmap="inferno", tau_cmap="viridis",
                      min_freq_thz=0.1, marker_s=9, dpi=300, out_dirs=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    nb = freqs_skel.shape[1]
    # mask the soft acoustic region near Gamma (gamma -> 0, tau -> inf)
    Gp = np.where(Yp >= min_freq_thz, Gp, np.nan)
    Taup = lifetime_from_gamma(Gp)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, figsize=(6.4, 7.4), dpi=150,
        gridspec_kw=dict(hspace=0.10),
    )
    for ax in (ax0, ax1):
        for b in range(nb):
            ax.plot(x_skel, freqs_skel[:, b], color="0.85", lw=0.5, zorder=0)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel("Frequency (THz)", fontsize=12)

    # top: linewidth gamma, linear 0..p98
    gm = np.isfinite(Gp) & np.isfinite(Yp)
    vg = float(np.nanpercentile(Gp[gm], 98)) if gm.any() else 1.0
    gnorm = Normalize(vmin=0.0, vmax=max(vg, 1e-9))
    og = np.argsort(Gp[gm])
    s0 = ax0.scatter(Xp[gm][og], Yp[gm][og], c=Gp[gm][og], cmap=gamma_cmap,
                     norm=gnorm, s=marker_s, linewidths=0, zorder=2)
    cb0 = fig.colorbar(s0, ax=ax0, aspect=28, pad=0.02)
    cb0.set_label(r"Linewidth $\gamma$ (THz)", fontsize=11)
    ax0.set_title(f"{cfg['label']} — equilibrium ph-ph linewidth & lifetime "
                  f"(T = 50 K){title_extra}", fontsize=11)

    # bottom: lifetime tau, linear 0..p98
    tm = np.isfinite(Taup) & (Taup > 0) & np.isfinite(Yp)
    vt = float(np.nanpercentile(Taup[tm], 98)) if tm.any() else 1.0
    tnorm = Normalize(vmin=0.0, vmax=max(vt, 1e-9))
    ot = np.argsort(Taup[tm])
    s1 = ax1.scatter(Xp[tm][ot], Yp[tm][ot], c=Taup[tm][ot], cmap=tau_cmap,
                     norm=tnorm, s=marker_s, linewidths=0, zorder=2)
    cb1 = fig.colorbar(s1, ax=ax1, aspect=28, pad=0.02)
    cb1.set_label(r"Lifetime $\tau = 1/(4\pi\gamma)$ (ps)", fontsize=11)
    ax1.set_xlabel("Wave vector", fontsize=12)

    for ax in (ax0, ax1):
        for xv in hs_x[1:-1]:
            ax.axvline(xv, color="k", lw=0.6, alpha=0.4)
        ax.set_xlim(float(x_skel[0]), float(x_skel[-1]))
    ax1.set_xticks(list(hs_x))
    ax1.set_xticklabels(_PATH_LABELS, fontsize=12)

    print(f"  [{name}/{suffix}] points={int(gm.sum())}  "
          f"gamma_max={vg:.4e} THz  tau_p98={vt:.4g} ps", flush=True)

    if out_dirs is None:
        out_dirs = [cfg["root"] / "production" / "figures", _FBP / "figures"]
    saved = []
    for od in out_dirs:
        od = Path(od)
        od.mkdir(parents=True, exist_ok=True)
        stem = od / f"{name}_{suffix}"
        fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=dpi)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        saved.append(stem)
        print(f"  [{name}] saved {stem}.png / .pdf", flush=True)
    plt.close(fig)
    return saved


def plot_system(name, npoints=101, min_freq_thz=0.1,
                gamma_cmap="inferno", tau_cmap="viridis",
                dpi=300, out_dirs=None):
    """Interpolated two-panel (gamma / tau) dispersion figure, linear scales."""
    cfg = SYSTEMS[name]
    qpts, freq_mesh, gamma_mesh = load_mesh(cfg)
    q_path, freqs_p, x, hs_x, reclat = compute_band_structure(
        str(cfg["yaml"]), str(cfg["fc2"]), npoints=npoints
    )
    gamma_path, scales = gamma_on_path(
        q_path, freqs_p, reclat, qpts, freq_mesh, gamma_mesh, cfg["mesh"]
    )
    nb = freqs_p.shape[1]
    Xp = np.repeat(x[:, None], nb, axis=1).ravel()
    return _render_two_panel(
        name, cfg, x, freqs_p, hs_x, Xp, freqs_p.ravel(), gamma_path.ravel(),
        suffix="linewidth_lifetime_band_path",
        gamma_cmap=gamma_cmap, tau_cmap=tau_cmap,
        min_freq_thz=min_freq_thz, marker_s=9, dpi=dpi, out_dirs=out_dirs,
    )


def plot_system_exact(name, npoints=101, min_freq_thz=0.1,
                      gamma_cmap="inferno", tau_cmap="viridis",
                      dpi=300, out_dirs=None):
    """Non-interpolated figure: only mesh modes lying exactly on the path."""
    cfg = SYSTEMS[name]
    qpts, freq_mesh, gamma_mesh = load_mesh(cfg)
    _, freqs_p, x, hs_x, reclat = compute_band_structure(
        str(cfg["yaml"]), str(cfg["fc2"]), npoints=npoints
    )
    Xp, Fp, Gp = exact_path_points(qpts, freq_mesh, gamma_mesh, reclat, hs_x)
    n_on = len(np.unique(np.round(Xp, 9))) if Xp.size else 0
    print(f"  [{name}] {n_on} on-path mesh q-points (exact, no interpolation)",
          flush=True)
    return _render_two_panel(
        name, cfg, x, freqs_p, hs_x, Xp, Fp, Gp,
        suffix="linewidth_lifetime_band_path_meshpoints",
        title_extra=" — on-path mesh points",
        gamma_cmap=gamma_cmap, tau_cmap=tau_cmap,
        min_freq_thz=min_freq_thz, marker_s=34, dpi=dpi, out_dirs=out_dirs,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", choices=sorted(SYSTEMS),
                    default=sorted(SYSTEMS))
    ap.add_argument("--npoints", type=int, default=101)
    ap.add_argument("--min-freq-thz", type=float, default=0.1)
    ap.add_argument("--gamma-cmap", default="inferno")
    ap.add_argument("--tau-cmap", default="viridis")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--mode", choices=["interp", "exact", "both"], default="both",
                    help="interp: (q,omega) interpolation; exact: on-path mesh "
                         "points only; both (default).")
    args = ap.parse_args()
    for name in args.systems:
        print(f"Processing {name} ...", flush=True)
        kw = dict(npoints=args.npoints, min_freq_thz=args.min_freq_thz,
                  gamma_cmap=args.gamma_cmap, tau_cmap=args.tau_cmap, dpi=args.dpi)
        if args.mode in ("interp", "both"):
            plot_system(name, **kw)
        if args.mode in ("exact", "both"):
            plot_system_exact(name, **kw)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
