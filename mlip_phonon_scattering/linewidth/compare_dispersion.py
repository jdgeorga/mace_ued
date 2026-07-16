#!/usr/bin/env python
"""Comparison figures: phonon dispersion colored by linewidth gamma / lifetime tau.

Given several *variants* (e.g. DFPT-harmonic vs MLIP-harmonic, loto on/off, with/
without reference-force subtraction) this renders two publication-quality figures:

* ``dispersion_linewidth.png/pdf`` -- omega(q) along Gamma-K-M-Gamma, one panel per
  variant SIDE BY SIDE, points colored by the phonon linewidth gamma (THz) with a
  single SHARED colorbar/normalization across all panels.
* ``dispersion_lifetime.png/pdf`` -- same layout, colored by the lifetime tau (ps).

and a ``comparison_summary.csv`` with the top-optical gamma/tau at K and M for each
variant plus every pairwise ratio (labelled by variant, e.g. ``dfpt_harm / mlip_strict``).

Scientific conventions (verbatim project rules)
-----------------------------------------------
* **Interpolate gamma, NEVER tau.** gamma is smooth / additive / non-negative; we map
  the mesh gamma onto the band path with :func:`linewidth_path.gamma_on_path` (joint
  ``(q, omega)`` inverse-distance interpolation) and then DERIVE tau from the
  interpolated gamma with :func:`linewidth_path.lifetime_from_gamma`
  (``tau = 1 / (4*pi*gamma)``; ``+inf`` at ``gamma == 0``).
* **fc2 units / frequency factor (correction C6).** The band structure / mesh
  frequencies come from diagonalizing each variant's fc2 (via the reused
  ``compute_band_structure`` / ``mesh_frequencies`` helpers, which use phonopy's
  default ``VaspToTHz`` factor appropriate for an **eV/Ang^2** MLIP fc2). If a variant's
  fc2 is **QE-derived (Ry/bohr^2)** the correct conversion is
  ``frequency_factor_to_THz = 108.97077184367376`` -- since phonopy frequencies scale
  linearly with the conversion factor (``omega = factor * sqrt(lambda)``), we obtain the
  QE-correct omega by rescaling the default-factor frequencies by
  ``108.97077184367376 / VaspToTHz``. gamma itself (produced by the scattering pipeline
  in true THz) is never rescaled; both the path and mesh frequencies are rescaled by the
  SAME factor so the ``(q, omega)`` interpolation stays exact at K/M. Set a variant's
  units via ``fc2_units='ev'`` (default) or ``fc2_units='ry'`` (or an explicit
  ``factor=...``).

This module reuses, and does not duplicate, the linewidth-on-path machinery in
:mod:`mlip_phonon_scattering.linewidth.linewidth_path`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mlip_phonon_scattering.linewidth.linewidth_path import (
    _PATH_LABELS,
    compute_band_structure,
    gamma_on_path,
    lifetime_from_gamma,
    mesh_frequencies,
)

# phonopy's default frequency conversion factor (eV/Ang^2, amu -> THz).  Hard-coded to
# stay version-independent across phonopy releases (== phonopy.units.VaspToTHz).
_VASP_TO_THZ = 15.633302300230191
# QE frequency conversion factor for a Ry/bohr^2 fc2 (correction C6).
_QE_FACTOR_TO_THZ = 108.97077184367376

# figure filenames (without extension); each is written as both .png and .pdf.
_FIG_GAMMA = "dispersion_linewidth"
_FIG_TAU = "dispersion_lifetime"
_SUMMARY_CSV = "comparison_summary.csv"

_GAMMA_LABEL = r"Linewidth $\gamma$ (THz)"
_TAU_LABEL = r"Lifetime $\tau = 1/(4\pi\gamma)$ (ps)"


# ---------------------------------------------------------------------------
# Per-variant prepared dispersion (the pure, plot-ready representation).
# ---------------------------------------------------------------------------
@dataclass
class VariantDispersion:
    """Everything the renderer / CSV writer needs for one variant.

    Attributes
    ----------
    label : str
        Human-readable variant name (panel title, CSV row label).
    x : np.ndarray, shape (Nq,)
        Path coordinate (the skeleton x used for the gray dispersion lines).
    freqs : np.ndarray, shape (Nq, nb)
        Band-path frequencies (THz), already in the correct THz factor for this variant.
    hs_x : Sequence[float]
        High-symmetry x positions ``[Gamma, K, M, Gamma]``.
    gamma_path : np.ndarray, shape (Nq, nb)
        INTERPOLATED linewidth gamma (THz) on the band path (never tau).
    """

    label: str
    x: np.ndarray
    freqs: np.ndarray
    hs_x: Sequence[float]
    gamma_path: np.ndarray


# ---------------------------------------------------------------------------
# Variant normalization + data loading.
# ---------------------------------------------------------------------------
def _factor_for(spec: dict) -> float:
    """Frequency->THz conversion factor for a variant (correction C6)."""
    if spec.get("factor"):
        return float(spec["factor"])
    units = str(spec.get("fc2_units", "ev")).strip().lower()
    if units in ("ry", "qe", "ryd", "ry/bohr2", "ry/bohr^2"):
        return _QE_FACTOR_TO_THZ
    if units in ("ev", "", "mlip", "ev/ang2", "ev/ang^2"):
        return _VASP_TO_THZ
    raise ValueError(f"unknown fc2_units {units!r} (use 'ev' or 'ry')")


def _normalize_variant(variant) -> dict:
    """Coerce a dict or ``(label, yaml, fc2, gamma[, units[, modes]])`` tuple."""
    if isinstance(variant, (tuple, list)):
        if len(variant) < 4:
            raise ValueError(
                "variant tuple must be (label, phono3py_yaml, fc2_path, gamma_npz[, units])"
            )
        spec = dict(
            label=variant[0],
            phono3py_yaml=variant[1],
            fc2_path=variant[2],
            gamma_npz=variant[3],
        )
        if len(variant) > 4 and variant[4]:
            spec["fc2_units"] = variant[4]
        if len(variant) > 5 and variant[5]:
            spec["injected_modes"] = variant[5]
    elif isinstance(variant, dict):
        spec = dict(variant)
    else:
        raise TypeError(f"variant must be a dict or tuple, got {type(variant)!r}")

    label = spec.get("label")
    yaml = spec.get("phono3py_yaml") or spec.get("yaml")
    fc2 = spec.get("fc2_path") or spec.get("fc2")
    gamma = spec.get("gamma_npz") or spec.get("gamma")
    missing = [n for n, v in (("label", label), ("phono3py_yaml", yaml),
                              ("fc2_path", fc2), ("gamma_npz", gamma)) if not v]
    if missing:
        raise ValueError(f"variant {label or spec!r} missing required field(s): {missing}")
    return dict(
        label=str(label),
        phono3py_yaml=str(yaml),
        fc2_path=str(fc2),
        gamma_npz=str(gamma),
        factor=_factor_for(spec),
        injected_modes=(
            str(spec["injected_modes"] or spec.get("injected_modes_npz"))
            if spec.get("injected_modes") or spec.get("injected_modes_npz")
            else None
        ),
    )


def _load_mesh_gamma(gamma_npz: str):
    """Return ``(qpoints_frac (Nmesh,3), gamma_mesh (Nmesh,nb))`` from a gamma npz.

    Accepts the canonical pipeline keys (``qpoints_frac``/``gamma_W_full``) with a few
    tolerant fall-backs.
    """
    z = np.load(gamma_npz)

    def _first(keys):
        for k in keys:
            if k in z:
                return np.asarray(z[k])
        raise KeyError(f"{gamma_npz}: none of {keys} present (has {list(z.keys())})")

    q = _first(["qpoints_frac", "q_frac", "qpoints", "q"]).astype(np.float64)
    g = _first(["gamma_W_full", "gamma", "gamma_mesh", "gamma_thz"]).astype(np.float64)
    return q, g


def _row_permutation(
    source: np.ndarray, target: np.ndarray, *, label: str, require_all: bool = True
) -> np.ndarray:
    """Return source-row indices that put rows into target order.

    BZ grids may carry equivalent boundary points more than once after folding;
    pair repeated rows stably by occurrence order rather than treating them as
    an ambiguity.
    """
    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1:] != target.shape[1:]:
        raise ValueError(f"{label}: incompatible row shapes {source.shape} and {target.shape}")
    if require_all and len(source) != len(target):
        raise ValueError(f"{label}: source/target row counts differ")
    if len(target) > len(source):
        raise ValueError(f"{label}: target has more rows than source")
    lookup: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(source):
        key = tuple(np.asarray(row, dtype=np.int64))
        lookup.setdefault(key, []).append(index)
    permutation = []
    for row in target:
        key = tuple(np.asarray(row, dtype=np.int64))
        candidates = lookup.get(key)
        if not candidates:
            raise ValueError(f"{label}: target row {key} is absent from source")
        permutation.append(candidates.pop(0))
    if require_all and any(lookup.values()):
        raise ValueError(f"{label}: source has unmatched rows")
    return np.asarray(permutation, dtype=np.intp)


def _periodic_q_keys(qpoints: np.ndarray, *, scale: int = 10**8) -> np.ndarray:
    """Canonical integer keys for fractional q points, modulo reciprocal lattice vectors."""
    wrapped = np.mod(np.asarray(qpoints, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=0.5 / scale)] = 0.0
    return np.rint(wrapped * scale).astype(np.int64) % scale


def _injected_mesh_frequencies(
    phono3py_yaml: str, injected_modes: str, q_mesh: np.ndarray, mesh
) -> np.ndarray:
    """Load matdyn-injected frequencies and reorder them onto gamma's mesh order.

    ``matdyn_modes.py`` writes the Phono3py BZ-grid addresses alongside its QE
    frequencies.  Validate those addresses against the requested YAML/mesh first,
    then use the corresponding wrapped q points to align with the gamma artifact.
    """
    import phono3py

    with np.load(injected_modes) as z:
        for key in ("frequencies", "grid_address", "mesh"):
            if key not in z:
                raise KeyError(f"{injected_modes}: required key {key!r} is absent")
        frequencies = np.asarray(z["frequencies"], dtype=np.float64)
        injected_address = np.asarray(z["grid_address"], dtype=np.int64)
        injected_mesh = np.asarray(z["mesh"], dtype=np.int64)
    expected_mesh = np.asarray(mesh, dtype=np.int64)
    if not np.array_equal(injected_mesh, expected_mesh):
        raise ValueError(
            f"{injected_modes}: mesh {injected_mesh.tolist()} does not match "
            f"requested mesh {expected_mesh.tolist()}"
        )
    if frequencies.ndim != 2 or injected_address.shape != (frequencies.shape[0], 3):
        raise ValueError(
            f"{injected_modes}: frequencies {frequencies.shape} and grid_address "
            f"{injected_address.shape} are incompatible"
        )

    ph3 = phono3py.load(phono3py_yaml, log_level=0)
    ph3.mesh_numbers = expected_mesh.tolist()
    expected_address = np.asarray(ph3.grid.addresses, dtype=np.int64)
    address_order = _row_permutation(
        injected_address, expected_address, label=f"{injected_modes}: grid_address"
    )
    frequencies_on_grid = frequencies[address_order]

    grid_q = np.dot(expected_address, ph3.grid.QDinv)
    q_order = _row_permutation(
        _periodic_q_keys(grid_q),
        _periodic_q_keys(q_mesh),
        label=f"{injected_modes}: q mesh",
        require_all=False,
    )
    return frequencies_on_grid[q_order]


def _prepare_variant(variant, mesh, npoints: int) -> VariantDispersion:
    """Load one variant and build its plot-ready :class:`VariantDispersion`.

    Reuses ``compute_band_structure`` (band path), ``mesh_frequencies`` (mesh omega
    from the SAME fc2) and ``gamma_on_path`` (interpolate gamma).  A variant with
    ``injected_modes`` instead uses its matdyn/QE mesh frequencies, already in THz,
    for gamma's branch interpolation; the plotted path skeleton remains the raw-fc2
    diagonalization until a matching matdyn path artifact is supplied.
    """
    spec = _normalize_variant(variant)
    q_path, freqs_p, x, hs_x, reclat = compute_band_structure(
        spec["phono3py_yaml"], spec["fc2_path"], npoints=npoints
    )
    q_mesh, gamma_mesh = _load_mesh_gamma(spec["gamma_npz"])
    injected = spec["injected_modes"]
    if injected is None:
        freq_mesh = mesh_frequencies(spec["phono3py_yaml"], spec["fc2_path"], q_mesh)
    else:
        freq_mesh = _injected_mesh_frequencies(
            spec["phono3py_yaml"], injected, q_mesh, mesh
        )

    if freq_mesh.shape != gamma_mesh.shape:
        raise ValueError(
            f"variant {spec['label']!r}: mesh freq {freq_mesh.shape} vs gamma "
            f"{gamma_mesh.shape} mismatch (fc2 / gamma-npz band count disagree)"
        )

    scale = spec["factor"] / _VASP_TO_THZ
    if not np.isclose(scale, 1.0):
        freqs_p = freqs_p * scale
        if injected is None:
            freq_mesh = freq_mesh * scale

    gamma_path, _ = gamma_on_path(
        q_path, freqs_p, reclat, q_mesh, freq_mesh, gamma_mesh, mesh
    )
    return VariantDispersion(
        label=spec["label"], x=np.asarray(x, float), freqs=np.asarray(freqs_p, float),
        hs_x=[float(v) for v in hs_x], gamma_path=np.asarray(gamma_path, float),
    )


# ---------------------------------------------------------------------------
# Summary (top-optical gamma/tau at K and M + pairwise ratios).
# ---------------------------------------------------------------------------
def _top_optical_at(x_target, x, freqs, gamma_path):
    """Return ``(omega, gamma, tau)`` for the highest-frequency branch nearest x_target."""
    idx = int(np.argmin(np.abs(np.asarray(x, float) - float(x_target))))
    branch = int(np.argmax(freqs[idx]))
    omega = float(freqs[idx, branch])
    gamma = float(gamma_path[idx, branch])
    tau = float(lifetime_from_gamma(np.array([gamma]))[0])   # derive tau from gamma
    return omega, gamma, tau


def _ratio(num, den):
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0.0:
        return float("nan")
    return float(num) / float(den)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and not np.isfinite(v):
        return "inf" if v > 0 else ("-inf" if v < 0 else "nan")
    return f"{v:.6g}"


def _write_summary(vdisps: Sequence[VariantDispersion], out_dir) -> Path:
    """Write ``comparison_summary.csv`` (per-variant K/M + all pairwise ratios).

    Columns: ``kind,label,point,omega_THz,gamma_THz,tau_ps``.  For ``kind=variant`` the
    values are the top-optical branch at that high-symmetry point; for ``kind=ratio`` the
    gamma/tau columns hold the (dimensionless) ratio of the two named variants and
    omega is blank.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _SUMMARY_CSV

    points = {"K": 1, "M": 2}   # hs_x index of each high-symmetry point
    per_variant = {}            # label -> {point -> (omega, gamma, tau)}
    rows = []
    for v in vdisps:
        per_variant[v.label] = {}
        for pt, hi in points.items():
            omega, gamma, tau = _top_optical_at(v.hs_x[hi], v.x, v.freqs, v.gamma_path)
            per_variant[v.label][pt] = (omega, gamma, tau)
            rows.append(["variant", v.label, pt, _fmt(omega), _fmt(gamma), _fmt(tau)])

    labels = [v.label for v in vdisps]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            for pt in points:
                _, ga, ta = per_variant[a][pt]
                _, gb, tb = per_variant[b][pt]
                rows.append(["ratio", f"{a} / {b}", pt, "",
                             _fmt(_ratio(ga, gb)), _fmt(_ratio(ta, tb))])

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "label", "point", "omega_THz", "gamma_THz", "tau_ps"])
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Renderer: side-by-side per-variant panels with ONE shared colorbar/norm.
# ---------------------------------------------------------------------------
def _shared_vmax(colored_values, percentile=98.0) -> float:
    """Robust shared upper color limit (percentile of the pooled finite values)."""
    pooled = np.concatenate(colored_values) if colored_values else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return 1.0
    vmax = float(np.percentile(pooled, percentile))
    return vmax if (np.isfinite(vmax) and vmax > 0.0) else 1.0


def _render(
    vdisps: Sequence[VariantDispersion],
    out_dir,
    *,
    quantity: str,             # "gamma" or "tau"
    filename: str,
    cmap: str,
    cbar_label: str,
    min_freq_thz: float = 0.1,
    dpi: int = 300,
    temperature: Optional[float] = None,
    marker_s: float = 10.0,
) -> Path:
    """Render one comparison figure (all variants, shared colorbar). Saves .png + .pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if quantity not in ("gamma", "tau"):
        raise ValueError("quantity must be 'gamma' or 'tau'")

    # Prepare per-panel colored values.  Mask the soft near-Gamma acoustic region
    # (gamma -> 0 => tau -> inf) so it never dominates the shared color scale.
    panels = []
    colored = []
    for v in vdisps:
        nb = v.freqs.shape[1]
        Xp = np.repeat(np.asarray(v.x, float)[:, None], nb, axis=1).ravel()
        Yp = np.asarray(v.freqs, float).ravel()
        gamma_flat = np.where(Yp >= min_freq_thz, v.gamma_path.ravel(), np.nan)
        if quantity == "gamma":
            cvals = gamma_flat
            mask = np.isfinite(cvals) & np.isfinite(Yp) & (cvals >= 0.0)
        else:  # tau, DERIVED from the (interpolated) gamma -- never interpolated directly
            cvals = lifetime_from_gamma(gamma_flat)
            mask = np.isfinite(cvals) & np.isfinite(Yp) & (cvals > 0.0)
        panels.append((v, Xp, Yp, cvals, mask))
        colored.append(cvals[mask])

    vmax = _shared_vmax(colored)
    norm = Normalize(vmin=0.0, vmax=vmax)

    n = len(vdisps)
    fig_w = max(3.3 * n + 1.6, 4.2)
    fig, axes = plt.subplots(
        1, n, sharey=True, figsize=(fig_w, 4.7), dpi=150,
        squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    for ax, (v, Xp, Yp, cvals, mask) in zip(axes, panels):
        nb = v.freqs.shape[1]
        for b in range(nb):
            ax.plot(v.x, v.freqs[:, b], color="0.85", lw=0.5, zorder=0)
        order = np.argsort(cvals[mask])          # draw large values on top
        ax.scatter(
            Xp[mask][order], Yp[mask][order], c=cvals[mask][order],
            cmap=cmap, norm=norm, s=marker_s, linewidths=0, zorder=2,
        )
        ax.set_title(v.label, fontsize=11)
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(float(v.x[0]), float(v.x[-1]))
        for xv in list(v.hs_x)[1:-1]:
            ax.axvline(xv, color="k", lw=0.6, alpha=0.4)
        ax.set_xticks(list(v.hs_x))
        ax.set_xticklabels(_PATH_LABELS, fontsize=11)
        ax.set_xlabel("Wave vector", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)

    axes[0].set_ylabel("Frequency (THz)", fontsize=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), aspect=32, pad=0.02, fraction=0.06)
    cbar.set_label(cbar_label, fontsize=12)

    quantity_name = "linewidth $\\gamma$" if quantity == "gamma" else "lifetime $\\tau$"
    title = f"Phonon dispersion colored by {quantity_name}"
    if temperature is not None:
        title += f"  (T = {temperature:g} K)"
    fig.suptitle(title, fontsize=13)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / filename
    fig.savefig(f"{stem}.png", dpi=dpi)          # constrained_layout manages spacing
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"  [{filename}] {n} variant(s); shared vmax={vmax:.4g}; "
          f"saved {stem}.png / .pdf", flush=True)
    return Path(f"{stem}.png")


# ---------------------------------------------------------------------------
# Core: both figures + the CSV from prepared VariantDispersion objects.
# ---------------------------------------------------------------------------
def _compare_core(
    vdisps: Sequence[VariantDispersion],
    out_dir,
    *,
    temperature: float = 50.0,
    gamma_cmap: str = "inferno",
    tau_cmap: str = "viridis",
    min_freq_thz: float = 0.1,
    dpi: int = 300,
) -> dict:
    """Render both figures + write the CSV from already-prepared dispersions.

    This is the pure, phono3py-free heart of :func:`compare` (unit-testable with
    synthetic :class:`VariantDispersion` arrays).
    """
    if len(vdisps) < 1:
        raise ValueError("need at least one variant")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_gamma = _render(
        vdisps, out_dir, quantity="gamma", filename=_FIG_GAMMA,
        cmap=gamma_cmap, cbar_label=_GAMMA_LABEL,
        min_freq_thz=min_freq_thz, dpi=dpi, temperature=temperature,
    )
    fig_tau = _render(
        vdisps, out_dir, quantity="tau", filename=_FIG_TAU,
        cmap=tau_cmap, cbar_label=_TAU_LABEL,
        min_freq_thz=min_freq_thz, dpi=dpi, temperature=temperature,
    )
    summary = _write_summary(vdisps, out_dir)
    return {
        "linewidth_figure": fig_gamma,
        "lifetime_figure": fig_tau,
        "summary_csv": summary,
        "variants": [v.label for v in vdisps],
    }


def compare(
    variants,
    mesh,
    temperature: float = 50.0,
    out_dir="." ,
    *,
    npoints: int = 101,
    gamma_cmap: str = "inferno",
    tau_cmap: str = "viridis",
    min_freq_thz: float = 0.1,
    dpi: int = 300,
) -> dict:
    """Build the gamma/tau comparison figures + CSV for a list of variants.

    Parameters
    ----------
    variants : list of dict or tuple
        Each variant provides ``label``, ``phono3py_yaml``, ``fc2_path`` and
        ``gamma_npz`` (dict keys, or a ``(label, yaml, fc2, gamma[, units])`` tuple).
        Optional ``fc2_units`` (``'ev'`` default / ``'ry'``) or explicit ``factor``
        selects the THz conversion (correction C6).
    mesh : sequence of 3 int
        The BZ mesh the gamma npz lives on (used for the interpolation length scale).
    temperature : float
        Temperature (K) for the figure titles.
    out_dir : path-like
        Output directory; receives ``dispersion_linewidth.{png,pdf}``,
        ``dispersion_lifetime.{png,pdf}`` and ``comparison_summary.csv``.

    Returns
    -------
    dict with the written figure/CSV paths and the ordered variant labels.
    """
    vdisps = [_prepare_variant(v, mesh, npoints) for v in variants]
    return _compare_core(
        vdisps, out_dir, temperature=temperature, gamma_cmap=gamma_cmap,
        tau_cmap=tau_cmap, min_freq_thz=min_freq_thz, dpi=dpi,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_variant_spec(spec: str, default_units: str) -> dict:
    """Parse ``name=yaml,fc2,gamma[,units[,injected_modes]]`` into a variant dict."""
    name, sep, rest = spec.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(
            f"variant spec {spec!r} must be NAME=YAML,FC2,GAMMA[,UNITS[,INJECTED_MODES]]"
        )
    parts = [p.strip() for p in rest.split(",")]
    if len(parts) < 3 or not all(parts[:3]):
        raise argparse.ArgumentTypeError(
            f"variant spec {spec!r} must be NAME=YAML,FC2,GAMMA[,UNITS[,INJECTED_MODES]]"
        )
    out = dict(
        label=name.strip(), phono3py_yaml=parts[0], fc2_path=parts[1],
        gamma_npz=parts[2], fc2_units=parts[3] if len(parts) > 3 and parts[3] else default_units,
    )
    if len(parts) > 4 and parts[4]:
        out["injected_modes"] = parts[4]
    if len(parts) > 5:
        raise argparse.ArgumentTypeError(
            f"variant spec {spec!r} has too many fields; expected NAME=YAML,FC2,GAMMA[,UNITS[,INJECTED_MODES]]"
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare phonon dispersions colored by linewidth gamma and "
                    "lifetime tau across variants (shared colorbar per figure).",
    )
    ap.add_argument(
        "--variants", nargs="+", action="append", required=True,
        metavar="NAME=YAML,FC2,GAMMA[,UNITS[,INJECTED_MODES]]",
        help="repeatable (or space-separated) variant specs; UNITS in {ev,ry}.",
    )
    ap.add_argument("--mesh", nargs=3, type=int, required=True,
                    metavar=("MX", "MY", "MZ"), help="BZ mesh of the gamma npz.")
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--npoints", type=int, default=101,
                    help="band-path sampling density.")
    ap.add_argument("--fc2-units", default="ev", choices=["ev", "ry"],
                    help="default fc2 units for variants without an explicit UNITS.")
    ap.add_argument("--gamma-cmap", default="inferno")
    ap.add_argument("--tau-cmap", default="viridis")
    ap.add_argument("--min-freq-thz", type=float, default=0.1)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    # flatten repeated / space-separated --variants
    flat = [s for group in args.variants for s in group]
    variants = [_parse_variant_spec(s, args.fc2_units) for s in flat]

    result = compare(
        variants, mesh=args.mesh, temperature=args.temperature, out_dir=args.out_dir,
        npoints=args.npoints, gamma_cmap=args.gamma_cmap, tau_cmap=args.tau_cmap,
        min_freq_thz=args.min_freq_thz, dpi=args.dpi,
    )
    print("Wrote:")
    print(f"  {result['linewidth_figure']}")
    print(f"  {result['lifetime_figure']}")
    print(f"  {result['summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
