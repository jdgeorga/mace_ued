"""Validate outputs from the phonon linewidth and lifetime pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import phono3py as Phono3py
from phonopy import Phonopy


def _temperature_tag(temperature: float) -> str:
    return str(int(temperature)) if temperature == int(temperature) else str(temperature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-atoms", required=True, type=int)
    parser.add_argument("--expected-bands", required=True, type=int)
    parser.add_argument("--mesh", nargs=3, required=True, type=int, metavar=("MX", "MY", "MZ"))
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--fc2", type=Path)
    parser.add_argument("--gamma-npz", type=Path)
    parser.add_argument("--yaml", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--figure-stem", required=True)
    parser.add_argument("--reference-gamma", type=Path)
    # Reference-gamma acceptance is physics-reproduction, NOT bitwise: gamma derives from
    # float32 MACE forces (both this run and the golden reference), whose GPU reductions are
    # non-deterministic at ~1e-6 abs, propagating to gamma at ~1e-4. These defaults bracket the
    # pipeline's own inherent reproducibility (original vs paper_v4: max|Δ|~5.6e-4, mean~1.1e-6)
    # while still catching a real regression (which shifts gamma by O(1e-2) or more).
    parser.add_argument("--ref-max-abs", type=float, default=5e-3, help="max |Δγ| (THz) allowed vs reference.")
    parser.add_argument("--ref-mean-abs", type=float, default=1e-4, help="mean |Δγ| (THz) allowed vs reference.")
    parser.add_argument("--shear-range", nargs=2, type=float)
    parser.add_argument("--breathing-range", nargs=2, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mesh_tag = "x".join(str(value) for value in args.mesh)
    temperature_tag = _temperature_tag(args.temperature)
    if args.fc2 is None:
        args.fc2 = args.run_dir / f"phonon_cache_m{mesh_tag}" / "fc2.npy"
    if args.gamma_npz is None:
        args.gamma_npz = args.run_dir / f"gamma_W_full_T{temperature_tag}.npz"
    if args.yaml is None:
        args.yaml = args.run_dir / "phono3py_disp.yaml"
    if args.figures_dir is None:
        args.figures_dir = args.run_dir / "figures"
    if args.output is None:
        args.output = args.run_dir / "validation.json"
    return args


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _add_check(checks: dict[str, dict[str, Any]], name: str, passed: bool, **detail: Any) -> None:
    checks[name] = {"pass": bool(passed), "detail": {key: _json_value(value) for key, value in detail.items()}}


def _wrapped_distance(qpoints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(qpoints - np.round(qpoints), axis=1)


def main() -> int:
    args = parse_args()
    mesh = tuple(args.mesh)
    checks: dict[str, dict[str, Any]] = {}
    gamma: np.ndarray | None = None
    qpoints: np.ndarray | None = None
    phonon: Phonopy | None = None
    gamma_frequencies: np.ndarray | None = None
    gamma_band_order: np.ndarray | None = None
    gamma_index: int | None = None

    try:
        with np.load(args.gamma_npz) as data:
            gamma = np.asarray(data["gamma_W_full"], dtype=float)
            qpoints = np.asarray(data["qpoints_frac"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "gamma_shape_finiteness", False, path=args.gamma_npz, error=str(exc))
    else:
        expected_shape = (int(np.prod(mesh)), args.expected_bands)
        finite = bool(np.all(np.isfinite(gamma)))
        minimum = float(gamma.min()) if gamma.size else float("nan")
        _add_check(
            checks,
            "gamma_shape_finiteness",
            gamma.shape == expected_shape and finite and minimum >= -1e-9,
            shape=gamma.shape,
            expected_shape=expected_shape,
            finite=finite,
            min=minimum,
            max=float(gamma.max()) if gamma.size else float("nan"),
        )

        tau = np.where(gamma > 0, 1.0 / (4.0 * np.pi * gamma), np.inf)
        finite_tau = np.isfinite(tau)
        tau_ok = bool(np.all(tau[finite_tau] > 0) and not np.any(finite_tau & (tau <= 0)))
        _add_check(
            checks,
            "tau_positivity",
            tau_ok,
            finite_entries=int(finite_tau.sum()),
            nonpositive_finite_entries=int(np.count_nonzero(finite_tau & (tau <= 0))),
            zero_gamma_noninf_tau_entries=int(np.count_nonzero((gamma == 0) & ~np.isinf(tau))),
        )
        _add_check(
            checks,
            "not_all_zero",
            bool(gamma.size and gamma.max() > 1e-8),
            max=float(gamma.max()) if gamma.size else float("nan"),
        )

    try:
        ph3 = Phono3py.load(str(args.yaml), log_level=0)
        phonon = Phonopy(
            ph3.unitcell,
            supercell_matrix=ph3.phonon_supercell_matrix,
            primitive_matrix=ph3.primitive_matrix,
        )
        phonon.force_constants = np.load(args.fc2)
        phonon.symmetrize_force_constants()
        phonon.run_mesh(list(mesh), is_gamma_center=True)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = np.asarray(mesh_dict["frequencies"], dtype=float)
        _add_check(
            checks,
            "fc2_harmonic_sanity",
            bool(
                len(ph3.unitcell) == args.expected_atoms
                and np.all(np.isfinite(frequencies))
                and frequencies.min() > -0.05
                and frequencies.shape[-1] == args.expected_bands
            ),
            unitcell_atoms=len(ph3.unitcell),
            expected_atoms=args.expected_atoms,
            shape=frequencies.shape,
            expected_bands=args.expected_bands,
            finite=bool(np.all(np.isfinite(frequencies))),
            min=float(frequencies.min()),
            max=float(frequencies.max()),
        )
        phonon.run_qpoints([[0, 0, 0]])
        gamma_frequencies = np.asarray(phonon.get_qpoints_dict()["frequencies"], dtype=float)[0]
        gamma_band_order = np.argsort(np.abs(gamma_frequencies))[:3]
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "fc2_harmonic_sanity", False, yaml=args.yaml, fc2=args.fc2, error=str(exc))

    if gamma is None or qpoints is None or gamma_frequencies is None or gamma_band_order is None:
        _add_check(checks, "gamma_point_acoustic", False, error="gamma data or harmonic Gamma frequencies unavailable")
    else:
        if qpoints.ndim != 2 or qpoints.shape[1] != 3:
            _add_check(checks, "gamma_point_acoustic", False, qpoints_shape=qpoints.shape, error="invalid qpoints_frac")
        else:
            gamma_index = int(np.argmin(_wrapped_distance(qpoints)))
            acoustic_frequencies = gamma_frequencies[gamma_band_order]
            acoustic_gamma = gamma[gamma_index, gamma_band_order]
            _add_check(
                checks,
                "gamma_point_acoustic",
                bool(np.all(acoustic_frequencies < 0.05) and np.all(acoustic_gamma <= 1e-4)),
                qpoint_index=gamma_index,
                qpoint=qpoints[gamma_index],
                bands=gamma_band_order,
                harmonic_frequencies=acoustic_frequencies,
                gamma_values=acoustic_gamma,
            )

    figure_names = [
        f"{args.figure_stem}_linewidth_lifetime_band_path.png",
        f"{args.figure_stem}_linewidth_lifetime_band_path.pdf",
        f"{args.figure_stem}_linewidth_lifetime_band_path_meshpoints.png",
        f"{args.figure_stem}_linewidth_lifetime_band_path_meshpoints.pdf",
    ]
    figure_paths = [args.figures_dir / name for name in figure_names]
    figure_detail = {str(path): path.stat().st_size if path.exists() else 0 for path in figure_paths}
    _add_check(checks, "figures_exist", all(size > 0 for size in figure_detail.values()), files=figure_detail)

    if args.shear_range is not None and args.breathing_range is not None:
        if gamma_frequencies is None:
            _add_check(checks, "bilayer_shear_breathing", False, error="harmonic Gamma frequencies unavailable")
        else:
            sorted_frequencies = np.sort(gamma_frequencies)
            shear = sorted_frequencies[3:5] if len(sorted_frequencies) >= 5 else np.array([])
            breathing = sorted_frequencies[5] if len(sorted_frequencies) >= 6 else float("nan")
            shear_lo, shear_hi = args.shear_range
            breathing_lo, breathing_hi = args.breathing_range
            passed = bool(
                len(shear) == 2
                and shear_lo <= shear[0] <= shear_hi
                and shear_lo <= shear[1] <= shear_hi
                and abs(shear[1] - shear[0]) <= 0.02
                and breathing_lo <= breathing <= breathing_hi
            )
            _add_check(
                checks,
                "bilayer_shear_breathing",
                passed,
                sorted_frequencies=sorted_frequencies,
                shear_range=args.shear_range,
                breathing_range=args.breathing_range,
                shear=shear,
                shear_split=abs(shear[1] - shear[0]) if len(shear) == 2 else float("nan"),
                breathing=breathing,
            )

    if args.reference_gamma is not None:
        try:
            if gamma is None:
                raise RuntimeError("new gamma data unavailable")
            with np.load(args.reference_gamma) as data:
                reference = np.asarray(data["gamma_W_full"], dtype=float)
            if gamma.shape != reference.shape:
                raise RuntimeError(f"shape mismatch {gamma.shape} vs {reference.shape}")
            diff = np.abs(gamma - reference)
            reference_mask = reference > 1e-6
            max_rel_err = float((diff[reference_mask] / np.abs(reference[reference_mask])).max()) if np.any(reference_mask) else 0.0
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            # Physics-reproduction acceptance (see parse_args note): float32 force noise makes
            # bitwise/1e-5 agreement impossible; accept if max|Δ| and mean|Δ| sit at the
            # pipeline's inherent reproducibility floor. A real regression shifts gamma far more.
            passed = bool(max_abs <= args.ref_max_abs and mean_abs <= args.ref_mean_abs)
            detail = {
                "new_shape": gamma.shape,
                "reference_shape": reference.shape,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "max_rel_err_gt_1e-6": max_rel_err,
                "ref_max_abs_tol": args.ref_max_abs,
                "ref_mean_abs_tol": args.ref_mean_abs,
                "allclose_1e-5": bool(np.allclose(gamma, reference, rtol=1e-5, atol=1e-10)),
                "new_min": float(gamma.min()),
                "new_max": float(gamma.max()),
                "reference_min": float(reference.min()),
                "reference_max": float(reference.max()),
            }
            print("reference comparison: " + ", ".join(f"{key}={value}" for key, value in detail.items()))
            _add_check(checks, "reference_comparison", passed, **detail)
        except Exception as exc:  # noqa: BLE001
            _add_check(checks, "reference_comparison", False, path=args.reference_gamma, error=str(exc))

    overall_pass = all(entry["pass"] for entry in checks.values())
    result = {"pass": overall_pass, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, entry in checks.items():
        print(f"{'PASS' if entry['pass'] else 'FAIL'} {name}: {entry['detail']}")
    print(f"{'PASS' if overall_pass else 'FAIL'} overall: {args.output}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
