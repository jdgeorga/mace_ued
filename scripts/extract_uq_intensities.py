#!/usr/bin/env python3
"""Extract tiled Qz=0 UED intensities from a phonon eigenvector HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_phonon_scattering.ued_intensity import write_qz0_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, help="Input uniform-mesh eigenvector.h5 file.")
    parser.add_argument("--output-dir", required=True, help="Directory for tiled CSVs and figures.")
    parser.add_argument("--temperature-k", type=float, default=100.0, help="Bose occupation temperature.")
    parser.add_argument(
        "--temperature-sweep-k",
        type=float,
        nargs="*",
        default=list(range(0, 1501, 50)),
        help=(
            "Temperatures for Bragg-point Debye-Waller and zero-phonon plots. "
            "Pass the flag with no values to skip temperature-dependent outputs."
        ),
    )
    parser.add_argument(
        "--temperature-target-g",
        type=float,
        nargs=3,
        action="append",
        metavar=("H", "K", "L"),
        default=None,
        help=(
            "Reciprocal-lattice vector for temperature-dependent Bragg diagnostics. "
            "Repeat for multiple targets."
        ),
    )
    parser.add_argument("--phwmin-mev", type=float, default=0.2, help="Low-energy phonon cutoff in meV.")
    parser.add_argument(
        "--gmax",
        type=float,
        default=3.0,
        help="Radial |Q| cutoff in units of the shortest in-plane primitive reciprocal-vector length.",
    )
    parser.add_argument("--qz", type=float, default=0.0, help="Fractional reciprocal q_3 slice.")
    parser.add_argument("--qz-tol", type=float, default=1.0e-8, help="Absolute tolerance for the fractional q_3 slice.")
    parser.add_argument(
        "--eigenvector-gauge",
        choices=("raw", "phonopy_to_phx"),
        default="phonopy_to_phx",
        help="Eigenvector phase convention for phonopy HDF5 input.",
    )
    scattering_choices = ("peng", "peng_low_s", "peng_high_s", "atomic_number", "unity", "vand5_or_z", "vand5")
    parser.add_argument(
        "--electron-scattering-model",
        dest="electron_scattering_model",
        choices=scattering_choices,
        default="peng",
        help=(
            "Elastic electron atomic scattering-factor model. "
            "peng selects the low-s or high-s Peng coefficient block from s=|Q|/(4*pi)."
        ),
    )
    parser.add_argument(
        "--form-factor-model",
        dest="electron_scattering_model",
        choices=scattering_choices,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-tiled-csv",
        action="store_true",
        help="Do not write tiled_intensities_qz0.csv or tiled_intensities_qz0_long.csv.",
    )
    parser.add_argument("--no-long", action="store_true", help="Do not write the mode-resolved long CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperature_target_g = args.temperature_target_g or [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    outputs = write_qz0_outputs(
        input_h5=Path(args.input_h5).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        temperature_k=args.temperature_k,
        phwmin_mev=args.phwmin_mev,
        gmax=args.gmax,
        qz=args.qz,
        qz_tol=args.qz_tol,
        electron_scattering_model=args.electron_scattering_model,
        eigenvector_gauge=args.eigenvector_gauge,
        write_tiled_csv=not args.no_tiled_csv,
        write_long=not args.no_long,
        temperature_sweep_k=args.temperature_sweep_k or None,
        temperature_target_g_frac=temperature_target_g,
    )
    print(f"Wrote UED intensity outputs to {Path(args.output_dir).resolve()}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
