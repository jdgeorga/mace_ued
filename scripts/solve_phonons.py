#!/usr/bin/env python3
"""Solve phonons from phonopy displacements and MACE forces."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_phonon_scattering.solve import solve_phonons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xyz", required=True, help="Relaxed structure.")
    parser.add_argument("--forces", required=True, help="Forces NumPy array from compute_mace_forces.py.")
    parser.add_argument("--phonopy-yaml", required=True, help="Phonopy displacement YAML.")
    parser.add_argument("--mesh", nargs=3, type=int, default=[36, 36, 1], help="Gamma-centered phonopy mesh.")
    parser.add_argument(
        "--qpoints-file",
        default=None,
        help="Optional whitespace file of explicit fractional q-points for eigenvector.h5.",
    )
    parser.add_argument("--band-npoints", type=int, default=31, help="Points per band path segment.")
    parser.add_argument(
        "--band-path",
        default=None,
        help="Optional ASE compact path string, e.g. GXMGRX,MR. Defaults to ASE's path for the input lattice.",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solve_phonons(
        input_xyz=Path(args.input_xyz).resolve(),
        forces_path=Path(args.forces).resolve(),
        phonopy_yaml=Path(args.phonopy_yaml).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        mesh=tuple(args.mesh),
        qpoints_file=Path(args.qpoints_file).resolve() if args.qpoints_file else None,
        band_npoints=args.band_npoints,
        band_path_string=args.band_path,
    )
    print(f"Wrote phonon outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
