#!/usr/bin/env python3
"""Generate phonopy displacement YAML from a relaxed structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_phonon_scattering.phonopy_io import generate_displacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Relaxed structure readable by ASE.")
    parser.add_argument("--supercell", nargs=3, type=int, default=[3, 3, 1], help="Phonopy supercell dimensions.")
    parser.add_argument("--output", default="phonon_with_displacements.yaml", help="Output phonopy YAML file.")
    parser.add_argument("--distance", type=float, default=0.01, help="Displacement distance in Angstrom.")
    parser.add_argument("--symprec", type=float, default=1.0e-5, help="Phonopy symmetry tolerance.")
    parser.add_argument("--no-diagonal", action="store_true", help="Disable diagonal displacement generation.")
    parser.add_argument("--plusminus", action="store_true", help="Generate plus/minus displacements.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = generate_displacements(
        input_file=Path(args.input_file).resolve(),
        output_file=Path(args.output).resolve(),
        supercell=tuple(args.supercell),
        distance=args.distance,
        is_diagonal=not args.no_diagonal,
        is_plusminus=args.plusminus,
        symprec=args.symprec,
    )
    print(f"Wrote {args.output} with {count} displacements")


if __name__ == "__main__":
    main()
