#!/usr/bin/env python3
"""Compute MACE energies and forces for phonopy displacement supercells."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_phonon_scattering.calculator import add_mace_arguments, config_from_args
from mlip_phonon_scattering.forces import compute_displacement_forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relaxed_file", help="Relaxed primitive/supercell structure.")
    parser.add_argument("phonopy_yaml", help="Phonopy displacement YAML.")
    parser.add_argument("--forces-output", default="forces.npy", help="Output NumPy force array.")
    parser.add_argument("--energies-output", default="energies.npy", help="Output NumPy energy array.")
    add_mace_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_displacement_forces(
        relaxed_file=Path(args.relaxed_file).resolve(),
        phonopy_yaml=Path(args.phonopy_yaml).resolve(),
        forces_output=Path(args.forces_output).resolve(),
        energies_output=Path(args.energies_output).resolve(),
        calculator_config=config_from_args(args),
    )


if __name__ == "__main__":
    main()
