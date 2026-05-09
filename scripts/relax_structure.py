#!/usr/bin/env python3
"""Relax a structure with a MACE foundation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_phonon_scattering.calculator import add_mace_arguments, config_from_args
from mlip_phonon_scattering.relax import relax_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input structure readable by ASE.")
    parser.add_argument("output_prefix", help="Output prefix for relaxed structure and trajectory.")
    parser.add_argument("--input-format", default=None, help="Optional ASE input format override.")
    parser.add_argument("--fmax", type=float, default=1.0e-5, help="Relaxation force threshold in eV/Angstrom.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum optimizer steps.")
    parser.add_argument("--maxstep", type=float, default=0.05, help="Maximum FIRE step size.")
    add_mace_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relax_structure(
        input_file=Path(args.input_file).resolve(),
        output_prefix=Path(args.output_prefix).resolve(),
        calculator_config=config_from_args(args),
        fmax=args.fmax,
        steps=args.steps,
        maxstep=args.maxstep,
        input_format=args.input_format,
    )


if __name__ == "__main__":
    main()
