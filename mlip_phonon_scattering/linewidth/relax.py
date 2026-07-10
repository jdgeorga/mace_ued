"""Relax a structure for the phonon-linewidth workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..calculator import add_interlayer_arguments, add_mace_arguments, config_from_args
from ..relax import relax_structure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for linewidth relaxation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xyz", help="Input structure in extxyz format.")
    parser.add_argument("output_prefix", help="Output prefix for relaxed structure and trajectory.")
    parser.add_argument("--fmax", type=float, default=1.3e-5, help="Relaxation force threshold in eV/Angstrom.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum optimizer steps.")
    parser.add_argument("--maxstep", type=float, default=0.05, help="Maximum FIRE step size.")
    add_mace_arguments(parser)
    parser.set_defaults(device="cuda")
    add_interlayer_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the package relaxation workflow with linewidth defaults."""

    args = parse_args()
    relax_structure(
        Path(args.input_xyz),
        Path(args.output_prefix),
        config_from_args(args),
        fmax=args.fmax,
        steps=args.steps,
        maxstep=args.maxstep,
        input_format="extxyz",
        require_convergence=True,
        append_output_suffixes=True,
    )


if __name__ == "__main__":
    main()
