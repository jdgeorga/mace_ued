#!/usr/bin/env python3
"""Relax a structure with Quantum ESPRESSO pw.x."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlip_phonon_scattering.qe_calculator import add_qe_arguments, qe_config_from_args
from mlip_phonon_scattering.qe_relax import relax_structure_qe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input structure readable by ASE.")
    parser.add_argument("output_prefix", help="Output prefix for relaxed structure.")
    add_qe_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relax_structure_qe(
        input_file=Path(args.input_file).resolve(),
        output_prefix=Path(args.output_prefix).resolve(),
        config=qe_config_from_args(args),
    )


if __name__ == "__main__":
    main()
