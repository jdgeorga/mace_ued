"""Generate Phonopy and Phono3py displacement YAML files for linewidth workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read
from phono3py import Phono3py

from .. import phonopy_io


def parse_phonopy_args() -> argparse.Namespace:
    """Parse command-line arguments for second-order Phonopy displacements."""

    parser = argparse.ArgumentParser(description="Generate a Phonopy displacement YAML file.")
    parser.add_argument("input_xyz", help="Relaxed input structure.")
    parser.add_argument("--supercell", nargs=3, type=int, default=[3, 3, 1], help="Supercell dimensions.")
    parser.add_argument("--output", default="phonon_with_displacements.yaml", help="Output YAML filename.")
    return parser.parse_args()


def main_phonopy() -> None:
    """Generate second-order Phonopy displacements."""

    args = parse_phonopy_args()
    phonopy_io.generate_displacements(
        Path(args.input_xyz),
        Path(args.output),
        supercell=tuple(args.supercell),
        distance=0.01,
        is_diagonal=True,
        is_plusminus=False,
        symprec=1.0e-5,
    )


def parse_phono3py_args() -> argparse.Namespace:
    """Parse command-line arguments for third-order Phono3py displacements."""

    parser = argparse.ArgumentParser(description="Generate a Phono3py displacement YAML file.")
    parser.add_argument("input_xyz", help="Relaxed input structure.")
    parser.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 1], help="FC3 supercell dimensions.")
    parser.add_argument(
        "--phonon-supercell",
        nargs=3,
        type=int,
        default=[2, 2, 1],
        help="Phonon (FC2) supercell dimensions.",
    )
    parser.add_argument("--output", default="phono3py_disp.yaml", help="Output YAML filename.")
    return parser.parse_args()


def main_phono3py() -> None:
    """Generate third-order Phono3py displacements without primitive-cell detection."""

    args = parse_phono3py_args()
    atoms = read(args.input_xyz)
    phono3py_obj = Phono3py(
        unitcell=phonopy_io.ase_to_phonopy_atoms(atoms, include_masses=False),
        supercell_matrix=tuple(args.supercell),
        phonon_supercell_matrix=tuple(args.phonon_supercell),
        primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        log_level=1,
    )
    phono3py_obj.generate_displacements()
    phono3py_obj.save(args.output, settings={"displacements": True})


if __name__ == "__main__":
    main_phonopy()
