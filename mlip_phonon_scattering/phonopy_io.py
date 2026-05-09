"""Self-contained ASE/Phonopy helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from phonopy import Phonopy, load
from phonopy.structure.atoms import PhonopyAtoms


def ase_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE atoms to Phonopy atoms."""

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell,
        masses=atoms.get_masses(),
    )


def phonopy_atoms_to_ase(atoms: PhonopyAtoms) -> Atoms:
    """Convert Phonopy atoms to ASE atoms."""

    return Atoms(
        cell=atoms.cell,
        scaled_positions=atoms.scaled_positions,
        numbers=atoms.numbers,
        masses=atoms.masses,
        pbc=True,
    )


def copy_repeated_arrays(reference: Atoms, displaced: Atoms) -> Atoms:
    """Copy per-atom arrays from a primitive cell onto a displaced supercell."""

    repeats = len(displaced) // len(reference)
    if repeats * len(reference) != len(displaced):
        raise ValueError(
            "Displaced supercell atom count is not an integer multiple of the reference atom count."
        )

    for name, values in reference.arrays.items():
        if name in {"numbers", "positions", "masses"}:
            continue
        if len(values) == len(reference):
            displaced.arrays[name] = np.repeat(values, repeats, axis=0)
    return displaced


def generate_displacements(
    input_file: Path,
    output_file: Path,
    supercell: tuple[int, int, int],
    distance: float = 0.01,
    is_diagonal: bool = True,
    is_plusminus: bool = False,
    symprec: float = 1.0e-5,
) -> int:
    """Generate and save a phonopy displacement YAML file."""

    atoms = read(input_file)
    phonon = Phonopy(
        ase_to_phonopy_atoms(atoms),
        supercell_matrix=np.diag(supercell),
        log_level=2,
        symprec=symprec,
    )
    phonon.generate_displacements(
        is_diagonal=is_diagonal,
        is_plusminus=is_plusminus,
        distance=distance,
    )
    phonon.save(str(output_file))
    return len(phonon.displacements)


def load_displaced_supercells(
    relaxed_file: Path,
    phonopy_yaml: Path,
) -> tuple[object, list[Atoms]]:
    """Load phonopy displacements as ASE supercells with copied metadata arrays."""

    relaxed_atoms = read(relaxed_file, index=-1)
    phonon = load(str(phonopy_yaml))
    displaced = [phonopy_atoms_to_ase(sc) for sc in phonon.supercells_with_displacements]
    displaced = [copy_repeated_arrays(relaxed_atoms, atoms) for atoms in displaced]
    return phonon, displaced
