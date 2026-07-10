"""Self-contained ASE/Phonopy helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from phonopy import Phonopy, load
from phonopy.structure.atoms import PhonopyAtoms


def ase_to_phonopy_atoms(atoms: Atoms, *, include_masses: bool = True) -> PhonopyAtoms:
    """Convert ASE atoms to Phonopy atoms."""

    if include_masses:
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.cell,
            masses=atoms.get_masses(),
        )
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell,
    )


def phonopy_atoms_to_ase(atoms: PhonopyAtoms, *, include_masses: bool = True) -> Atoms:
    """Convert Phonopy atoms to ASE atoms."""

    if include_masses:
        return Atoms(
            cell=atoms.cell,
            scaled_positions=atoms.scaled_positions,
            numbers=atoms.numbers,
            masses=atoms.masses,
            pbc=True,
        )
    return Atoms(
        cell=atoms.cell,
        scaled_positions=atoms.scaled_positions,
        numbers=atoms.numbers,
        pbc=True,
    )


def copy_repeated_arrays(reference: Atoms, displaced: Atoms, *, array_names=None) -> Atoms:
    """Copy per-atom arrays from a primitive cell onto a displaced supercell."""

    repeats = len(displaced) // len(reference)
    if repeats * len(reference) != len(displaced):
        raise ValueError(
            "Displaced supercell atom count is not an integer multiple of the reference atom count."
        )

    for name, values in reference.arrays.items():
        if array_names is not None and name not in array_names:
            continue
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
    *,
    input_format=None,
    array_names=None,
    include_masses: bool = True,
) -> tuple[object, list[Atoms]]:
    """Load phonopy displacements as ASE supercells with copied metadata arrays."""

    relaxed_atoms = read(relaxed_file, index=-1, format=input_format)
    phonon = load(str(phonopy_yaml))
    displaced = [
        phonopy_atoms_to_ase(sc, include_masses=include_masses)
        for sc in phonon.supercells_with_displacements
    ]
    displaced = [
        copy_repeated_arrays(relaxed_atoms, atoms, array_names=array_names)
        for atoms in displaced
    ]
    return phonon, displaced
