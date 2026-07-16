"""Vendored helpers from the `phonon_unfolding` package.

Only the functions the linewidth/lifetime workflow actually uses are copied here
so `phonon_unfolding` is not a runtime dependency (and the heavy `sklearn`
import it carried is dropped). Sourced from
phonon_unfolding/{displacement.py,utils.py} in the twist-anything tree.

Provides:
  - ase_to_phonopy_atoms / phonopy_atoms_to_ase / gen_ase  (structure conversion)
  - get_displacement_atoms                                 (phonopy 2nd-order displacements)
"""
from __future__ import annotations

import numpy as np


def phonopy_atoms_to_ase(atoms_phonopy):
    """Convert PhonopyAtoms to ASE Atoms."""
    from ase.atoms import Atoms

    return Atoms(
        cell=atoms_phonopy.cell,
        scaled_positions=atoms_phonopy.scaled_positions,
        numbers=atoms_phonopy.numbers,
        pbc=True,
    )


def gen_ase(cell, scaled_positions, numbers):
    """Convert cell, scaled_positions, and numbers to ASE Atoms."""
    from ase.atoms import Atoms

    return Atoms(cell=cell, scaled_positions=scaled_positions, numbers=numbers, pbc=True)


def ase_to_phonopy_atoms(ase_atoms):
    """Convert ASE Atoms to PhonopyAtoms."""
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        scaled_positions=ase_atoms.get_scaled_positions(),
        cell=ase_atoms.cell,
        masses=ase_atoms.get_masses(),
    )


def get_displacement_atoms(
    ase_atom, supercell_n, is_diagonal=True, is_plusminus=False,
    distance=0.01, symprec=0.00001,
):
    """Generate phonopy 2nd-order displacements for an ASE Atoms object.

    :param ase_atom: ASE Atoms object.
    :param supercell_n: three ints, the supercell dimensions.
    :return: (Phonopy object, list of displaced ASE Atoms).
    """
    from phonopy import Phonopy

    phonopy_atoms = ase_to_phonopy_atoms(ase_atom)
    print(np.diag(supercell_n))
    phonon = Phonopy(
        phonopy_atoms, supercell_matrix=np.diag(supercell_n), log_level=2, symprec=symprec
    )
    phonon.generate_displacements(
        is_diagonal=is_diagonal, is_plusminus=is_plusminus, distance=distance
    )
    ase_disp_list = [phonopy_atoms_to_ase(x) for x in phonon.supercells_with_displacements]
    return phonon, ase_disp_list
