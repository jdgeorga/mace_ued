"""Force and energy evaluation on phonopy displacement supercells."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .calculator import MACECalculatorConfig, build_calculator
from .phonopy_io import load_displaced_supercells


def _chunk_indices(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    base = n_items // n_chunks
    remainder = n_items % n_chunks
    chunks: list[tuple[int, int]] = []
    start = 0
    for i in range(n_chunks):
        stop = start + base + (1 if i < remainder else 0)
        chunks.append((start, stop))
        start = stop
    return chunks


def _evaluate_structures(atoms_list: Iterable, calculator_config: MACECalculatorConfig):
    atoms_list = list(atoms_list)
    if not atoms_list:
        return [], []
    calc = build_calculator(calculator_config, atoms_list[0])
    forces = []
    energies = []
    for i, atoms in enumerate(atoms_list, start=1):
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()))
        forces.append(atoms.get_forces())
        if i == 1 or i % 10 == 0:
            print(f"Computed MACE forces for {i} local displacement structures", flush=True)
    return forces, energies


def compute_displacement_forces(
    relaxed_file: Path,
    phonopy_yaml: Path,
    forces_output: Path,
    energies_output: Path,
    calculator_config: MACECalculatorConfig,
) -> None:
    """Compute MACE forces for all phonopy displacements.

    Uses MPI if launched under ``mpirun``/``srun`` with ``mpi4py`` available.
    Otherwise runs serially.
    """

    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None

    phonon, displaced = load_displaced_supercells(relaxed_file, phonopy_yaml)
    n_structures = len(displaced)

    if MPI is None:
        print(f"Running serial force calculation for {n_structures} structures", flush=True)
        forces, energies = _evaluate_structures(displaced, calculator_config)
        np.save(forces_output, np.asarray(forces))
        np.save(energies_output, np.asarray(energies))
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running MPI force calculation on {size} ranks for {n_structures} structures", flush=True)
        chunks = [displaced[start:stop] for start, stop in _chunk_indices(n_structures, size)]
        print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}", flush=True)
    else:
        chunks = None

    local_atoms = comm.scatter(chunks, root=0)
    if len(local_atoms) == 0:
        local_forces: list[np.ndarray] = []
        local_energies: list[float] = []
        print(f"Rank {rank}: no structures assigned", flush=True)
    else:
        print(f"Rank {rank}: received {len(local_atoms)} structures", flush=True)
        local_forces, local_energies = _evaluate_structures(local_atoms, calculator_config)

    gathered_forces = comm.gather(local_forces, root=0)
    gathered_energies = comm.gather(local_energies, root=0)

    if rank == 0:
        forces = [force for chunk in gathered_forces for force in chunk]
        energies = [energy for chunk in gathered_energies for energy in chunk]
        np.save(forces_output, np.asarray(forces))
        np.save(energies_output, np.asarray(energies))
        print(f"Wrote {forces_output} and {energies_output}", flush=True)

    if MPI is not None:
        MPI.Finalize()
