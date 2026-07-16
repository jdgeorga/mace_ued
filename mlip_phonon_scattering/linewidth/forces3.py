"""Evaluate force-only Phonopy and Phono3py displacement calculations under MPI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import phono3py
from ase.io import read
from mpi4py import MPI

from .. import forces, phonopy_io
from ..calculator import add_interlayer_arguments, add_mace_arguments, config_from_args


def _partition_contiguous(n: int, rank: int, size: int) -> tuple[int, int]:
    """Return the start index and count for one contiguous MPI chunk."""

    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return start, count


def subtract_reference_forces(F, F0):
    """Subtract a per-atom reference force array F0 (nat,3) from a stack of
    force arrays F (n,nat,3), broadcasting F0 over the leading axis."""

    return np.asarray(F) - np.asarray(F0)[None]


def _run_force_evaluation(
    displaced,
    calculator_config,
    output: str,
    *,
    reference_atoms=None,
    subtract_reference: bool = False,
) -> None:
    """Scatter ordered structures, evaluate forces, then gather in source order."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        chunks = [
            displaced[start : start + count]
            for start, count in (_partition_contiguous(len(displaced), i, size) for i in range(size))
        ]
        print(f"Found {len(displaced)} structures to process", flush=True)
        print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}", flush=True)
    else:
        chunks = None

    local_list = comm.scatter(chunks, root=0)
    if local_list:
        local_forces, _ = forces.evaluate_structures(
            local_list,
            calculator_config,
            compute_energies=False,
            nlayer_atoms=local_list,
        )
    else:
        local_forces = []
        print(f"Rank {rank}: no structures assigned", flush=True)

    gathered = comm.gather(local_forces, root=0)
    if rank == 0:
        all_forces = np.array([f for rank_forces in gathered for f in rank_forces])
        if subtract_reference:
            if reference_atoms is None:
                raise ValueError("subtract_reference requires reference_atoms")
            ref_forces_list, _ = forces.evaluate_structures(
                [reference_atoms],
                calculator_config,
                compute_energies=False,
                nlayer_atoms=[reference_atoms],
            )
            f0 = np.asarray(ref_forces_list[0])
            all_forces = subtract_reference_forces(all_forces, f0)
            output_path = Path(output)
            ref_output = output_path.with_name(
                f"{output_path.stem}_reference_forces.npy"
            )
            np.save(ref_output, f0)
            print(f"Wrote {ref_output}", flush=True)
        np.save(output, all_forces)
        print(f"Wrote {output}", flush=True)


def _add_common_force_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("relaxed_xyz", help="Relaxed reference structure in extxyz format.")
    parser.add_argument("disp_yaml", help="Phonopy or Phono3py displacement YAML file.")
    add_mace_arguments(parser)
    parser.set_defaults(device="cuda")
    add_interlayer_arguments(parser)


def parse_forces2_args() -> argparse.Namespace:
    """Parse command-line arguments for Phonopy second-order forces."""

    parser = argparse.ArgumentParser(description="Generate forces for Phonopy displacements.")
    _add_common_force_arguments(parser)
    parser.add_argument("--output", default="disp_forces_mace.npy", help="Output force-array filename.")
    return parser.parse_args()


def main_forces2() -> None:
    """Evaluate forces for second-order Phonopy displacement supercells."""

    args = parse_forces2_args()
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        _, displaced = phonopy_io.load_displaced_supercells(
            Path(args.relaxed_xyz),
            Path(args.disp_yaml),
            input_format="extxyz",
            array_names=("atom_types", "layer_ids"),
            include_masses=False,
        )
    else:
        displaced = None
    _run_force_evaluation(displaced, config_from_args(args), args.output)


def _load_phono3py_displacements(relaxed_xyz: str, disp_yaml: str, supercell_matrix, *, phonon: bool):
    """Load and metadata-copy either FC3 or phonon Phono3py displacement cells."""

    relaxed_atoms = read(relaxed_xyz, format="extxyz", index=-1)
    ph3 = phono3py.load(disp_yaml, supercell_matrix=tuple(supercell_matrix), log_level=1)
    supercells = ph3.phonon_supercells_with_displacements if phonon else ph3.supercells_with_displacements
    displaced = [
        phonopy_io.copy_repeated_arrays(
            relaxed_atoms,
            phonopy_io.phonopy_atoms_to_ase(supercell, include_masses=False),
            array_names=("atom_types", "layer_ids"),
        )
        for supercell in supercells
        if supercell is not None
    ]
    reference_supercell = ph3.phonon_supercell if phonon else ph3.supercell
    reference_atoms = phonopy_io.copy_repeated_arrays(
        relaxed_atoms,
        phonopy_io.phonopy_atoms_to_ase(reference_supercell, include_masses=False),
        array_names=("atom_types", "layer_ids"),
    )
    return displaced, reference_atoms


def _parse_phono3py_force_args(*, phonon: bool) -> argparse.Namespace:
    description = (
        "Generate second-order forces from Phono3py phonon displacements."
        if phonon
        else "Generate third-order forces from Phono3py displacements."
    )
    parser = argparse.ArgumentParser(description=description)
    _add_common_force_arguments(parser)
    parser.add_argument(
        "--supercell-matrix",
        nargs=3,
        type=int,
        default=[1, 1, 1] if phonon else [2, 2, 1],
        help="Phono3py supercell dimensions.",
    )
    parser.add_argument(
        "--output",
        default="disp_forces_2nd_from_3rd_mace.npy" if phonon else "disp_forces_3rd_mace.npy",
        help="Output force-array filename.",
    )
    parser.add_argument(
        "--subtract-reference-forces",
        action="store_true",
        default=False,
        help=(
            "Evaluate MLIP forces on the undisplaced reference supercell (F0) and "
            "subtract it from every displaced-cell force array before saving; also "
            "writes <output_stem>_reference_forces.npy next to --output. Default OFF preserves "
            "existing behavior/golden outputs."
        ),
    )
    return parser.parse_args()


def main_forces3() -> None:
    """Evaluate forces for FC3 Phono3py displacement supercells."""

    args = _parse_phono3py_force_args(phonon=False)
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        displaced, reference_atoms = _load_phono3py_displacements(
            args.relaxed_xyz,
            args.disp_yaml,
            args.supercell_matrix,
            phonon=False,
        )
    else:
        displaced, reference_atoms = None, None
    _run_force_evaluation(
        displaced,
        config_from_args(args),
        args.output,
        reference_atoms=reference_atoms,
        subtract_reference=args.subtract_reference_forces,
    )


def main_forces2_from3() -> None:
    """Evaluate forces for Phono3py phonon (FC2) displacement supercells."""

    args = _parse_phono3py_force_args(phonon=True)
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        displaced, reference_atoms = _load_phono3py_displacements(
            args.relaxed_xyz,
            args.disp_yaml,
            args.supercell_matrix,
            phonon=True,
        )
    else:
        displaced, reference_atoms = None, None
    _run_force_evaluation(
        displaced,
        config_from_args(args),
        args.output,
        reference_atoms=reference_atoms,
        subtract_reference=args.subtract_reference_forces,
    )


if __name__ == "__main__":
    main_forces3()
