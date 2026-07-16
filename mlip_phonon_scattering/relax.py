"""ASE relaxation with a MACE foundation-model calculator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE

from .calculator import MACECalculatorConfig, build_calculator


def max_force_magnitude(atoms) -> float:
    """Return the maximum per-atom force magnitude."""

    return float(np.sqrt((atoms.get_forces()**2).sum(axis=1)).max())


def relax_structure(
    input_file: Path,
    output_prefix: Path,
    calculator_config: MACECalculatorConfig,
    fmax: float = 1.0e-5,
    steps: int = 1000,
    maxstep: float = 0.05,
    input_format: str | None = None,
    *,
    require_convergence: bool = False,
    append_output_suffixes: bool = False,
) -> None:
    """Relax an input structure and write ``.xyz``, ``.traj``, and ``.traj.xyz`` outputs."""

    atoms = read(input_file, format=input_format)
    atoms.calc = build_calculator(calculator_config, atoms)

    unrelaxed_energy = atoms.get_potential_energy()
    print(f"Unrelaxed total energy: {unrelaxed_energy:.8f} eV", flush=True)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path = (
        Path(f"{output_prefix}.traj")
        if append_output_suffixes
        else output_prefix.with_suffix(".traj")
    )
    dyn = FIRE(atoms, trajectory=str(trajectory_path), maxstep=maxstep)
    converged = dyn.run(fmax=fmax, steps=steps)
    if require_convergence and (not converged or max_force_magnitude(atoms) > fmax):
        raise RuntimeError("Structure relaxation did not converge to the requested fmax.")

    relaxed_energy = atoms.get_potential_energy()
    print(f"Relaxed total energy: {relaxed_energy:.8f} eV", flush=True)

    relaxed_xyz = (
        Path(f"{output_prefix}.xyz")
        if append_output_suffixes
        else output_prefix.with_suffix(".xyz")
    )
    write(relaxed_xyz, atoms, format="extxyz")

    if trajectory_path.exists():
        images = [image for image in Trajectory(str(trajectory_path))]
        write(f"{output_prefix}.traj.xyz", images, format="extxyz")
