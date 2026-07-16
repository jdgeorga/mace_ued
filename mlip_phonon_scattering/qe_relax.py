"""QE-based structure relaxation using pw.x."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write

from .qe_calculator import QECalculatorConfig, parse_qe_output, run_pw_subprocess, write_qe_input


def relax_structure_qe(
    input_file: Path,
    output_prefix: Path,
    config: QECalculatorConfig,
) -> None:
    """Relax a structure with QE pw.x (calculation=relax) and write relaxed .xyz.

    The QE run directory is ``<output_prefix>_qe_relax/``.
    Outputs: ``<output_prefix>.xyz`` (ASE extxyz, relaxed geometry + forces).
    """
    input_file = Path(input_file).resolve()
    output_prefix = Path(output_prefix).resolve()
    run_dir = output_prefix.parent / (output_prefix.name + "_qe_relax")
    run_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(str(input_file))
    print(f"Relaxing {len(atoms)}-atom structure with QE (relax) in {run_dir}", flush=True)

    pwi = run_dir / "scf.pwi"
    pwo = run_dir / "scf.pwo"
    write_qe_input(atoms, config, pwi, calculation="relax")

    proc = run_pw_subprocess(pwi, pwo, config.nranks_per_disp, config)
    proc.wait()

    relaxed = parse_qe_output(pwo)
    energy = relaxed.get_potential_energy()
    forces = relaxed.get_forces()
    fmax = float((forces ** 2).sum(axis=1).max() ** 0.5)
    print(f"Relaxed total energy: {energy:.8f} eV  |F|_max: {fmax:.2e} eV/Å", flush=True)

    out_xyz = output_prefix.with_suffix(".xyz")
    write(str(out_xyz), relaxed, format="extxyz")
    print(f"Wrote relaxed structure to {out_xyz}", flush=True)
