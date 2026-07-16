"""QE-based force evaluation on phonopy displaced supercells."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .phonopy_io import load_displaced_supercells
from .qe_calculator import QECalculatorConfig, parse_qe_output, run_pw_subprocess, write_qe_input


def _run_one_displacement(args: tuple) -> tuple[int, np.ndarray, float]:
    """Worker: write input, run pw.x, return (index, forces, energy).

    Runs in a subprocess spawned by ProcessPoolExecutor, so it re-imports
    everything it needs from the package.
    """
    idx, disp_dir_str, config_dict = args
    from mlip_phonon_scattering.qe_calculator import (
        QECalculatorConfig,
        parse_qe_output,
        run_pw_subprocess,
    )
    import dataclasses

    config = QECalculatorConfig(**config_dict)
    disp_dir = Path(disp_dir_str)
    pwi = disp_dir / "scf.pwi"
    pwo = disp_dir / "scf.pwo"

    if pwo.exists():
        text = pwo.read_text()
        if "JOB DONE" in text:
            print(f"  disp {idx:04d}: skipping (already done)", flush=True)
            atoms = parse_qe_output(pwo)
            return idx, atoms.get_forces(), float(atoms.get_potential_energy())

    print(f"  disp {idx:04d}: launching pw.x in {disp_dir}", flush=True)
    proc = run_pw_subprocess(pwi, pwo, config.nranks_per_disp, config)
    proc.wait()
    atoms = parse_qe_output(pwo)
    print(f"  disp {idx:04d}: done (E={atoms.get_potential_energy():.6f} eV)", flush=True)
    return idx, atoms.get_forces(), float(atoms.get_potential_energy())


def compute_displacement_forces_qe(
    relaxed_file: Path,
    phonopy_yaml: Path,
    forces_output: Path,
    energies_output: Path,
    config: QECalculatorConfig,
) -> None:
    """Compute QE forces for all phonopy displacements in parallel.

    Writes all pw.x inputs first, then launches up to ``config.max_concurrent``
    QE calculations simultaneously using ``srun --exact --overlap``.
    Each QE job uses ``config.nranks_per_disp`` MPI ranks (GPUs).

    Skips displacements whose output already contains "JOB DONE" (restart-safe).
    """
    relaxed_file = Path(relaxed_file).resolve()
    phonopy_yaml = Path(phonopy_yaml).resolve()
    forces_output = Path(forces_output).resolve()
    energies_output = Path(energies_output).resolve()

    _, displaced = load_displaced_supercells(relaxed_file, phonopy_yaml)
    n = len(displaced)
    run_dir = forces_output.parent / "qe_disps"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing QE inputs for {n} displaced supercells in {run_dir}", flush=True)
    disp_dirs: list[Path] = []
    for i, atoms in enumerate(displaced):
        d = run_dir / f"disp_{i:04d}"
        d.mkdir(exist_ok=True)
        pwi = d / "scf.pwi"
        if not pwi.exists():
            write_qe_input(atoms, config, pwi, calculation="scf")
        disp_dirs.append(d)

    # Pass config as plain dict so it survives pickling across processes
    import dataclasses
    config_dict = {
        f.name: getattr(config, f.name) for f in dataclasses.fields(config)
    }

    worker_args = [(i, str(disp_dirs[i]), config_dict) for i in range(n)]

    forces_list: list[np.ndarray | None] = [None] * n
    energies_list: list[float | None] = [None] * n

    print(
        f"Running {n} QE SCF calculations ({config.max_concurrent} concurrent, "
        f"{config.nranks_per_disp} ranks each)",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=config.max_concurrent) as pool:
        future_to_idx = {pool.submit(_run_one_displacement, arg): arg[0] for arg in worker_args}
        completed = 0
        for future in as_completed(future_to_idx):
            idx, frc, eng = future.result()
            forces_list[idx] = frc
            energies_list[idx] = eng
            completed += 1
            print(f"Progress: {completed}/{n} displacements done", flush=True)

    forces = np.array(forces_list)     # (n, n_atoms, 3) eV/Å
    energies = np.array(energies_list) # (n,) eV
    np.save(forces_output, forces)
    np.save(energies_output, energies)
    print(f"Wrote {forces_output} shape={forces.shape}", flush=True)
    print(f"Wrote {energies_output} shape={energies.shape}", flush=True)
