"""QE calculator configuration, input writing, subprocess execution, and output parsing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.atoms import Atoms
from ase.io import read
from ase.io.espresso import write_espresso_in


@dataclass(frozen=True)
class QECalculatorConfig:
    """Configuration for Quantum ESPRESSO pw.x calculations."""

    pseudo_dir: str
    pseudo_files: dict = field(default_factory=dict)     # {element: filename}
    ecutwfc: float = 60.0                                # Ry
    ecutrho: float = 480.0                               # Ry
    input_dft: str = "vdw-df-c09"
    assume_isolated: str = "2D"
    conv_thr: float = 1.0e-8
    mixing_mode: str = "local-TF"
    mixing_beta: float = 0.2
    degauss: float = 0.01                               # Ry
    disk_io: str = "medium"
    relax_kpoints: tuple = (12, 12, 1)
    relax_koffset: tuple = (0, 0, 0)
    scf_kpoints: tuple = (1, 1, 1)
    scf_koffset: tuple = (0, 0, 0)
    nranks_per_disp: int = 4
    max_concurrent: int = 8
    qe_module: str = "espresso/7.5-libxc-7.0.0-gpu"
    max_seconds: int = 1500

    def __post_init__(self):
        # Freeze mutable defaults
        if isinstance(self.pseudo_files, dict):
            object.__setattr__(self, "pseudo_files", dict(self.pseudo_files))
        object.__setattr__(self, "relax_kpoints", tuple(self.relax_kpoints))
        object.__setattr__(self, "relax_koffset", tuple(self.relax_koffset))
        object.__setattr__(self, "scf_kpoints", tuple(self.scf_kpoints))
        object.__setattr__(self, "scf_koffset", tuple(self.scf_koffset))


def _build_input_data(config: QECalculatorConfig, calculation: str) -> dict[str, Any]:
    """Assemble QE namelist dict for write_espresso_in."""
    data: dict[str, Any] = {
        "control": {
            "calculation": calculation,
            "tprnfor": True,
            "tstress": False,
            "disk_io": config.disk_io,
            "max_seconds": config.max_seconds,
        },
        "system": {
            "ecutwfc": config.ecutwfc,
            "ecutrho": config.ecutrho,
            "input_dft": config.input_dft,
            "assume_isolated": config.assume_isolated,
            "occupations": "smearing",
            "smearing": "gaussian",
            "degauss": config.degauss,
        },
        "electrons": {
            "conv_thr": config.conv_thr,
            "mixing_mode": config.mixing_mode,
            "mixing_beta": config.mixing_beta,
        },
    }
    if calculation == "relax":
        data["ions"] = {"ion_dynamics": "bfgs"}
    return data


def write_qe_input(
    atoms: Atoms,
    config: QECalculatorConfig,
    outpath: Path,
    calculation: str = "scf",
) -> None:
    """Write a pw.x input file for the given atoms and calculation type."""
    outpath = Path(outpath)
    if calculation == "relax":
        kpts = list(config.relax_kpoints)
        koffset = list(config.relax_koffset)
    else:
        kpts = list(config.scf_kpoints)
        koffset = list(config.scf_koffset)

    input_data = _build_input_data(config, calculation)
    with open(outpath, "w") as fh:
        write_espresso_in(
            fh,
            atoms,
            input_data=input_data,
            pseudopotentials=config.pseudo_files,
            kpts=kpts,
            koffset=koffset,
        )
    # Inject pseudo_dir into the written file (ASE does not write it)
    _inject_pseudo_dir(outpath, config.pseudo_dir)


def _inject_pseudo_dir(pwi_path: Path, pseudo_dir: str) -> None:
    """Insert pseudo_dir into the &CONTROL namelist of a pw.x input file."""
    text = pwi_path.read_text()
    if "pseudo_dir" in text:
        return
    text = text.replace(
        "&CONTROL",
        f"&CONTROL\n  pseudo_dir = '{pseudo_dir}',",
        1,
    )
    pwi_path.write_text(text)


def run_pw_subprocess(
    input_file: Path,
    output_file: Path,
    nranks: int,
    config: QECalculatorConfig,
) -> subprocess.Popen:
    """Launch pw.x via srun --exact --overlap and return the Popen handle."""
    input_file = Path(input_file).resolve()
    output_file = Path(output_file).resolve()
    ntasks_per_node = min(nranks, 4)
    n_nodes = max(1, nranks // ntasks_per_node)

    cmd = [
        "srun",
        "--exact",
        "--overlap",
        f"--ntasks={nranks}",
        f"--nodes={n_nodes}",
        f"--ntasks-per-node={ntasks_per_node}",
        "--gpus-per-task=1",
        "--gpu-bind=none",
        "pw.x",
        "-ndiag", "1",
        "-in", str(input_file),
    ]
    with open(output_file, "w") as out_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=out_fh,
            stderr=subprocess.STDOUT,
            cwd=str(input_file.parent),
        )
    return proc


def parse_qe_output(output_file: Path) -> Atoms:
    """Read the final geometry and properties from a pw.x output file.

    Returns an ASE Atoms object with a SinglePointCalculator attached,
    providing .get_forces() (eV/Å) and .get_potential_energy() (eV).
    """
    output_file = Path(output_file)
    if not output_file.exists():
        raise FileNotFoundError(f"QE output not found: {output_file}")
    text = output_file.read_text()
    if "JOB DONE" not in text:
        raise RuntimeError(f"QE job did not complete successfully: {output_file}")
    atoms = read(str(output_file), format="espresso-out", index=-1)
    return atoms


def qe_config_from_yaml(yaml_path: Path) -> QECalculatorConfig:
    """Load a qe_config.yaml file and return a QECalculatorConfig."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)

    pseudo_files = raw.pop("pseudo_files", {})
    # Convert list k-points to tuples
    for key in ("relax_kpoints", "relax_koffset", "scf_kpoints", "scf_koffset"):
        if key in raw:
            raw[key] = tuple(raw[key])

    return QECalculatorConfig(pseudo_files=pseudo_files, **raw)


def add_qe_arguments(parser) -> None:
    """Add QE calculator arguments to an argparse parser."""
    parser.add_argument(
        "--qe-config",
        required=True,
        help="Path to qe_config.yaml with QE parameters.",
    )
    parser.add_argument(
        "--nranks-per-disp",
        type=int,
        default=None,
        help="Override nranks_per_disp from config (GPUs per QE job).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Override max_concurrent from config (simultaneous QE jobs).",
    )


def qe_config_from_args(args) -> QECalculatorConfig:
    """Build QECalculatorConfig from parsed argparse arguments."""
    config = qe_config_from_yaml(args.qe_config)
    overrides: dict = {}
    if getattr(args, "nranks_per_disp", None) is not None:
        overrides["nranks_per_disp"] = args.nranks_per_disp
    if getattr(args, "max_concurrent", None) is not None:
        overrides["max_concurrent"] = args.max_concurrent
    if overrides:
        import dataclasses
        config = dataclasses.replace(config, **overrides)
    return config
