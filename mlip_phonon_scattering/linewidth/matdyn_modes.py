"""Small QE ``matdyn.x`` bridge used by the linewidth comparison workflow.

The q coordinates emitted by ``matdyn.modes``/``phband.freq`` are in matdyn's
internal basis.  Consequently, q-point identity is deliberately carried by
input order: crystallographic coordinates are written to and recovered from
``matdyn.in`` only.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Sequence

import numpy as np


_FLOAT = r"[-+0-9.eE]+"
_FREQ_RE = re.compile(
    rf"freq\s*\(\s*\d+\s*\)\s*=\s*({_FLOAT})\s*\[THz\]\s*=\s*"
    rf"({_FLOAT})\s*\[cm-1\]",
    re.IGNORECASE,
)
_VECTOR_RE = re.compile(
    rf"\(\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+"
    rf"({_FLOAT})\s+({_FLOAT})\s*\)"
)


def matdyn_qpoints_for_mesh(
    phono3py_yaml: str, mesh: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full BZ-grid addresses and crystallographic q coordinates."""
    import phono3py

    ph3 = phono3py.load(phono3py_yaml, log_level=0)
    ph3.mesh_numbers = mesh
    grid = ph3.grid
    if grid is None:  # Defensive: mesh_numbers should always initialise this.
        raise RuntimeError("phono3py did not initialise its BZ grid")
    grid_address = grid.addresses.copy()
    q_cryst = grid_address.astype(float) @ grid.QDinv.T
    return grid_address, q_cryst


def write_matdyn_input(
    ifc_xml: str, qlist: np.ndarray, loto_2d: bool, workdir
) -> Path:
    """Write ``matdyn.in`` for an explicit crystal-coordinate q-point list."""
    path = Path(workdir) / "matdyn.in"
    qpoints = np.asarray(qlist, dtype=float)
    if qpoints.ndim != 2 or qpoints.shape[1] != 3:
        raise ValueError("qlist must have shape (nq, 3)")
    nq = len(qpoints)
    weight = 1.0 / nq if nq else 0.0
    loto = ".true." if loto_2d else ".false."
    lines = [
        "&input",
        "  asr = 'crystal'",
        f"  flfrc = '{ifc_xml}'",
        "  flfrq = 'phband.freq'",
        "  flvec = 'matdyn.modes'",
        "  q_in_band_form = .false.",
        "  q_in_cryst_coord = .true.",
        f"  loto_2d = {loto}",
        "/",
        f"{nq} crystal",
    ]
    lines.extend(f"  {q[0]: .12f}  {q[1]: .12f}  {q[2]: .12f}  {weight:.12f}" for q in qpoints)
    path.write_text("\n".join(lines) + "\n")
    return path


def run_matdyn(ifc_xml: str, qlist: np.ndarray, loto_2d: bool, workdir) -> str:
    """Run serial ``matdyn.x`` and return the produced eigenvector-file path."""
    directory = Path(workdir)
    input_path = write_matdyn_input(ifc_xml, qlist, loto_2d, directory)
    output_path = directory / "matdyn.out"
    completed = subprocess.run(
        ["matdyn.x", "-in", str(input_path)],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_path.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"matdyn.x failed with exit code {completed.returncode}:\n{completed.stdout}"
        )
    return str(directory / "matdyn.modes")


def parse_matdyn_modes(flfrq, flvec, masses, grid_address=None) -> dict:
    """Parse QE mode vectors, retaining QE/phonopy mass-weighted convention.

    ``masses`` is accepted for call-site parity only.  QE has already applied
    the mass weighting in this output, so no mass rescaling is performed.
    """
    text = Path(flvec).read_text()
    blocks = re.split(r"^\s*q\s*=.*$", text, flags=re.MULTILINE)[1:]
    if not blocks:
        raise ValueError(f"no q-point blocks found in {flvec}")

    all_freq, all_cm, all_vec = [], [], []
    nat = None
    for iblock, block in enumerate(blocks):
        frequency_matches = _FREQ_RE.findall(block)
        if not frequency_matches:
            raise ValueError(f"no frequencies found in q-point block {iblock}")
        thz = np.asarray([float(pair[0]) for pair in frequency_matches], dtype=float)
        cm = np.asarray([float(pair[1]) for pair in frequency_matches], dtype=float)
        nb = len(thz)
        if nb % 3:
            raise ValueError(f"mode count {nb} is not divisible by three in block {iblock}")
        block_nat = nb // 3
        if nat is None:
            nat = block_nat
        elif nat != block_nat:
            raise ValueError("inconsistent atom count across q-point blocks")
        vector_matches = _VECTOR_RE.findall(block)
        if len(vector_matches) != nb * block_nat:
            raise ValueError(
                f"expected {nb * block_nat} atom vectors in block {iblock}, "
                f"found {len(vector_matches)}"
            )
        raw = np.asarray(vector_matches, dtype=float).reshape(nb, block_nat, 6)
        vec = raw[..., 0::2] + 1j * raw[..., 1::2]
        norms = np.linalg.norm(vec.reshape(nb, -1), axis=1)
        if np.any(norms == 0):
            raise ValueError(f"zero-norm eigenvector in q-point block {iblock}")
        all_freq.append(thz)
        all_cm.append(cm)
        all_vec.append(vec / norms[:, None, None])

    assert nat is not None
    if masses is not None and len(masses) != nat:
        raise ValueError(f"masses has length {len(masses)}, expected {nat}")
    frequencies = np.asarray(all_freq, dtype=np.float64)
    result = {
        "frequencies": frequencies,
        "frequencies_cm": load_phband_freq(flfrq) if flfrq is not None else np.asarray(all_cm, dtype=np.float64),
        "eigenvectors": np.asarray(all_vec, dtype=np.complex128),
    }
    if result["frequencies_cm"].shape != frequencies.shape:
        raise ValueError("frequency-file q-point/mode dimensions do not match matdyn.modes")
    if grid_address is not None:
        result["grid_address"] = grid_address
    return result


def load_phband_freq(path) -> np.ndarray:
    """Load the deterministic QE ``&plot`` frequency file (in cm^-1)."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError("empty phband frequency file")
    header = re.search(r"nbnd\s*=\s*(\d+)\s*,\s*nks\s*=\s*(\d+)", lines[0], re.I)
    if header is None:
        raise ValueError("could not parse nbnd/nks from &plot header")
    nbnd, nks = map(int, header.groups())
    cursor = 1
    rows = []
    for iq in range(nks):
        if cursor >= len(lines):
            raise ValueError(f"missing q-vector line for q-point {iq}")
        qfields = lines[cursor].split()
        cursor += 1
        if len(qfields) != 4:
            raise ValueError(f"q-vector line for q-point {iq} does not contain four fields")
        try:
            [float(value) for value in qfields]
        except ValueError as exc:
            raise ValueError(f"invalid q-vector line for q-point {iq}") from exc
        values: list[float] = []
        while len(values) < nbnd:
            if cursor >= len(lines):
                raise ValueError(f"truncated frequencies for q-point {iq}")
            fields = lines[cursor].split()
            cursor += 1
            try:
                values.extend(float(value) for value in fields)
            except ValueError as exc:
                raise ValueError(f"invalid frequency row for q-point {iq}") from exc
        if len(values) != nbnd:
            raise ValueError(f"frequency row for q-point {iq} has {len(values)} values, expected {nbnd}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def load_matdyn_path(matdyn_in_path) -> np.ndarray:
    """Read the explicit crystal-coordinate q-point list from ``matdyn.in``."""
    lines = Path(matdyn_in_path).read_text().splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip() == "/"), None)
    if end is None:
        raise ValueError("matdyn input has no terminating namelist slash")
    cursor = end + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("matdyn input has no q-point count")
    count_fields = lines[cursor].split()
    try:
        nq = int(count_fields[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("invalid q-point count line") from exc
    cursor += 1
    points = []
    for iq in range(nq):
        if cursor >= len(lines):
            raise ValueError(f"matdyn input ends before q-point {iq}")
        fields = lines[cursor].split()
        cursor += 1
        if len(fields) < 3:
            raise ValueError(f"q-point {iq} has fewer than three coordinates")
        try:
            points.append([float(value) for value in fields[:3]])
        except ValueError as exc:
            raise ValueError(f"invalid q-point {iq}") from exc
    return np.asarray(points, dtype=np.float64)


def main(argv=None):
    """Run matdyn on a phono3py grid, gauge-align its modes, and save them."""

    import argparse

    from ase.io import read as ase_read

    from mlip_phonon_scattering.linewidth import dfpt_read as _dfpt_read
    from mlip_phonon_scattering.linewidth.dfpt_read import (
        dynamical_matrix_from_dfpt,
        frequencies_and_eigenvectors_at_q,
    )
    from mlip_phonon_scattering.linewidth.gauge import (
        apply_selected_gauge,
        select_gauge,
    )

    parser = argparse.ArgumentParser(
        description="Run matdyn on phono3py's BZ grid; gauge-correct and write modes."
    )
    parser.add_argument("--phono3py-yaml", required=True)
    parser.add_argument("--ifc-xml", required=True)
    parser.add_argument(
        "--mesh", nargs=3, type=int, required=True, metavar=("MX", "MY", "MZ")
    )
    parser.add_argument("--loto-2d", choices=["on", "off"], required=True)
    parser.add_argument("--dfpt-fc2", required=True)
    parser.add_argument("--dfpt-structure", required=True)
    parser.add_argument(
        "--dfpt-supercell-matrix",
        nargs=3,
        type=int,
        default=[6, 6, 1],
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    loto_2d = args.loto_2d == "on"
    mesh = list(args.mesh)
    grid_address, q_cryst = matdyn_qpoints_for_mesh(args.phono3py_yaml, mesh)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    flvec = run_matdyn(args.ifc_xml, q_cryst, loto_2d, workdir)
    flfrq = workdir / "phband.freq"
    structure = ase_read(args.dfpt_structure)
    modes = parse_matdyn_modes(
        flfrq, flvec, structure.get_masses(), grid_address=grid_address
    )

    e_qe = modes["eigenvectors"]
    freqs = modes["frequencies"]
    nb = e_qe.shape[1]
    nat = e_qe.shape[2]
    if nat * 3 != nb:
        raise ValueError(f"unexpected band/atom count: nb={nb}, nat={nat}")

    data = _dfpt_read.DfptData(
        structure=structure,
        fc2=np.load(args.dfpt_fc2),
        supercell_matrix=np.diag(args.dfpt_supercell_matrix),
        nac={},
    )
    dm = dynamical_matrix_from_dfpt(data)
    e_ph = np.empty_like(e_qe)
    freq_ph = np.empty_like(freqs)
    for iq, q in enumerate(q_cryst):
        freq_ph[iq], e_ph[iq] = frequencies_and_eigenvectors_at_q(dm, q)

    transform, residual, qflip_perm = select_gauge(
        e_ph, e_qe, q_cryst, structure.get_scaled_positions(), freqs=freq_ph
    )
    print(f"gauge transform: {transform}  residual={residual:.4g}")
    gauged = apply_selected_gauge(
        e_qe, q_cryst, structure.get_scaled_positions(), transform, qflip_perm
    )

    # Store primary layout; a later Phono3py consumer converts it to (Nq, nat*3, mode).
    np.savez(
        out_path,
        frequencies=freqs,
        eigenvectors=gauged,
        grid_address=grid_address,
        mesh=np.asarray(mesh),
    )
    print(
        f"wrote {out_path}: Ngrid={len(grid_address)} nb={nb} nat={nat} "
        f"loto_2d={loto_2d}"
    )


if __name__ == "__main__":
    main()
