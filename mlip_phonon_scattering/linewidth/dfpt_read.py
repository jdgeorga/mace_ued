"""Read the custom XML force-constant dump produced by QE ``q2r.x``.

The dump handled here is XML, unlike the plain-text q2r output consumed by
``phonopy.interface.qe.PH_Q2R``.  Its force-constant ordering is nevertheless
the same, so this module deliberately reuses PH_Q2R's private geometry helpers
instead of maintaining a second implementation of that subtle permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from ase.data import atomic_masses, chemical_symbols
from phonopy.harmonic.force_constants import distribute_force_constants_by_translations
from phonopy.interface.qe import PH_Q2R, read_pwscf
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive, get_supercell

from mlip_phonon_scattering.phonopy_io import phonopy_atoms_to_ase


BOHR_TO_ANGSTROM = 0.529177210903


@dataclass
class DfptData:
    """Harmonic DFPT data in the force-constant order used by phono3py.

    ``fc2`` retains QE's native Ry/bohr**2 units.  ``nac['area']`` is in
    bohr**2 and ``nac['c']`` is the out-of-plane cell repeat in bohr (it is
    not a speed of light).
    """

    structure: object
    fc2: np.ndarray
    supercell_matrix: np.ndarray
    nac: dict


def _numbers(element: ET.Element | None, *, shape: tuple[int, ...]) -> np.ndarray:
    """Read whitespace-separated floats from an XML element."""

    if element is None or element.text is None:
        raise ValueError("Required value is absent from the q2r XML dump.")
    values = np.fromstring(element.text, sep=" ", dtype=float)
    if values.size != int(np.prod(shape)):
        raise ValueError(
            f"Malformed q2r XML value: expected {int(np.prod(shape))} numbers, "
            f"found {values.size}."
        )
    return values.reshape(shape)


def _mass_symbols(
    geometry: ET.Element, layer_symbols: list[list[str]] | None
) -> list[str]:
    """Identify XML atom species from the reliable type masses, never type names."""

    if layer_symbols is not None:
        candidates = list(dict.fromkeys(symbol for layer in layer_symbols for symbol in layer))
    else:
        candidates = [symbol for symbol in chemical_symbols[1:] if symbol]

    mass_by_type: dict[int, float] = {}
    for child in geometry:
        match = re.fullmatch(r"MASS\.(\d+)", child.tag)
        if match and child.text:
            mass_by_type[int(match.group(1))] = float(child.text)

    symbols: list[str] = []
    for atom in sorted(
        (child for child in geometry if re.fullmatch(r"ATOM\.\d+", child.tag)),
        key=lambda item: int(item.tag.split(".")[1]),
    ):
        try:
            mass = mass_by_type[int(atom.attrib["INDEX"])]
        except (KeyError, ValueError) as exc:
            raise ValueError("XML ATOM entry has no usable mass type INDEX.") from exc
        candidate_masses = np.array([atomic_masses[chemical_symbols.index(s)] for s in candidates])
        index = int(np.argmin(np.abs(candidate_masses - mass)))
        if abs(candidate_masses[index] - mass) > 0.5:
            raise ValueError(
                f"Could not identify element with XML mass {mass:.6g} amu "
                "within 0.5 amu."
            )
        symbols.append(candidates[index])
    return symbols


def _xml_structure(geometry: ET.Element, layer_symbols: list[list[str]] | None) -> PhonopyAtoms:
    """Best-effort structure fallback when the QE pwscf input is unavailable.

    This custom dump labels these fields as atomic units, but its values are
    numerically Angstrom-like in the production files.  We intentionally keep
    those values unchanged here; the preferred ``scf.in`` route avoids relying
    on this historically ambiguous header.
    """

    celldm = _numbers(geometry.find("CELL_DIMENSIONS"), shape=(6,))[0]
    lattice = _numbers(geometry.find("AT"), shape=(3, 3)) * celldm
    symbols = _mass_symbols(geometry, layer_symbols)
    atoms = sorted(
        (child for child in geometry if re.fullmatch(r"ATOM\.\d+", child.tag)),
        key=lambda item: int(item.tag.split(".")[1]),
    )
    tau = np.array(
        [np.fromstring(atom.attrib["TAU"], sep=" ", dtype=float) for atom in atoms]
    )
    # TAU is cartesian in units of the XML alat in this dump.
    scaled_positions = (tau * celldm) @ np.linalg.inv(lattice)
    return PhonopyAtoms(symbols=symbols, cell=lattice, scaled_positions=scaled_positions)


def _read_cell(
    dfpt_dir: Path, geometry: ET.Element, layer_symbols: list[list[str]] | None
) -> PhonopyAtoms:
    """Read the QE structure, with a documented XML-only fallback."""

    for filename in (dfpt_dir / "scf.in", dfpt_dir.parent / "scf.in"):
        if filename.is_file():
            cell, _ = read_pwscf(filename)
            return cell
    return _xml_structure(geometry, layer_symbols)


def atom_order_map(
    qe_positions: np.ndarray, phonopy_supercell: PhonopyAtoms, symprec: float = 1e-4
) -> np.ndarray:
    """Map phonopy-supercell atoms to QE-order fractional positions.

    ``qe_positions`` must be the q2r-order fractional positions (as returned by
    ``PH_Q2R._get_q2r_positions``).  A failed one-to-one match raises a clear
    error rather than silently applying a wrong FC2 permutation.
    """

    qe_positions = np.asarray(qe_positions, dtype=float)
    spos = np.asarray(phonopy_supercell.scaled_positions, dtype=float)
    if qe_positions.shape != spos.shape:
        raise ValueError(
            "atom-order mismatch — check symprec / structure alignment: "
            f"position shapes differ ({qe_positions.shape} != {spos.shape})."
        )

    mapping: list[int] = []
    for position in spos:
        diff = qe_positions - position
        diff -= np.rint(diff)
        distances = np.linalg.norm(diff @ phonopy_supercell.cell, axis=1)
        matches = np.flatnonzero(distances < symprec)
        if matches.size != 1:
            raise ValueError(
                "atom-order mismatch — check symprec / structure alignment: "
                f"found {matches.size} matches for one supercell site."
            )
        mapping.append(int(matches[0]))
    result = np.array(mapping, dtype=np.intp)
    if np.unique(result).size != result.size:
        raise ValueError(
            "atom-order mismatch — check symprec / structure alignment: "
            "mapping is not a bijection."
        )
    return result


def check_d3_and_acoustic(ph_out_path: str | Path, gamma_freqs: np.ndarray) -> dict:
    """Check D3 evidence and the three acoustic Gamma frequencies.

    Returns ``d3_detected``, ``acoustic_min_thz``, ``acoustic_ok``, and
    ``ph_out_path``.  The combined failure is the relevant guard: a missing D3
    read line is only fatal when it accompanies an unstable acoustic mode.
    """

    path = Path(ph_out_path)
    text = path.read_text(errors="replace")
    d3_detected = bool(re.search(r"(?:grimme[ -]?d3|dft[ -]?d3)", text, re.I))
    frequencies = np.sort(np.asarray(gamma_freqs, dtype=float).ravel())
    if frequencies.size < 3:
        raise ValueError("gamma_freqs must contain at least the three acoustic modes.")
    acoustic_min = float(np.min(frequencies[:3]))
    acoustic_ok = acoustic_min >= -0.05
    result = {
        "d3_detected": d3_detected,
        "acoustic_min_thz": acoustic_min,
        "acoustic_ok": acoustic_ok,
        "ph_out_path": str(path),
    }
    if not d3_detected and not acoustic_ok:
        raise RuntimeError(
            "No Grimme-D3/DFT-D3 evidence was found and Gamma has an unstable "
            f"acoustic mode ({acoustic_min:.6g} THz): possible vdW-clobbering run."
        )
    return result


def resolve_dfpt_log_path(dfpt_dir: str | Path) -> Path | None:
    """Return the first available q1 ph.x log using production-compatible names.

    Older runs conventionally wrote ``ph.out`` while the production DFPT data
    writes ``out.log``.  Preserve that precedence, then accept other matching
    output/log names for compatible datasets.
    """

    q1_dir = Path(dfpt_dir) / "q1"
    for candidate in (q1_dir / "ph.out", q1_dir / "out.log"):
        if candidate.is_file():
            return candidate
    for pattern in ("*.out", "*.log"):
        matches = sorted(q1_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_dfpt(
    dfpt_dir: str | Path,
    layer_symbols: list[list[str]] | None = None,
    atom_types: np.ndarray | None = None,
    layer_ids: np.ndarray | None = None,
    symprec: float = 1e-4,
) -> DfptData:
    """Read XML q2r FC2, QE structure, and 2D NAC inputs from ``dfpt_dir``."""

    dfpt_dir = Path(dfpt_dir)
    xml_path = dfpt_dir / "collect" / "ifc.q2r.xml"
    root = ET.parse(xml_path).getroot()
    geometry = root.find("GEOMETRY_INFO")
    if geometry is None:
        raise ValueError("q2r XML is missing GEOMETRY_INFO.")
    ifc = root.find("INTERATOMIC_FORCE_CONSTANTS")
    dielectric_properties = root.find("DIELECTRIC_PROPERTIES")
    if ifc is None or dielectric_properties is None:
        raise ValueError("q2r XML is missing force constants or dielectric properties.")

    natom = int(_numbers(geometry.find("NUMBER_OF_ATOMS"), shape=(1,))[0])
    dim = _numbers(ifc.find("MESH_NQ1_NQ2_NQ3"), shape=(3,)).astype("int64")
    ndim = int(np.prod(dim))
    alpha_ewald = float(_numbers(ifc.find("alpha_ewald"), shape=(1,))[0])
    dielectric = _numbers(dielectric_properties.find("EPSILON"), shape=(3, 3))
    zstar = dielectric_properties.find("ZSTAR")
    if zstar is None:
        raise ValueError("q2r XML is missing ZSTAR Born effective charges.")
    born = np.array(
        [_numbers(zstar.find(f"Z_AT_.{i}"), shape=(3, 3)) for i in range(1, natom + 1)]
    )
    volume = float(_numbers(geometry.find("UNIT_CELL_VOLUME_AU"), shape=(1,))[0])

    cell = _read_cell(dfpt_dir, geometry, layer_symbols)
    if len(cell) != natom:
        raise ValueError(f"QE structure has {len(cell)} atoms, XML has {natom}.")

    if layer_symbols is not None:
        flattened = [symbol for layer in layer_symbols for symbol in layer]
        if flattened != list(cell.symbols):
            raise ValueError(
                "layer_symbols do not match the QE structure atom order: "
                f"expected {flattened}, found {list(cell.symbols)}."
            )
        atom_types_array = np.arange(natom, dtype=int)
        layer_ids_array = np.concatenate(
            [np.full(len(layer), layer_index, dtype=int) for layer_index, layer in enumerate(layer_symbols)]
        )
    else:
        if atom_types is None or layer_ids is None:
            raise ValueError("Provide layer_symbols or both atom_types and layer_ids.")
        atom_types_array = np.asarray(atom_types, dtype=int)
        layer_ids_array = np.asarray(layer_ids, dtype=int)
        if atom_types_array.shape != (natom,) or layer_ids_array.shape != (natom,):
            raise ValueError("atom_types and layer_ids must each have length natom.")

    # Use PH_Q2R only for its exact q2r/phonopy geometry permutation logic.
    q2r = PH_Q2R.__new__(PH_Q2R)
    q2r.dimension = dim
    q2r._symprec = symprec
    q2r_spos = q2r._get_q2r_positions(cell)
    scell = get_supercell(cell, np.diag(dim))
    pcell = get_primitive(scell, np.diag(1.0 / dim))
    site_map = q2r._get_site_mapping(scell.scaled_positions, q2r_spos, scell.cell)

    q2r_fc = np.zeros((natom, natom * ndim, 3, 3), dtype="double", order="C")
    trans = [translation[::-1] for translation in np.ndindex(tuple(dim[::-1]))]
    tag_pattern = re.compile(r"s_s1_m1_m2_m3\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(\d+)")
    found = 0
    for entry in ifc:
        match = tag_pattern.fullmatch(entry.tag)
        if match is None:
            continue
        s, s1, m1, m2, m3 = (int(value) for value in match.groups())
        i_dim = trans.index((m1 - 1, m2 - 1, m3 - 1))
        block = _numbers(entry.find("IFC"), shape=(3, 3))
        q2r_fc[s1 - 1, (s - 1) * ndim + i_dim] = block.T
        found += 1
    expected = natom * natom * ndim
    if found != expected:
        raise ValueError(f"Expected {expected} IFC blocks in q2r XML, found {found}.")

    natom_s = len(pcell) * ndim
    fc2 = np.zeros((natom_s, natom_s, 3, 3), dtype="double", order="C")
    fc2[pcell.p2s_map, :] = q2r_fc[:, site_map]
    distribute_force_constants_by_translations(fc2, pcell)

    # read_pwscf returns bohr, while ASE interprets its input cell/positions as
    # Angstrom.  Keep ``pcell`` in bohr above for PH_Q2R's site mapping and
    # convert only this final, returned MLIP structure.
    structure = phonopy_atoms_to_ase(pcell, include_masses=True)
    structure.set_cell(structure.cell.array * BOHR_TO_ANGSTROM, scale_atoms=True)
    structure.arrays["atom_types"] = atom_types_array
    structure.arrays["layer_ids"] = layer_ids_array
    area = float(np.linalg.norm(np.cross(pcell.cell[0], pcell.cell[1])))
    # QE XML quantities below remain in atomic units: area is bohr**2 and c is
    # the vacuum-padded out-of-plane cell repeat in bohr, not a speed of light.
    nac = {
        "born": born,
        "dielectric": dielectric,
        "alpha_ewald": alpha_ewald,
        "area": area,
        "c": volume / area,
        "periodic_axes": (0, 1),
        "factor": 2.0,
    }
    return DfptData(
        structure=structure,
        fc2=fc2,
        supercell_matrix=np.diag(dim),
        nac=nac,
    )


QE_FREQUENCY_FACTOR_THZ = 108.97077184367376
"""QE Ry/bohr**2/amu to THz conversion (not phonopy's VaspToTHz)."""


def _acoustic_sum_rule_fixup(fc2: np.ndarray) -> np.ndarray:
    """Return a working FC2 copy with a one-shot row-sum ASR correction.

    This diagnostic/reference-only approximation must not be used to mutate the
    raw FC2 stored in :class:`DfptData`. QE's ``matdyn.x`` applies its own more
    careful symmetry-preserving ASR projection when diagonalizing.
    """

    fixed = fc2.astype(np.float64, copy=True)
    idx = np.arange(fixed.shape[0])
    fixed[idx, idx] -= fixed.sum(axis=1)
    return fixed


def dynamical_matrix_from_dfpt(data: "DfptData"):
    """Build a plain, short-range-only phonopy dynamical matrix from DFPT data.

    The primitive reconstruction follows :func:`read_dfpt` so its atom order
    matches the returned primitive ASE structure and, consequently, the FC2
    row/column order. A defensive comparison raises rather than silently using
    mismatched force constants if phonopy's construction convention changes.
    """

    from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
    from phonopy.structure.cells import get_primitive, get_supercell

    from mlip_phonon_scattering.phonopy_io import ase_to_phonopy_atoms

    unitcell = ase_to_phonopy_atoms(data.structure, include_masses=True)
    scell = get_supercell(unitcell, data.supercell_matrix)
    pcell = get_primitive(scell, np.linalg.inv(data.supercell_matrix))

    delta = pcell.scaled_positions - unitcell.scaled_positions
    delta -= np.rint(delta)
    order_ok = list(pcell.symbols) == list(unitcell.symbols) and np.abs(
        delta @ pcell.cell
    ).max() < 1e-6
    if not order_ok:
        raise RuntimeError(
            "Reconstructed primitive-cell atom order does not match "
            "DfptData.structure; refusing to build a dynamical matrix that "
            "would silently mis-index fc2."
        )

    return get_dynamical_matrix(
        _acoustic_sum_rule_fixup(data.fc2), scell, pcell, nac_params=None
    )


def frequencies_and_eigenvectors_at_q(dm, q):
    """Diagonalize ``dm`` at crystal ``q`` and return THz frequencies and modes."""

    dm.run(np.asarray(q, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(dm.dynamical_matrix)
    freq = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * QE_FREQUENCY_FACTOR_THZ
    nat = eigvecs.shape[0] // 3
    return freq, eigvecs.T.reshape(-1, nat, 3)


def main(argv=None):
    """Write reusable structure, raw FC2, and 2D-NAC artifacts from QE DFPT."""

    import argparse
    import ast

    from ase.io import write as ase_write

    parser = argparse.ArgumentParser(
        description="Read DFPT structure/fc2/NAC and write mlip-linewidth artifacts."
    )
    parser.add_argument("--dfpt-dir", required=True)
    parser.add_argument(
        "--layer-symbols",
        required=True,
        help="Python literal, e.g. \"[['Mo','Se','Se'],['W','Se','Se']]\"",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--symprec", type=float, default=1e-4)
    args = parser.parse_args(argv)

    data = read_dfpt(
        args.dfpt_dir,
        layer_symbols=ast.literal_eval(args.layer_symbols),
        symprec=args.symprec,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structure_path = out_dir / "dfpt_structure.xyz"
    fc2_path = out_dir / "dfpt_fc2.npy"
    nac_path = out_dir / "nac_2d.npz"
    ase_write(str(structure_path), data.structure, format="extxyz")
    np.save(fc2_path, data.fc2)
    np.savez(nac_path, **data.nac)

    print(f"wrote {structure_path}")
    print(f"wrote {fc2_path}  shape={data.fc2.shape}")
    print(f"wrote {nac_path}  keys={sorted(data.nac)}")

    ph_out_path = resolve_dfpt_log_path(args.dfpt_dir)
    if ph_out_path is not None:
        dm = dynamical_matrix_from_dfpt(data)
        freq_gamma, _ = frequencies_and_eigenvectors_at_q(dm, [0.0, 0.0, 0.0])
        print(f"gate results: {check_d3_and_acoustic(ph_out_path, freq_gamma)}")
    else:
        print(f"gate check skipped: {Path(args.dfpt_dir) / 'q1' / 'ph.out'} not found")


if __name__ == "__main__":
    main()
