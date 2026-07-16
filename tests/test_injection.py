"""CPU-only tests for DFPT phonon-mode injection layout and bookkeeping."""

import numpy as np
import phono3py
from phonopy.structure.atoms import PhonopyAtoms


def test_reshape_transpose_preserves_norm_and_consistency():
    """Flattening atom/cartesian components preserves each mode norm exactly."""
    rng = np.random.default_rng(1234)
    ngrid, nb, nat = 5, 6, 2
    e = rng.normal(size=(ngrid, nb, nat, 3)) + 1j * rng.normal(size=(ngrid, nb, nat, 3))
    flat = e.reshape(ngrid, nb, nat * 3).transpose(0, 2, 1)

    for mode in range(nb):
        assert np.allclose(
            np.linalg.norm(flat[:, :, mode], axis=-1),
            np.linalg.norm(e[:, mode], axis=(-1, -2)),
        )
    assert np.allclose(flat.transpose(0, 2, 1).reshape(ngrid, nb, nat, 3), e)


def _toy_phono3py_with_force_constants():
    """Construct a tiny Phono3py model with force constants from synthetic forces."""
    cell = PhonopyAtoms(
        symbols=["Si", "Si"],
        cell=np.diag([5.0, 5.0, 12.0]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.0]],
        masses=[28.0855, 28.0855],
    )
    ph3 = phono3py.Phono3py(
        cell,
        supercell_matrix=[2, 2, 1],
        phonon_supercell_matrix=[2, 2, 1],
        log_level=0,
    )
    ph3.generate_displacements()
    rng = np.random.default_rng(5678)
    nsc = len(ph3.supercell)
    ph3.forces = rng.normal(scale=1e-4, size=(len(ph3.supercells_with_displacements), nsc, 3))
    ph3.phonon_forces = rng.normal(
        scale=1e-4,
        size=(len(ph3.phonon_supercells_with_displacements), nsc, 3),
    )
    ph3.produce_fc3(symmetrize_fc3r=True)
    ph3.produce_fc2(symmetrize_fc2=True)
    return ph3


def test_injection_sets_phonon_data_and_is_not_overwritten():
    """Injected full-mesh modes set phonon_done and survive a solver call."""
    ph3 = _toy_phono3py_with_force_constants()
    ph3.mesh_numbers = [4, 4, 1]
    ph3.init_phph_interaction(nac_q_direction=None)

    ngrid = len(ph3.grid.addresses)
    nb = 3 * len(ph3.primitive)
    frequencies = np.arange(ngrid * nb, dtype=np.float64).reshape(ngrid, nb) / 100.0
    eigenvectors = np.tile(np.eye(nb, dtype=np.complex128), (ngrid, 1, 1))
    grid_address = np.array(ph3.grid.addresses, dtype=np.int64, copy=True)

    ph3.set_phonon_data(frequencies, eigenvectors, grid_address)
    assert np.all(ph3.phph_interaction.get_phonons()[2] == 1)
    assert np.allclose(ph3.phph_interaction.get_phonons()[0], frequencies)

    ph3.phph_interaction.run_phonon_solver()
    assert np.allclose(ph3.phph_interaction.get_phonons()[0], frequencies)


def test_reshape_matches_documented_flat_layout():
    """Injection stores each original mode as a column in the flattened layout."""
    rng = np.random.default_rng(9012)
    ngrid, nb, nat = 4, 6, 2
    e = rng.normal(size=(ngrid, nb, nat, 3)) + 1j * rng.normal(size=(ngrid, nb, nat, 3))
    flat = e.reshape(ngrid, nb, nat * 3).transpose(0, 2, 1)

    assert flat.shape == (ngrid, nat * 3, nb)
    for grid, mode in [(0, 0), (1, 4), (3, 5)]:
        assert np.allclose(flat[grid, :, mode], e[grid, mode].reshape(nat * 3))
