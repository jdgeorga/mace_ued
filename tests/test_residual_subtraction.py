import numpy as np
from mlip_phonon_scattering.linewidth.forces3 import subtract_reference_forces


def test_subtract_reference():
    F = np.random.default_rng(0).standard_normal((5, 6, 3))
    F0 = F.mean(axis=0)                         # pretend nonzero reference
    Fc = subtract_reference_forces(F, F0)
    assert np.allclose(Fc, F - F0[None])
    # a set whose every frame equals F0 -> all zero
    assert np.allclose(subtract_reference_forces(np.tile(F0, (3, 1, 1)), F0), 0.0)
