import numpy as np

from mlip_phonon_scattering.linewidth.gauge import (
    apply_bloch_gauge,
    apply_bloch_gauge_flat,
    overlap,
    select_gauge,
    subspace_overlap,
)


def test_gauge_phase_and_roundtrip():
    rng = np.random.default_rng(0)
    Nq, nb, nat = 4, 6, 2
    e = rng.standard_normal((Nq, nb, nat, 3)) + 1j*rng.standard_normal((Nq, nb, nat, 3))
    q = rng.standard_normal((Nq, 3)); tau = rng.standard_normal((nat, 3))
    e1 = apply_bloch_gauge(e, q, tau, sign=+1)
    phase = np.exp(1j*2*np.pi*(q @ tau.T))            # (Nq,nat)
    assert np.allclose(e1, e * phase[:, None, :, None])
    # +then- restores
    assert np.allclose(apply_bloch_gauge(e1, q, tau, sign=-1), e)


def test_overlap_identity():
    rng = np.random.default_rng(1); e = rng.standard_normal((3, 4, 2, 3)) + 0j
    ov = overlap(e, e)
    assert np.allclose(ov, 1.0, atol=1e-12)


def test_flat_gauge_matches_primary_layout():
    rng = np.random.default_rng(2)
    nq, nb, nat = 3, 5, 2
    e = rng.standard_normal((nq, nb, nat, 3)) + 1j * rng.standard_normal((nq, nb, nat, 3))
    q = rng.standard_normal((nq, 3))
    tau = rng.standard_normal((nat, 3))
    flat = e.reshape(nq, nb, nat * 3).transpose(0, 2, 1)
    expected = apply_bloch_gauge(e, q, tau, sign=-1).reshape(nq, nb, nat * 3).transpose(0, 2, 1)
    assert np.allclose(apply_bloch_gauge_flat(flat, q, tau, sign=-1), expected)


def test_subspace_overlap_identity_for_orthonormal_modes():
    nq, nb, nat = 2, 4, 2
    e = np.zeros((nq, nb, nat, 3), dtype=complex)
    for iq in range(nq):
        for mode in range(nb):
            e[iq, mode].reshape(-1)[mode] = 1.0
    assert np.allclose(subspace_overlap(e, e), 1.0, atol=1e-12)


def test_select_gauge_recovers_inverse_phase():
    rng = np.random.default_rng(3)
    e_ph = rng.standard_normal((3, 4, 2, 3)) + 1j * rng.standard_normal((3, 4, 2, 3))
    # select_gauge scores degenerate subspaces, whose eigenvectors are
    # orthonormal in physical phonon data.
    e_ph = np.asarray([np.linalg.qr(block.reshape(4, 6).T)[0].T.reshape(4, 2, 3) for block in e_ph])
    q = rng.standard_normal((3, 3))
    tau = rng.standard_normal((2, 3))
    e_qe = apply_bloch_gauge(e_ph, q, tau, sign=+1)
    transform, residual, qflip_perm = select_gauge(e_ph, e_qe, q, tau)
    assert transform.sign == -1
    assert not transform.conjugate
    assert not transform.qflip
    assert qflip_perm is None
    assert residual < 1e-12
