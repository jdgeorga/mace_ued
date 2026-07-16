"""Regression coverage for force-constant cache source selection."""

import numpy as np

from mlip_phonon_scattering.linewidth.gpu_scattering_W_phonons_bilayer_comm_mesh import (
    set_phono3py_forces_with_mesh_fc_cache,
)


class _SelfComm:
    def Get_rank(self):
        return 0

    def Barrier(self):
        pass


class _FakePhono3py:
    def produce_fc3(self, **_kwargs):
        self.fc3 = np.array([33.0])

    def produce_fc2(self, **_kwargs):  # pragma: no cover - must not be called here.
        raise AssertionError("DFPT FC2 source must bypass produce_fc2")


def test_dfpt_fc2_bypasses_stale_legacy_cache(tmp_path):
    """A supplied DFPT FC2 wins even when a legacy cache happens to exist."""
    forces2 = tmp_path / "forces2.npy"
    forces3 = tmp_path / "forces3.npy"
    dfpt_fc2 = tmp_path / "dfpt_fc2.npy"
    np.save(forces2, np.zeros((1, 1, 3)))
    np.save(forces3, np.zeros((1, 1, 3)))
    np.save(dfpt_fc2, np.array([42.0]))

    legacy = tmp_path / "phonon_cache"
    legacy.mkdir()
    np.save(legacy / "fc2.npy", np.array([999.0]))
    np.save(legacy / "fc3.npy", np.array([998.0]))

    mesh_cache = tmp_path / "mesh_cache"
    ph3 = _FakePhono3py()
    set_phono3py_forces_with_mesh_fc_cache(
        ph3,
        str(forces2),
        str(forces3),
        str(mesh_cache),
        legacy_cache_dir=str(legacy),
        comm=_SelfComm(),
        dfpt_fc2=str(dfpt_fc2),
    )

    assert np.array_equal(ph3.fc2, [42.0])
    assert np.array_equal(ph3.fc3, [33.0])
    assert np.array_equal(np.load(mesh_cache / "fc2.npy"), [42.0])
    assert np.array_equal(np.load(mesh_cache / "fc3.npy"), [33.0])
