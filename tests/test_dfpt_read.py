import numpy as np
import pytest
from mlip_phonon_scattering.linewidth.dfpt_read import read_dfpt

DFPT = "/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix"


@pytest.mark.slow
def test_read_dfpt_fc2_and_mapping():
    d = read_dfpt(DFPT, layer_symbols=[['Mo', 'Se', 'Se'], ['W', 'Se', 'Se']])
    N = 6 * 6 * 1 * 6  # 6x6x1 supercell of a 6-atom cell
    assert d.fc2.shape == (N, N, 3, 3)
    assert list(np.diag(d.supercell_matrix)) == [6, 6, 1]
    assert set(np.unique(d.structure.arrays['layer_ids'])) == {0, 1}
    assert d.structure.arrays['atom_types'].shape[0] == d.structure.get_global_number_of_atoms()


@pytest.mark.slow
def test_read_dfpt_structure_is_in_angstrom():
    d = read_dfpt(DFPT, layer_symbols=[['Mo', 'Se', 'Se'], ['W', 'Se', 'Se']])
    lengths = np.linalg.norm(d.structure.cell.array, axis=1)
    assert np.isclose(lengths[0], 3.295, atol=0.02)
    assert np.isclose(lengths[2], 40.0, atol=0.5)


@pytest.mark.slow
def test_read_dfpt_nac_values():
    d = read_dfpt(DFPT, layer_symbols=[['Mo', 'Se', 'Se'], ['W', 'Se', 'Se']])
    eps = d.nac['dielectric']
    assert np.allclose(np.diag(eps), [5.8788, 5.8788, 1.2818], atol=1e-3)
    assert d.nac['born'].shape == (6, 3, 3)
    assert np.isclose(d.nac['born'][0, 0, 0], -1.77642705, atol=1e-5)
    assert np.isclose(d.nac['alpha_ewald'], 0.98211261577626741, atol=1e-9)
    assert d.nac['periodic_axes'] == (0, 1)
    assert np.isclose(d.nac['factor'], 2.0)
