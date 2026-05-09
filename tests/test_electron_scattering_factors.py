from __future__ import annotations

import numpy as np

from mlip_phonon_scattering.electron_scattering_factors import PENG_NEUTRAL_HIGH_S, PENG_NEUTRAL_LOW_S
from mlip_phonon_scattering.ued_intensity import elastic_electron_atomic_scattering_factors


def test_existing_high_s_entries_match_lammps_peng_table() -> None:
    expected = {
        "Mo": ([0.3069, 1.1714, 3.2293, 3.4254, 2.1224], [0.1101, 1.0222, 5.9613, 25.1965, 93.5831]),
        "W": ([0.3661, 1.6191, 3.2455, 4.0856, 3.2064], [0.0761, 0.7543, 4.0952, 18.2886, 68.0967]),
        "S": ([0.0915, 0.4312, 1.0847, 2.4671, 1.0852], [0.0838, 0.7788, 4.3462, 15.5846, 44.6365]),
        "Se": ([0.1574, 0.7614, 1.4834, 3.0016, 1.7978], [0.0686, 0.6808, 3.1163, 14.3458, 44.0455]),
    }
    for symbol, coeffs in expected.items():
        assert np.allclose(PENG_NEUTRAL_HIGH_S[symbol][0], coeffs[0])
        assert np.allclose(PENG_NEUTRAL_HIGH_S[symbol][1], coeffs[1])


def test_peng_model_selects_low_and_high_s_blocks() -> None:
    q_norm = np.array([4.0 * np.pi * 1.0, 4.0 * np.pi * 3.0])
    values = elastic_electron_atomic_scattering_factors(q_norm, ["Si"], model="peng")[:, 0]

    low_a, low_b = PENG_NEUTRAL_LOW_S["Si"]
    high_a, high_b = PENG_NEUTRAL_HIGH_S["Si"]
    expected_low = np.sum(np.asarray(low_a) * np.exp(-np.asarray(low_b) * 1.0**2))
    expected_high = np.sum(np.asarray(high_a) * np.exp(-np.asarray(high_b) * 3.0**2))

    assert np.allclose(values, [expected_low, expected_high])
