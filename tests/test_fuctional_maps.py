import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from shape_utils.config import FunctionalMapConfig
from shape_utils.functional_maps import (
    calculate_functional_maps,
    calculate_functional_maps_chem,
    calculate_p2p_map,
    compute_shape_difference,
)


class TestFunctionalMaps(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.FM = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.mock_model.get_p2p.return_value = np.array([2, 0, 1])
        self.mock_model.D_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.mock_model.D_c = np.array([[0.5, 0.6], [0.7, 0.8]])

    def test_calculate_functional_maps_uses_config_fit_params(self):
        config = FunctionalMapConfig(
            w_descr=2.0,
            w_lap=0.2,
            w_dcomm=0.3,
            w_orient=0.4,
            n_cpus=4,
            verbose=False,
        )

        model, fm = calculate_functional_maps(self.mock_model, config)

        self.assertIs(model, self.mock_model)
        npt.assert_array_equal(fm, self.mock_model.FM)
        self.mock_model.fit.assert_called_once_with(
            w_descr=2.0,
            w_lap=0.2,
            w_dcomm=0.3,
            w_orient=0.4,
            verbose=False,
        )
        self.mock_model.get_p2p.assert_not_called()

    def test_calculate_functional_maps_preserves_old_n_cpus_refine_call_style(self):
        model, fm = calculate_functional_maps(self.mock_model, n_cpus=3, refine="icp")

        self.assertIs(model, self.mock_model)
        npt.assert_array_equal(fm, self.mock_model.FM)
        self.mock_model.change_FM_type.assert_called_once_with("classic")
        self.mock_model.icp_refine.assert_called_once_with(n_jobs=3, verbose=True)

    def test_calculate_functional_maps_zoomout_uses_config(self):
        config = FunctionalMapConfig(refine="zoomout", zoomout_nit=5, zoomout_step=2, verbose=False)

        calculate_functional_maps(self.mock_model, config)

        self.mock_model.change_FM_type.assert_called_once_with("classic")
        self.mock_model.zoomout_refine.assert_called_once_with(nit=5, step=2, verbose=False)

    def test_calculate_p2p_map(self):
        result = calculate_p2p_map(self.mock_model, n_cpus=7)

        npt.assert_array_equal(result, np.array([2, 0, 1]))
        self.mock_model.get_p2p.assert_called_once_with(n_jobs=7)

    def test_compute_shape_difference(self):
        d_area, d_conf = compute_shape_difference(self.mock_model)

        self.mock_model.compute_SD.assert_called_once()
        npt.assert_array_equal(d_area, self.mock_model.D_a)
        npt.assert_array_equal(d_conf, self.mock_model.D_c)

    def test_calculate_functional_maps_chem_uses_config_fit_params(self):
        descr1 = np.random.rand(10, 5)
        descr2 = np.random.rand(10, 5)
        config = FunctionalMapConfig(w_descr=2.0, verbose=False)

        fm = calculate_functional_maps_chem(self.mock_model, descr1, descr2, config)

        npt.assert_array_equal(fm, self.mock_model.FM)
        self.mock_model.fit_othdescr.assert_called_once_with(
            descr1,
            descr2,
            w_descr=2.0,
            w_lap=1e-2,
            w_dcomm=1e-1,
            w_orient=0.0,
            verbose=False,
        )
