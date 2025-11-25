import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from shape_utils.functional_maps import (
    calculate_functional_maps,
    compute_shape_difference,
    calculate_functional_maps_chem,
)


class TestFunctionalMaps(unittest.TestCase):

    def setUp(self):
        """
        Create a mock model object that behaves like a pyFM model.
        """
        self.mock_model = MagicMock()

        # Fake outputs
        self.mock_model.get_p2p.return_value = np.array([0, 1, 2])
        self.mock_model.FM = np.array([[1, 0], [0, 1]])

        # Shape difference outputs
        self.mock_model.D_a = "area_operator"
        self.mock_model.D_c = "conformal_operator"

    # ------------------------------------------------------------
    # Test calculate_functional_maps()
    # ------------------------------------------------------------

    def test_calculate_functional_maps_no_refine(self):
        p2p, FM = calculate_functional_maps(self.mock_model, n_cpus=2, refine=None)

        # Model.fit() must be called
        self.mock_model.fit.assert_called_once()

        # get_p2p should be called for non-refine mode
        self.mock_model.get_p2p.assert_called_once_with(n_jobs=2)

        # Outputs
        self.assertTrue((p2p == np.array([0, 1, 2])).all())
        self.assertTrue((FM == np.array([[1, 0], [0, 1]])).all())

    def test_calculate_functional_maps_icp(self):
        p2p, FM = calculate_functional_maps(self.mock_model, refine="icp")

        self.mock_model.change_FM_type.assert_called_with("classic")
        self.mock_model.icp_refine.assert_called_once()
        self.mock_model.get_p2p.assert_called()

    def test_calculate_functional_maps_zoomout(self):
        p2p, FM = calculate_functional_maps(self.mock_model, refine="zoomout")

        self.mock_model.change_FM_type.assert_called_with("classic")
        self.mock_model.zoomout_refine.assert_called_once()

        # The returned values should be those from mock_model
        self.assertTrue((FM == self.mock_model.FM).all())

    # ------------------------------------------------------------
    # Test compute_shape_difference()
    # ------------------------------------------------------------

    def test_compute_shape_difference(self):
        D_area, D_conf = compute_shape_difference(self.mock_model)

        # compute_SD should be called
        self.mock_model.compute_SD.assert_called_once()

        self.assertEqual(D_area, "area_operator")
        self.assertEqual(D_conf, "conformal_operator")

    # ------------------------------------------------------------
    # Test calculate_functional_maps_chem()
    # ------------------------------------------------------------

    def test_calculate_functional_maps_chem(self):
        descr1 = np.random.rand(10, 5)
        descr2 = np.random.rand(10, 5)

        FM = calculate_functional_maps_chem(self.mock_model, descr1, descr2)

        self.mock_model.fit_othdescr.assert_called_once()
        self.assertTrue((FM == self.mock_model.FM).all())


from shape_utils.functional_maps import (
    calculate_functional_maps,
    compute_shape_difference,
    calculate_functional_maps_chem,
)


class TestFunctionalMaps(unittest.TestCase):

    def setUp(self):
        """
        Create a mock pyFM model object with numeric matrices.
        """
        self.mock_model = MagicMock()

        # FM is a numeric matrix
        self.mock_model.FM = np.array([[1.0, 0.0], [0.0, 1.0]])

        # p2p result is an integer array
        self.mock_model.get_p2p.return_value = np.array([2, 0, 1])

        # Shape difference operators are matrices
        self.mock_model.D_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.mock_model.D_c = np.array([[0.5, 0.6], [0.7, 0.8]])

    # ------------------------------------------------------------
    # calculate_functional_maps()
    # ------------------------------------------------------------

    def test_calculate_functional_maps_no_refine(self):
        p2p, FM = calculate_functional_maps(self.mock_model, n_cpus=4, refine=None)

        # fit should be called once
        self.mock_model.fit.assert_called_once()

        # p2p should be called
        self.mock_model.get_p2p.assert_called_once_with(n_jobs=4)

        npt.assert_array_equal(p2p, np.array([2, 0, 1]))
        npt.assert_array_equal(FM, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_calculate_functional_maps_icp(self):
        p2p, FM = calculate_functional_maps(self.mock_model, refine="icp")

        self.mock_model.change_FM_type.assert_called_with("classic")
        self.mock_model.icp_refine.assert_called_once()
        self.mock_model.get_p2p.assert_called()

        npt.assert_array_equal(FM, self.mock_model.FM)

    def test_calculate_functional_maps_zoomout(self):
        p2p, FM = calculate_functional_maps(self.mock_model, refine="zoomout")

        self.mock_model.change_FM_type.assert_called_with("classic")
        self.mock_model.zoomout_refine.assert_called_once()

        npt.assert_array_equal(FM, self.mock_model.FM)

    # ------------------------------------------------------------
    # compute_shape_difference()
    # ------------------------------------------------------------

    def test_compute_shape_difference(self):
        D_area, D_conf = compute_shape_difference(self.mock_model)

        # compute_SD must be called exactly once
        self.mock_model.compute_SD.assert_called_once()

        # Compare matrices
        npt.assert_array_equal(D_area, np.array([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_array_equal(D_conf, np.array([[0.5, 0.6], [0.7, 0.8]]))

    # ------------------------------------------------------------
    # calculate_functional_maps_chem()
    # ------------------------------------------------------------

    def test_calculate_functional_maps_chem(self):
        descr1 = np.random.rand(100, 10)
        descr2 = np.random.rand(100, 10)

        FM = calculate_functional_maps_chem(self.mock_model, descr1, descr2)

        # fit_othdescr should be called exactly once with descriptors
        self.mock_model.fit_othdescr.assert_called_once()

        # Returned FM should match the mock model FM
        npt.assert_array_equal(FM, self.mock_model.FM)
