import unittest
from unittest.mock import MagicMock, patch

from shape_utils.spectral_descr import calculate_spectral_descriptors, distance_WKS

class TestDescriptorFunctions(unittest.TestCase):

    def setUp(self):
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.descr1 = [[1.0, 2.0, 3.0]]
        self.mock_model.descr2 = [[1.5, 2.5, 3.5]]
        self.landmarks = [0, 1, 2]

    @patch("shape_utils.spectral_descr.functional.SpectralDescriptors")
    def test_calculate_spectral_descriptors_calls_preprocess(self, mock_descriptors):
        descr_model = MagicMock()
        descr_model.descr = [[1.0, 2.0, 3.0]]
        mock_descriptors.return_value = descr_model

        result = calculate_spectral_descriptors(
            mesh="mesh",
            kprocess=50,
            n_ev=30,
            ndescr=100,
            step=2,
            landmarks=self.landmarks,
            descr_type="WKS",
        )

        mock_descriptors.assert_called_once_with("mesh")
        descr_model.preprocess_descriptors_mesh.assert_called_once_with(
            n_ev=(30, 30),
            subsample_step=2,
            descr_type="WKS",
            k_process=50,
            n_descr=100,
            landmarks=self.landmarks,
            verbose=True,
        )
        self.assertEqual(result, descr_model.descr)

    def test_distance_WKS_correctness(self):
        wks1 = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        wks2 = [[1.5, 2.5, 3.5], [1.0, 2.0, 3.0]]
        expected = [1.5, 3.0]  # |1-1.5| + |2-2.5| + |3-3.5| = 0.5+0.5+0.5 = 1.5, etc.

        result = distance_WKS(wks1, wks2)
        self.assertEqual(result, expected)

    def test_distance_WKS_handles_zero_division(self):
        wks1 = [[0.0, 1.0]]
        wks2 = [[0.0, 1.0]]
        result = distance_WKS(wks1, wks2)
        self.assertEqual(result, [0.0])  # No error even with zero denom
