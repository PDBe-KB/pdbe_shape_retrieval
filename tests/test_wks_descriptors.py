import unittest
from unittest.mock import MagicMock
from shape_utils.spectral_descr import calculate_descriptors, distance_WKS

class TestDescriptorFunctions(unittest.TestCase):

    def setUp(self):
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.descr1 = [[1.0, 2.0, 3.0]]
        self.mock_model.descr2 = [[1.5, 2.5, 3.5]]
        self.landmarks = [0, 1, 2]

    def test_calculate_descriptors_calls_preprocess(self):
        descr1, descr2 = calculate_descriptors(
            model=self.mock_model,
            kprocess=50,
            n_ev=30,
            ndescr=100,
            step=2,
            landmarks=self.landmarks,
            output_dir='some/path',
            descr_type='WKS'
        )

        self.mock_model.preprocess.assert_called_once()
        self.assertEqual(descr1, self.mock_model.descr1)
        self.assertEqual(descr2, self.mock_model.descr2)

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

