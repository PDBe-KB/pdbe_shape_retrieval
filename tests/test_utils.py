import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from shape_utils.utils import (
    save_data_to_csv,
    save_list_to_csv,
    find_minimum_distance_meshes,
)


class TestUtils(unittest.TestCase):

    # ------------------------------------------------------------
    # save_data_to_csv
    # ------------------------------------------------------------

    def test_save_data_to_csv_valid(self):
        data = np.array([[1, 2], [3, 4]])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.csv")

            save_data_to_csv(data, output_file)

            # Read back the CSV to validate contents
            df = pd.read_csv(output_file, header=None)
            expected = pd.DataFrame([[1, 2], [3, 4]])
            pd.testing.assert_frame_equal(df, expected)

    def test_save_data_to_csv_empty(self):
        data = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.csv")

            result = save_data_to_csv(data, output_file)
            self.assertIsNone(result)  # Should return None

            # File should NOT be created
            self.assertFalse(os.path.exists(output_file))

    # ------------------------------------------------------------
    # save_list_to_csv
    # ------------------------------------------------------------

    def test_save_list_to_csv_valid(self):
        data = [[10, 20, 30], [40, 50, 60]]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.csv")

            save_list_to_csv(data, output_file)

            df = pd.read_csv(output_file, header=None)
            expected = pd.DataFrame(data)
            pd.testing.assert_frame_equal(df, expected)

    def test_save_list_to_csv_empty(self):
        data = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.csv")

            result = save_list_to_csv(data, output_file)
            self.assertIsNone(result)

            self.assertFalse(os.path.exists(output_file))

    # ------------------------------------------------------------
    # find_minimum_distance_meshes
    # ------------------------------------------------------------

    def test_find_minimum_distance_meshes(self):
        # Mock mesh objects with vertlist arrays
        mesh1 = MagicMock()
        mesh2 = MagicMock()

        mesh1.vertlist = np.array([[0, 0, 0], [1, 1, 1]])
        mesh2.vertlist = np.array([[0, 0, 1], [5, 5, 5]])

        # Expected: min distance between [0,0,0] and [0,0,1] = 1
        expected = 1.0

        result = find_minimum_distance_meshes(mesh1, mesh2)

        self.assertAlmostEqual(result, expected, places=6)


