import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from analysis_tools import superposition_shapes as superposition


class TestSuperpositionShapes(unittest.TestCase):
    def test_optimal_rotation_translation_recovers_translation(self):
        points = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        translated = points + np.array([[2.0], [3.0], [4.0]])

        rotation, translation = superposition.optimal_rotation_translation(
            points, translated
        )

        npt.assert_allclose(rotation, np.eye(3), atol=1e-7)
        npt.assert_allclose(translation, np.array([[2.0], [3.0], [4.0]]), atol=1e-7)

    def test_calculate_rotation_translation_fixed_uses_one_based_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_file = Path(tmpdir) / "map.csv"
            map_file.write_text("1\n2\n")

            vertices_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            vertices_2 = vertices_1.copy()

            with patch(
                "analysis_tools.superposition_shapes.read_vertices",
                return_value=(vertices_1, vertices_2),
            ):
                rotation, translation = (
                    superposition.calculate_rotation_translation_fixed(
                        "a.obj", "b.obj", map_file
                    )
                )

        npt.assert_allclose(rotation, np.eye(3), atol=1e-7)
        npt.assert_allclose(translation, np.zeros((3, 1)), atol=1e-7)

    def test_calculate_rotation_translation_fixed_rejects_wrong_map_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_file = Path(tmpdir) / "map.csv"
            map_file.write_text("1\n2\n")

            with patch(
                "analysis_tools.superposition_shapes.read_vertices",
                return_value=(np.zeros((3, 3)), np.zeros((3, 3))),
            ):
                with self.assertRaises(ValueError):
                    superposition.calculate_rotation_translation_fixed(
                        "a.obj", "b.obj", map_file
                    )

    def test_compute_aligned_meshes_builds_aligned_objects(self):
        class FakeTriMesh:
            queue = [
                (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([[0, 1, 1]])),
                (np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), np.array([[0, 1, 1]])),
            ]

            def __init__(self, *args, **kwargs):
                if args:
                    self.vertlist, self.facelist = self.queue.pop(0)

        with patch("analysis_tools.superposition_shapes.mesh.TriMesh", FakeTriMesh):
            aligned_1, aligned_2 = superposition.compute_aligned_meshes(
                "a.obj", "b.obj", np.eye(3)
            )

        npt.assert_allclose(
            aligned_1.vertlist.mean(axis=0), aligned_2.vertlist.mean(axis=0)
        )
        npt.assert_array_equal(aligned_1.facelist, np.array([[0, 1, 1]]))
        npt.assert_array_equal(aligned_2.facelist, np.array([[0, 1, 1]]))
