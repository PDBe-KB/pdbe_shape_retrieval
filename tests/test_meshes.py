import os
import numpy as np
import unittest
from unittest.mock import MagicMock
from shape_utils.meshes import  fix_mesh, reduce_resolution_mesh, reconstruct_mesh, compute_center_of_mass, compute_min_dist, compute_centers_dist, remove_until_vertex
import pytest
from unittest.mock import MagicMock, patch
from tempfile import NamedTemporaryFile

class MockMesh:
    def __init__(self):
        self._v = np.array([[0, 0, 0], [1, 1, 1]])
        self._f = np.array([[0, 1, 1]])
        self.center_mass = np.array([0.5, 0.5, 0.5])

    def vertex_matrix(self):
        return self._v

    def face_matrix(self):
        return self._f

    def vertex_number(self):
        return len(self._v)

class MockMeshSet:
    def __init__(self):
        self._mesh = MockMesh()

    def load_new_mesh(self, file):
        pass

    def current_mesh(self):
        return self._mesh

    def add_mesh(self, mesh):
        # create simple mesh wrapper around (points, faces)
        class SimpleMesh:
            def __init__(self, v, f):
                self._v = v
                self._f = f

            def vertex_matrix(self):
                return self._v

            def face_matrix(self):
                return self._f

            def vertex_number(self):
                return len(self._v)

        self._mesh = SimpleMesh(mesh.points, mesh.faces)

    def apply_filter(self, *args, **kwargs):
        pass


class TestMeshes(unittest.TestCase):
    @patch("shape_utils.meshes.ml.MeshSet", return_value=MockMeshSet())
    @patch("shape_utils.meshes.pymeshfix.MeshFix")
    def test_fix_mesh_basic(self,mock_meshfix, mock_ms):
        mock_obj = MagicMock()
        mock_obj.points = np.array([[0,0,0],[1,1,1]])
        mock_obj.faces = np.array([[0,1,1]])
        mock_meshfix.return_value = mock_obj

        v, f = fix_mesh("fake.obj", resolution=0.5)

        assert v.shape[1] == 3
        assert f.shape[1] == 3
        mock_meshfix.assert_called_once()

    def test_reduce_resolution_mesh(self):
        ms = MockMeshSet()
        v, f = reduce_resolution_mesh(ms, resolution=0.5)
        assert isinstance(v, np.ndarray)
        assert isinstance(f, np.ndarray)

    def test_compute_center_of_mass(self):
        mesh = MockMesh()
        center = compute_center_of_mass(mesh)
        assert np.allclose(center, np.array([0.5, 0.5, 0.5]))




    @patch("shape_utils.meshes.trimesh.load_mesh")
    @patch("shape_utils.meshes.KDTree")
    def test_compute_min_dist(self,mock_kdtree, mock_load):
        mesh1 = MagicMock()
        mesh1.vertices = np.array([[0, 0, 0]])
        mesh2 = MagicMock()
        mesh2.vertices = np.array([[1, 0, 0]])

        mock_load.side_effect = [mesh1, mesh2]

        tree = MagicMock()
        tree.query.return_value = (np.array([1.0]), None)
        mock_kdtree.return_value = tree

        d = compute_min_dist("a.obj", "b.obj")
        assert d == 1.0




    @patch("shape_utils.meshes.trimesh.load_mesh")
    def test_compute_centers_dist(self,mock_load):
        mesh1 = MagicMock()
        mesh1.center_mass = np.array([0, 0, 0])
        mesh2 = MagicMock()
        mesh2.center_mass = np.array([3, 4, 0])

        mock_load.side_effect = [mesh1, mesh2]

        d = compute_centers_dist("a.obj", "b.obj")
        assert d == 5.0  # distance between (0,0,0) and (3,4,0)


    
    def test_remove_until_vertex(self):
        content = [
            "junk line\n",
            "! more junk\n",
            "v 1 2 3\n",
            "v 4 5 6\n"
        ]

        with NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write("".join(content))
            tmp_path = tmp.name

        remove_until_vertex(tmp_path)

        with open(tmp_path, "r") as f:
            result = f.read()

        os.remove(tmp_path)

        assert result == "v 1 2 3\nv 4 5 6\n"