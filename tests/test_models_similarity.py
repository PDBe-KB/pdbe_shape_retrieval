import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import torch

from shape_retrieval.models import NeuralNetworkModel, SimpleEuclideanModel
from shape_retrieval.similarity_scores import (
    get_pairs,
    get_pairs_fast,
    pairs_to_features,
    predict_similarity_zernike,
    read_dataset,
    read_inv,
)


class TestModels(unittest.TestCase):
    def test_simple_euclidean_model_returns_distance_or_similarity(self):
        model = SimpleEuclideanModel()
        inputs_1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        inputs_2 = torch.tensor([[3.0, 4.0], [1.0, 1.0]])

        distances = model(inputs_1, inputs_2, output_dist=True)
        similarities = model(inputs_1, inputs_2, output_dist=False)

        npt.assert_allclose(distances.detach().numpy(), np.array([5.0, 0.0]))
        npt.assert_allclose(
            similarities.detach().numpy(), np.array([1 / 6, 1.0]), rtol=1e-6
        )

    def test_neural_network_model_forward_shapes(self):
        model = NeuralNetworkModel(
            input_dim=2, hidden_dims=[3], fc_dims=[4], extra_feature_dim=1
        )
        model.eval()
        inputs_1 = torch.ones((2, 2))
        inputs_2 = torch.zeros((2, 2))
        extra = torch.ones((2, 1))

        output = model(inputs_1, inputs_2, extra)
        distance_output = model(inputs_1, inputs_2, extra, output_dist=True)

        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(tuple(distance_output.shape), (2, 1))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


class TestSimilarityScoresIO(unittest.TestCase):
    def test_pair_helpers_and_read_inv(self):
        self.assertEqual(get_pairs(["a", "b"]), [("a", "a"), ("a", "b"), ("b", "b")])
        self.assertEqual(
            get_pairs_fast(["a", "b"]), [("a", "a"), ("a", "b"), ("b", "b")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            inv_file = Path(tmpdir) / "a.inv"
            inv_file.write_text("999\n1\n2\n3\n")

            self.assertEqual(read_inv(inv_file), [1.0, 2.0, 3.0])

    def test_read_dataset_fullatom_and_mainchain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "a.inv").write_text("0\n1\n2\n")
            (tmp_path / "a_cacn.inv").write_text("0\n3\n4\n")

            fullatom = read_dataset(tmp_path, ["a"], "fullatom")
            mainchain = read_dataset(tmp_path, ["a"], "mainchain")

        self.assertEqual(fullatom["a"]["_3dzd"], [1.0, 2.0])
        self.assertEqual(mainchain["a"]["_3dzd"], [3.0, 4.0])

    def test_pairs_to_features_returns_tensors(self):
        data = {
            "a": {"_3dzd": [1.0, 2.0]},
            "b": {"_3dzd": [3.0, 4.0]},
        }

        inputs_1, inputs_2 = pairs_to_features([("a", "b"), ("b", "a")], data, data)

        self.assertEqual(tuple(inputs_1.shape), (2, 2))
        self.assertEqual(tuple(inputs_2.shape), (2, 2))
        npt.assert_allclose(inputs_1.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_predict_similarity_zernike_writes_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "a.inv").write_text("0\n1\n2\n")
            (tmp_path / "b.inv").write_text("0\n1\n4\n")

            predict_similarity_zernike(tmp_path, tmp_path, cuda="false")

            output = (tmp_path / "fullatom_prediction.txt").read_text()

        self.assertIn("Query\tTarget\tDis-similarity Probability", output)
        self.assertTrue("a\tb" in output or "b\ta" in output)
