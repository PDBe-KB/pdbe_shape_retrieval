import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering

from analysis_tools.clustering import (
    compute_clusters,
    compute_partial_scores_matrix_combined,
    compute_partial_scores_matrix_fast,
    compute_scores_sym_matrix,
    compute_scores_sym_matrix_fast,
    compute_volumes_pockets,
    find_optimal_num_clusters,
    get_pairs,
    get_pairs_fast,
    linkage_matrix,
)


class TestClusteringHelpers(unittest.TestCase):
    def test_pair_helpers(self):
        entries = ["a", "b", "c"]

        expected = [("a", "a"), ("a", "b"), ("a", "c"), ("b", "b"), ("b", "c"), ("c", "c")]

        self.assertEqual(get_pairs(entries), expected)
        self.assertEqual(get_pairs_fast(entries), expected)

    def test_score_matrix_builders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            entries = tmp_path / "entries.txt"
            scores = tmp_path / "scores.txt"
            entries.write_text("a\nb\nc\n")
            scores.write_text("a a 0\na b 1\na c 2\nb b 0\nb c 3\nc c 0\n")

            matrix, labels = compute_scores_sym_matrix(scores, entries)
            fast_matrix, fast_labels = compute_scores_sym_matrix_fast(scores, entries)

        expected = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        npt.assert_array_equal(matrix, expected)
        npt.assert_array_equal(fast_matrix, expected)
        self.assertEqual(labels, ["a", "b", "c"])
        self.assertEqual(fast_labels, ["a", "b", "c"])

    def test_partial_score_matrix_filters_missing_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            entries = tmp_path / "entries.txt"
            scores = tmp_path / "scores.txt"
            entries.write_text("a\nb\nc\n")
            scores.write_text("a b 2\na outside 99\n")

            matrix, labels = compute_partial_scores_matrix_fast(scores, entries, fill_value=-1.0)

        expected = np.array([[-1.0, 2.0, -1.0], [2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
        npt.assert_array_equal(matrix, expected)
        self.assertEqual(labels, ["a", "b", "c"])

    def test_combined_score_matrix_uses_volume_distance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            entries = tmp_path / "entries.txt"
            scores = tmp_path / "scores.txt"
            volumes = tmp_path / "volumes.txt"
            entries.write_text("a\nb\n")
            scores.write_text("a a 0\na b 5\nb b 10\n")
            volumes.write_text("a 10\nb 20\n")

            matrix, labels = compute_partial_scores_matrix_combined(
                scores,
                entries,
                volumes_file=volumes,
                w_vol=0.2,
                normalize_spec="max",
            )

        self.assertEqual(labels, ["a", "b"])
        npt.assert_allclose(matrix, np.array([[0.0, 0.55], [0.55, 1.0]]))

    def test_combined_score_matrix_rejects_bad_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            entries = tmp_path / "entries.txt"
            scores = tmp_path / "scores.txt"
            entries.write_text("a\n")
            scores.write_text("a a 0\n")

            with self.assertRaises(ValueError):
                compute_partial_scores_matrix_combined(scores, entries, normalize_spec="bad")

    def test_linkage_and_cluster_helpers(self):
        data = np.array([[0.0], [0.1], [5.0], [5.2]])
        model = AgglomerativeClustering(n_clusters=2, compute_distances=True).fit(data)

        link = linkage_matrix(model)
        self.assertEqual(link.shape, (3, 4))

        clusters, k, link_matrix, threshold = compute_clusters(
            data,
            ["a", "b", "c", "d"],
            cluster,
            no_clusters=2,
        )

        self.assertEqual(k, 2)
        self.assertEqual(link_matrix.shape, (3, 4))
        self.assertGreaterEqual(threshold, 0)
        self.assertEqual(sorted(sum(clusters, [])), ["a", "b", "c", "d"])

    def test_find_optimal_num_clusters_accepts_explicit_k(self):
        model = AgglomerativeClustering(n_clusters=2, compute_distances=True).fit(np.array([[0.0], [1.0], [5.0]]))

        self.assertEqual(find_optimal_num_clusters(model, np.eye(3), k=2), 2)

    def test_compute_volumes_pockets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            volumes = Path(tmpdir) / "volumes.txt"
            volumes.write_text("a 10\nb 20\nc 30\n")

            result = compute_volumes_pockets(volumes, ["b", "a"])

        self.assertEqual(result, [["b", "20"], ["a", "10"]])
