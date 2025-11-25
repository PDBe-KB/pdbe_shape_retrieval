import unittest
import numpy as np
import numpy.testing as npt

from shape_utils.similarity_scores import calculate_geodesic_norm_score


class TestGeodesicNormScore(unittest.TestCase):

    def test_identity_matrix(self):
        """
        The identity matrix has eigenvalues = [1, 1, ..., 1]
        log(|1|) = 0  → score = sqrt(sum(0^2)) = 0
        """
        FM = np.eye(3)
        score = calculate_geodesic_norm_score(FM)
        self.assertAlmostEqual(score, 0.0)

    def test_diagonal_matrix(self):
        """
        For a diagonal matrix diag([a, b]),
        eigenvalues = [a, b]
        score = sqrt((log|a|)^2 + (log|b|)^2)
        """
        FM = np.diag([2.0, 0.5])   # eigenvalues = [2, 0.5]
        expected = np.sqrt(np.log(2.0)**2 + np.log(0.5)**2)

        score = calculate_geodesic_norm_score(FM)
        self.assertAlmostEqual(score, expected, places=7)

    def test_arbitrary_matrix(self):
        """
        Check against a NumPy direct calculation for correctness.
        """
        FM = np.array([[3.0, 1.0], 
                       [0.0, 2.0]])  # eigenvalues = [3, 2]

        eigenvals = np.linalg.eigvals(FM)
        expected = np.sqrt(np.sum(np.log(np.abs(np.real(eigenvals))) ** 2))

        score = calculate_geodesic_norm_score(FM)
        self.assertAlmostEqual(score, expected, places=7)

    def test_complex_eigenvalues(self):
        """
        Matrix with complex eigenvalues should use real part in absolute().
        """
        FM = np.array([[0, -1],
                       [1,  0]])  # eigenvalues = ±i

        eigenvals = np.linalg.eigvals(FM)
        eps= 1e-12  # avoid log(0)
        expected = np.sqrt(np.sum(np.log(np.abs(np.real(eigenvals))+ eps)**2))

        score = calculate_geodesic_norm_score(FM)
        self.assertAlmostEqual(score, expected, places=7)

