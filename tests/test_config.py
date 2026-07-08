import unittest

from pydantic import ValidationError

from shape_retrieval.config import DenseMeshConfig, FunctionalMapConfig


class TestFunctionalMapConfig(unittest.TestCase):
    def test_fit_params_contains_only_model_weights(self):
        config = FunctionalMapConfig(w_descr=2.0, w_lap=0.2, w_dcomm=0.3, w_orient=0.4)

        self.assertEqual(
            config.fit_params(),
            {
                "w_descr": 2.0,
                "w_lap": 0.2,
                "w_dcomm": 0.3,
                "w_orient": 0.4,
            },
        )

    def test_from_value_preserves_backward_compatible_n_cpus_int(self):
        config = FunctionalMapConfig.from_value(4, refine="icp")

        self.assertEqual(config.n_cpus, 4)
        self.assertEqual(config.refine, "icp")

    def test_rejects_invalid_refine_method(self):
        with self.assertRaises(ValidationError):
            FunctionalMapConfig(refine="bad-method")

    def test_rejects_zero_cpus(self):
        with self.assertRaises(ValidationError):
            FunctionalMapConfig(n_cpus=0)


class TestDenseMeshConfig(unittest.TestCase):
    def test_process_params_adds_n_jobs(self):
        config = DenseMeshConfig(dist_ratio=4.0, verbose=False)

        params = config.process_params(n_cpus=3)

        self.assertEqual(params["dist_ratio"], 4.0)
        self.assertEqual(params["n_jobs"], 3)
        self.assertFalse(params["verbose"])
